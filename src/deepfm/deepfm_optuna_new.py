"""
새로운 독립적 DeepFM 파이프라인을 위한 Optuna 하이퍼파라미터 최적화
완전히 독립적이고 효율적인 최적화
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.deepfm_optimized import OptimizedDeepFM
from src.deepfm.deepfm_processor import DeepFMProcessor
from src.utils.config import WANDB_CONFIG
from src.utils.metrics import calculate_competition_metrics


def seed_everything(seed):
    """모든 랜덤 시드 고정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OptimizedDataLoader:
    """Optuna 최적화를 위한 효율적인 데이터 로더"""

    def __init__(self, data_path, batch_size=4096, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # 데이터 로드 (한 번만)
        print(f"   Loading {data_path}...")
        data = torch.load(data_path, weights_only=False)

        self.categorical_data = data['categorical']
        self.numerical_data = data['numerical']
        self.targets = data['targets']
        self.num_samples = len(self.targets)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            indices = torch.randperm(self.num_samples, generator=generator)
        else:
            indices = torch.arange(self.num_samples)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_categorical = {}
            for col, data in self.categorical_data.items():
                batch_categorical[col] = data[batch_indices]

            batch_numerical = None
            if self.numerical_data is not None:
                batch_numerical = self.numerical_data[batch_indices]

            batch_targets = self.targets[batch_indices]

            yield batch_categorical, batch_numerical, batch_targets

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def train_and_validate(trial, processor, device):
    """하나의 trial을 위한 훈련 및 검증"""

    # 하이퍼파라미터 제안
    params = {
        'embedding_dim': trial.suggest_categorical('embedding_dim', [16, 24, 32, 48, 64]),
        'hidden_units': trial.suggest_categorical('hidden_units', [
            [256, 128],
            [512, 256, 128],
            [512, 256, 128, 64],
            [768, 512, 256],
            [768, 512, 256, 128],
            [1024, 512, 256],
        ]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [1024, 2048, 4096]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'pos_weight': trial.suggest_float('pos_weight', 40.0, 70.0, step=5.0),
    }

    print(f"\n🧪 Trial {trial.number}: Testing hyperparameters...")
    for key, value in params.items():
        print(f"   {key}: {value}")

    # 모델 생성
    model = OptimizedDeepFM(
        feature_dims=processor.categorical_feature_dims,
        default_embedding_dim=params['embedding_dim'],
        hidden_units=params['hidden_units'],
        dropout=params['dropout'],
        num_numerical=len(processor.numerical_features)
    ).to(device)

    # 데이터 로더 생성
    train_loader = OptimizedDataLoader(
        'data/processed/deepfm_new/train_dataset.pt',
        batch_size=params['batch_size'],
        shuffle=True,
        seed=42
    )

    val_loader = OptimizedDataLoader(
        'data/processed/deepfm_new/val_dataset.pt',
        batch_size=params['batch_size'],
        shuffle=False,
        seed=42
    )

    # 최적화 설정
    pos_weight_tensor = torch.tensor([params['pos_weight']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.7, verbose=False
    )

    # 조기 종료 설정
    best_score = 0.0
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    max_epochs = 15  # Optuna에서는 빠른 탐색을 위해 짧게

    # 훈련 루프
    for epoch in range(max_epochs):
        # 훈련
        model.train()
        total_loss = 0
        total_samples = 0

        for batch_categorical, batch_numerical, batch_targets in train_loader:
            # GPU로 이동
            for key in batch_categorical:
                batch_categorical[key] = batch_categorical[key].to(device)

            if batch_numerical is not None:
                batch_numerical = batch_numerical.to(device)

            batch_targets = batch_targets.to(device).float()

            # 순전파
            optimizer.zero_grad()
            outputs = model(batch_categorical, batch_numerical)
            loss = criterion(outputs.squeeze(), batch_targets)

            # 역전파
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_targets)
            total_samples += len(batch_targets)

        train_loss = total_loss / total_samples

        # 검증
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch_categorical, batch_numerical, batch_target in val_loader:
                # GPU로 이동
                for key in batch_categorical:
                    batch_categorical[key] = batch_categorical[key].to(device)

                if batch_numerical is not None:
                    batch_numerical = batch_numerical.to(device)

                batch_target = batch_target.to(device).float()

                # 예측
                outputs = model(batch_categorical, batch_numerical)
                predictions = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                targets_np = batch_target.cpu().numpy()

                val_predictions.extend(predictions)
                val_targets.extend(targets_np)

        # 메트릭 계산
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        metrics = calculate_competition_metrics(val_targets, val_predictions)

        comp_score = metrics['comp_score']
        auc = metrics['auc']

        scheduler.step(train_loss)

        # 로깅 (첫 번째와 마지막 에포크만)
        if epoch == 0 or epoch == max_epochs - 1 or comp_score > best_score:
            print(f"    Epoch {epoch+1:2d}: Loss={train_loss:.4f}, CompScore={comp_score:.6f}, AUC={auc:.4f}")

        # 조기 종료 체크
        if comp_score > best_score:
            best_score = comp_score
            best_auc = auc
            patience_counter = 0
        else:
            patience_counter += 1

        # Optuna pruning
        trial.report(comp_score, epoch)
        if trial.should_prune():
            print(f"    Pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if patience_counter >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    print(f"    Final: CompScore={best_score:.6f}, AUC={best_auc:.4f}")

    # 메모리 정리
    del model, train_loader, val_loader, optimizer, scheduler
    torch.cuda.empty_cache()

    return best_score


def objective(trial):
    """Optuna objective function"""
    try:
        # 전역 시드 고정
        seed_everything(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 프로세서 로드 (매번 새로 로드하지 않도록 캐싱 고려)
        if not hasattr(objective, 'processor'):
            objective.processor = DeepFMProcessor.load('data/processed/deepfm_new/processor.pkl')

        return train_and_validate(trial, objective.processor, device)

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0  # 실패한 경우 최소값 반환


def main():
    print("🚀 DeepFM Hyperparameter Optimization with Optuna")
    print("New Clean Independent Architecture")

    # 기본 설정
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 데이터 체크
    data_dir = 'data/processed/deepfm_new'
    if not os.path.exists(f"{data_dir}/train_dataset.pt"):
        raise FileError(f"Data not found: {data_dir}/train_dataset.pt")

    with open(f"{data_dir}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"Data loaded: {metadata['feature_info']['total_features']} features")
    print(f"Target: Beat current best (0.350516) and reach 0.355+")

    # Optuna 스터디 생성
    study_name = f"deepfm-new-optuna-{datetime.now().strftime('%m%d_%H%M')}"

    study = optuna.create_study(
        direction="maximize",  # Competition score 최대화
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=3
        ),
        sampler=optuna.samplers.TPESampler(seed=42)  # 재현성을 위한 시드
    )

    print(f"\n📊 Starting optimization: {study_name}")
    print(f"   Sampler: TPE")
    print(f"   Pruner: MedianPruner")
    print(f"   Trials: 50 (약 3-4시간 예상)")

    # WandB 초기화
    wandb.init(
        project=WANDB_CONFIG['PROJECT'],
        entity=WANDB_CONFIG.get('ENTITY'),
        name=study_name,
        tags=['deepfm', 'optuna', 'hyperopt', 'new-architecture']
    )

    # 최적화 실행
    start_time = time.time()

    def callback(study, trial):
        """각 trial 후 실행되는 콜백"""
        if trial.value is not None:
            wandb.log({
                'trial_number': trial.number,
                'best_score': trial.value,
                'study_best_score': study.best_value,
                **trial.params
            })

            print(f"\n🏆 Trial {trial.number} completed: {trial.value:.6f}")
            print(f"   Current best: {study.best_value:.6f}")

            # 목표 달성 체크
            if trial.value >= 0.355:
                print(f"🎯 Target achieved! Score: {trial.value:.6f}")

    try:
        study.optimize(
            objective,
            n_trials=50,  # 50회 시도
            callbacks=[callback],
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")

    # 결과 분석
    optimization_time = time.time() - start_time

    print("\n" + "="*60)
    print("🎉 Optuna Optimization Complete!")
    print("="*60)

    print(f"📊 Results:")
    print(f"   Best Score: {study.best_value:.6f}")
    print(f"   Improvement: {study.best_value - 0.350516:+.6f}")
    print(f"   Total Trials: {len(study.trials)}")
    print(f"   Optimization Time: {optimization_time:.1f}s ({optimization_time/3600:.1f}h)")

    print(f"\n🏆 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")

    # WandB 최종 로깅
    wandb.log({
        'final_best_score': study.best_value,
        'improvement_vs_baseline': study.best_value - 0.350516,
        'total_trials': len(study.trials),
        'optimization_time': optimization_time,
        **study.best_params
    })

    # 결과 저장
    results_dir = f'checkpoints/deepfm_new/optuna_results'
    os.makedirs(results_dir, exist_ok=True)

    with open(f'{results_dir}/{study_name}_best_params.pkl', 'wb') as f:
        pickle.dump(study.best_params, f)

    print(f"\n📁 Results saved to: {results_dir}/{study_name}_best_params.pkl")
    print(f"🚀 Next step: Train final model with optimized parameters")

    wandb.finish()
    return study.best_value, study.best_params


if __name__ == "__main__":
    try:
        best_score, best_params = main()
        print(f"\n✅ Optimization completed successfully!")
        print(f"🎯 Best Score: {best_score:.6f}")
        print(f"🔧 Use these parameters for final training!")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()