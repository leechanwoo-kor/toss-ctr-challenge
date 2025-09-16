"""
DeepFM 전체 데이터 하이퍼파라미터 최적화 스크립트 (전처리된 데이터 사용)
정확한 최적화를 위해 전체 10.7M 데이터 사용
"""

import os
import gc
import time
import random
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import optuna
import wandb

from src.models.deepfm_core import DeepFM
from src.utils.metrics import calculate_competition_metrics


def seed_everything(seed):
    """시드 고정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 전역 변수로 데이터 캐싱
cached_data = None


def load_prepared_data():
    """전처리된 데이터 로드 (캐싱)"""
    global cached_data

    if cached_data is None:
        print("📁 Loading and caching prepared data...")

        data_dir = 'data/processed/deepfm'

        # 데이터셋 로드
        train_dataset = torch.load(f'{data_dir}/train_dataset.pt')
        val_dataset = torch.load(f'{data_dir}/val_dataset.pt')

        # 메타데이터 로드
        with open(f'{data_dir}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        cached_data = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'metadata': metadata
        }

        print(f"Data cached: Train {len(train_dataset)}, Val {len(val_dataset)}")

    return cached_data


def train_epoch_fast(model, dataloader, criterion, optimizer, device):
    """빠른 훈련 (체크포인트 없음)"""
    model.train()
    total_loss = 0
    num_batches = 0

    for categorical_inputs, numerical_inputs, targets in dataloader:
        # GPU로 이동
        categorical_inputs = {k: v.to(device, non_blocking=True) for k, v in categorical_inputs.items()}
        numerical_inputs = numerical_inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 순전파
        outputs = model(categorical_inputs, numerical_inputs)
        loss = criterion(outputs, targets.float())

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_fast(model, dataloader, device):
    """빠른 검증"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for categorical_inputs, numerical_inputs, targets in dataloader:
            # GPU로 이동
            categorical_inputs = {k: v.to(device, non_blocking=True) for k, v in categorical_inputs.items()}
            numerical_inputs = numerical_inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 예측
            outputs = model(categorical_inputs, numerical_inputs)

            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    predictions = np.array(all_preds)
    targets = np.array(all_targets)

    # AUC만 계산 (속도 최적화)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(targets, predictions)

    return auc


def objective(trial):
    """Optuna 최적화 목적 함수"""
    print(f"\n🔄 Trial {trial.number} started...")

    # 시드 고정
    seed_everything(42)

    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 하이퍼파라미터 제안 (현재 최적값 중심으로 좁은 범위)
    params = {
        'BATCH_SIZE': trial.suggest_categorical('batch_size', [256, 512, 1024]),  # 2048 제거
        'LEARNING_RATE': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),  # 범위 좁힘
        'EMBEDDING_DIM': trial.suggest_categorical('embedding_dim', [16, 24, 32, 48]),  # 현재 최적값(24) 중심
        'HIDDEN_UNITS': trial.suggest_categorical('hidden_units', [
            [256, 128],
            [512, 256],
            [512, 256, 128],  # 현재 최적값
            [768, 384, 128],
            [1024, 512, 256]
        ]),
        'DROPOUT': trial.suggest_float('dropout', 0.2, 0.5),  # 현재 최적값(0.337) 중심
        'WEIGHT_DECAY': trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True),
        'EPOCHS': 5,  # 전체 데이터이므로 에포크 줄임 (시간 절약)
        'EARLY_STOPPING': 2,  # 전체 데이터는 더 빨리 수렴
    }

    try:
        # 캐시된 데이터 로드
        data = load_prepared_data()
        train_dataset = data['train_dataset']
        val_dataset = data['val_dataset']
        metadata = data['metadata']

        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['BATCH_SIZE'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=params['BATCH_SIZE'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        # 모델 초기화
        model = DeepFM(
            feature_dims=metadata['feature_dims'],
            embedding_dim=params['EMBEDDING_DIM'],
            hidden_units=params['HIDDEN_UNITS'],
            dropout=params['DROPOUT'],
            num_numerical=metadata['num_numerical']
        ).to(device)

        # 옵티마이저 및 손실함수
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['LEARNING_RATE'],
            weight_decay=params['WEIGHT_DECAY']
        )

        # WandB 초기화 (각 trial별)
        run_name = f"trial-{trial.number:03d}-{datetime.now().strftime('%H%M')}"
        wandb.init(
            project='toss-ctr-deepfm',
            name=run_name,
            config=params,
            tags=['optuna', 'deepfm', 'fast-trial'],
            reinit=True
        )

        # 빠른 훈련
        best_auc = 0
        patience_counter = 0

        for epoch in range(params['EPOCHS']):
            # 훈련
            train_loss = train_epoch_fast(model, train_loader, criterion, optimizer, device)

            # 검증
            val_auc = validate_fast(model, val_loader, device)

            print(f"  Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val AUC {val_auc:.4f}")

            # WandB 로깅
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_auc': val_auc
            })

            # Optuna pruning을 위한 intermediate value 보고
            trial.report(val_auc, epoch)

            # Pruning 체크
            if trial.should_prune():
                print(f"    Trial pruned at epoch {epoch+1}")
                wandb.log({'pruned': True, 'pruned_epoch': epoch + 1})
                raise optuna.TrialPruned()

            # 최고 성능 추적
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1

            # 조기 종료
            if patience_counter >= params['EARLY_STOPPING']:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        print(f"Trial {trial.number} - Best AUC: {best_auc:.6f}")

        # WandB 종료
        wandb.log({'best_auc': best_auc})
        wandb.finish()

        # 메모리 정리
        del model, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

        return best_auc

    except optuna.TrialPruned:
        # Pruned trial - 정상적인 종료
        wandb.finish()
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        wandb.finish()
        return 0.5  # 실패 시 최소값 반환


def main():
    print("🚀 Starting DeepFM fast hyperparameter optimization...")

    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 데이터 미리 로드 (캐싱)
    start_time = time.time()
    load_prepared_data()
    load_time = time.time() - start_time
    print(f"Data loaded and cached in {load_time:.2f} seconds")

    # Optuna 스터디 생성
    study_name = "deepfm-fast-hyperopt"
    storage_url = f"sqlite:///optuna_{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,   # 전체 데이터이므로 더 빨리 pruning 시작
            n_warmup_steps=2,     # epoch 2 이후부터 pruning 고려 (전체 데이터는 빠르게 수렴)
            interval_steps=1      # 매 epoch마다 체크
        )
    )

    print(f"Optuna study created: {study_name}")
    print(f"Target: Maximize Validation AUC")

    # 최적화 실행
    start_time = time.time()

    try:
        study.optimize(
            objective,
            n_trials=25,  # 전체 데이터이므로 trial 수 줄임 (더 정확하지만 시간 오래 걸림)
            timeout=3600 * 8,  # 8시간 타임아웃 (전체 데이터이므로 더 길게)
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user")

    optimization_time = time.time() - start_time

    # 결과 분석
    print("\n" + "="*60)
    print("🎯 Fast Hyperparameter Optimization Results")
    print("="*60)

    print(f"Total trials: {len(study.trials)}")
    print(f"Optimization time: {optimization_time:.2f} seconds")
    print(f"Best AUC: {study.best_value:.6f}")

    print("\n📊 Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 최적 설정 저장
    print("\n💾 Saving optimized configuration...")

    optimized_config = {
        'SAMPLE_SIZE': None,  # 전체 데이터 사용
        'BATCH_SIZE': study.best_params.get('batch_size', 512),
        'EPOCHS': 15,  # 최종 훈련용 에포크
        'LEARNING_RATE': study.best_params.get('learning_rate', 0.001),
        'EMBEDDING_DIM': study.best_params.get('embedding_dim', 24),
        'HIDDEN_UNITS': study.best_params.get('hidden_units', [512, 256, 128]),
        'DROPOUT': study.best_params.get('dropout', 0.3),
        'WEIGHT_DECAY': study.best_params.get('weight_decay', 1e-6),
        'SEED': 42,
        'EARLY_STOPPING': 5,
    }

    os.makedirs('models', exist_ok=True)
    with open('models/deepfm_optimized_config_fast.pkl', 'wb') as f:
        pickle.dump(optimized_config, f)

    print("Optimized config saved to models/deepfm_optimized_config_fast.pkl")

    # 트라이얼 히스토리 저장
    trials_df = study.trials_dataframe()
    trials_df.to_csv('models/deepfm_optuna_trials_fast.csv', index=False)
    print("Trial history saved to models/deepfm_optuna_trials_fast.csv")

    # 상위 5개 트라이얼 출력
    print("\n🏆 Top 5 trials:")
    top_trials = trials_df.nlargest(5, 'value')[['number', 'value', 'params_batch_size',
                                                 'params_learning_rate', 'params_embedding_dim']]
    print(top_trials.to_string(index=False))

    print(f"\n🎉 DeepFM fast optimization completed!")
    print(f"Best validation AUC: {study.best_value:.6f}")
    print(f"Speed improvement: ~{(50 * 12 * 60) / optimization_time:.1f}x faster")

    # 전역 캐시 정리
    global cached_data
    cached_data = None
    gc.collect()


if __name__ == "__main__":
    main()