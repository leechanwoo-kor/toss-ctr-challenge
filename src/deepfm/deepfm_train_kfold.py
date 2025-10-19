"""
DeepFM with K-Fold Cross Validation for CTR Prediction
새로운 독립적인 DeepFM 아키텍처를 사용한 K-Fold 교차검증
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
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.deepfm_optimized import OptimizedDeepFM
from src.models.deepfm_dynamic_embedding import DynamicEmbeddingDeepFM
from src.deepfm.deepfm_processor import DeepFMProcessor
from src.utils.config import DEEPFM_CONFIG, DEEPFM_DYNAMIC_CONFIG, KFOLD_CONFIG
from src.utils.metrics import calculate_competition_metrics

# Dynamic Embedding 사용 여부 설정
USE_DYNAMIC_EMBEDDING = True  # True: Dynamic Embedding, False: v2.2 Original
CONFIG = DEEPFM_DYNAMIC_CONFIG if USE_DYNAMIC_EMBEDDING else DEEPFM_CONFIG


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


class DeepFMKFoldDataLoader:
    """K-Fold용 효율적인 DeepFM 데이터 로더"""

    def __init__(self, categorical_data, numerical_data, targets, batch_size=1024, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.categorical_data = categorical_data
        self.numerical_data = numerical_data
        self.targets = targets
        self.num_samples = len(targets)

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

            # Categorical batch
            batch_categorical = {}
            for col, data in self.categorical_data.items():
                batch_categorical[col] = data[batch_indices]

            # Numerical batch
            batch_numerical = None
            if self.numerical_data is not None:
                batch_numerical = self.numerical_data[batch_indices]

            # Targets batch
            batch_targets = self.targets[batch_indices]

            yield batch_categorical, batch_numerical, batch_targets

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def load_prepared_deepfm_data():
    """DeepFM 전처리된 데이터 로드"""
    print("Loading prepared DeepFM data...")

    data_dir = "data/processed/deepfm_new"

    # 메타데이터와 프로세서 로드
    with open(f"{data_dir}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    processor = DeepFMProcessor.load(f"{data_dir}/processor.pkl")

    print(f"   Features: {metadata['feature_info']['total_features']}")
    print(f"   - Categorical: {metadata['feature_info']['categorical_count']}")
    print(f"   - Numerical: {metadata['feature_info']['numerical_count']}")

    return metadata, processor, data_dir


def prepare_fold_data(train_indices, val_indices, full_categorical, full_numerical, full_targets):
    """폴드별 데이터 준비"""
    # Train fold data
    train_categorical = {}
    for col, data in full_categorical.items():
        train_categorical[col] = data[train_indices]

    train_numerical = None
    if full_numerical is not None:
        train_numerical = full_numerical[train_indices]

    train_targets = full_targets[train_indices]

    # Val fold data
    val_categorical = {}
    for col, data in full_categorical.items():
        val_categorical[col] = data[val_indices]

    val_numerical = None
    if full_numerical is not None:
        val_numerical = full_numerical[val_indices]

    val_targets = full_targets[val_indices]

    return (train_categorical, train_numerical, train_targets,
            val_categorical, val_numerical, val_targets)


def train_fold(model, train_loader, val_loader, device, fold_num, max_epochs=None):
    """단일 폴드 훈련"""
    print(f"   Training fold {fold_num}...")

    if max_epochs is None:
        max_epochs = CONFIG.get('EPOCHS', 30)

    # 하이퍼파라미터
    learning_rate = CONFIG.get('LEARNING_RATE', 0.001)
    weight_decay = CONFIG.get('WEIGHT_DECAY', 1e-5)
    pos_weight = CONFIG.get('POS_WEIGHT', 51.6)
    early_stopping_patience = CONFIG.get('EARLY_STOPPING', 5)

    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scheduler 설정 (config 기반)
    scheduler_type = CONFIG.get('SCHEDULER_TYPE', 'cosine')
    if scheduler_type == 'cosine':
        # CosineAnnealingLR - 안정적인 단일 코사인 감소
        T_max = CONFIG.get('COSINE_T_MAX', max_epochs)
        eta_min_ratio = CONFIG.get('COSINE_ETA_MIN_RATIO', 0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=learning_rate * eta_min_ratio
        )
    elif scheduler_type == 'cosine_restart':
        # CosineAnnealingWarmRestarts - 주기적 리셋
        T_0 = CONFIG.get('COSINE_T0', 5)
        eta_min_ratio = CONFIG.get('COSINE_ETA_MIN_RATIO', 0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            eta_min=learning_rate * eta_min_ratio
        )
    else:
        # ReduceLROnPlateau (fallback)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.7)

    best_val_score = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        train_samples = 0

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

            train_loss += loss.item() * len(batch_targets)
            train_samples += len(batch_targets)

        train_loss = train_loss / train_samples

        # Validate
        model.eval()
        val_predictions = []
        val_targets_list = []

        with torch.no_grad():
            for batch_categorical, batch_numerical, batch_targets in val_loader:
                # GPU로 이동
                for key in batch_categorical:
                    batch_categorical[key] = batch_categorical[key].to(device)

                if batch_numerical is not None:
                    batch_numerical = batch_numerical.to(device)

                batch_targets = batch_targets.to(device).float()

                # 예측
                outputs = model(batch_categorical, batch_numerical)
                predictions = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                targets_np = batch_targets.cpu().numpy()

                val_predictions.extend(predictions)
                val_targets_list.extend(targets_np)

        # 메트릭 계산
        val_predictions = np.array(val_predictions)
        val_targets_array = np.array(val_targets_list)

        metrics = calculate_competition_metrics(val_targets_array, val_predictions)
        comp_score = metrics['comp_score'] if isinstance(metrics, dict) else metrics

        # Scheduler step (스케줄러 타입에 따라 다르게 처리)
        if scheduler_type in ['cosine', 'cosine_restart']:
            scheduler.step()  # CosineAnnealing 계열은 매 epoch마다
        else:
            scheduler.step(train_loss)  # ReduceLROnPlateau는 loss 기반

        # 로깅 (첫 에포크, 최고 성능 에포크, 5 에포크마다)
        current_lr = optimizer.param_groups[0]['lr']
        if epoch == 0 or comp_score > best_val_score or (epoch + 1) % 5 == 0:
            print(f"     Epoch {epoch+1:2d}: Loss={train_loss:.4f}, CompScore={comp_score:.6f}, LR={current_lr:.6f}")

        # 조기 종료 체크
        if comp_score > best_val_score:
            best_val_score = comp_score
            patience_counter = 0
            # 최고 모델 상태 저장
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"     Early stop at epoch {epoch+1}")
            break

    # 최고 모델 복원
    model.load_state_dict(best_model_state)

    return best_val_score, val_predictions, val_targets_array


def train_deepfm_kfold(metadata, processor, data_dir):
    """DeepFM K-Fold Cross Validation 훈련"""
    print("Starting DeepFM K-Fold Cross Validation...")

    # 전체 훈련 데이터 로드
    print("   Loading full training data...")
    train_data = torch.load(f'{data_dir}/train_dataset.pt', weights_only=False)

    full_categorical = train_data['categorical']
    full_numerical = train_data['numerical']
    full_targets = train_data['targets']

    total_samples = len(full_targets)
    print(f"   Total training samples: {total_samples:,}")

    # K-fold 설정
    kfold = StratifiedKFold(
        n_splits=KFOLD_CONFIG['n_splits'],
        shuffle=KFOLD_CONFIG['shuffle'],
        random_state=KFOLD_CONFIG['random_state']
    )

    # 결과 저장
    fold_results = []
    oof_predictions = np.zeros(total_samples)
    models = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    # 각 폴드 훈련
    for fold, (train_indices, val_indices) in enumerate(kfold.split(full_targets, full_targets)):
        print(f"\nFold {fold+1}/{KFOLD_CONFIG['n_splits']}")

        # 모델 생성 (Dynamic Embedding or v2.2 Original)
        if USE_DYNAMIC_EMBEDDING:
            model = DynamicEmbeddingDeepFM(
                feature_dims=processor.categorical_feature_dims,
                hidden_units=CONFIG.get('HIDDEN_DIMS', [256, 128]),
                dropout=CONFIG.get('DROPOUT', 0.25),
                num_numerical=len(processor.numerical_features),
                min_embed_dim=CONFIG.get('MIN_EMBED_DIM', 4),
                max_embed_dim=CONFIG.get('MAX_EMBED_DIM', 50),
                use_dynamic_embedding=CONFIG.get('USE_DYNAMIC_EMBEDDING', True)
            ).to(device)
        else:
            model = OptimizedDeepFM(
                feature_dims=processor.categorical_feature_dims,
                default_embedding_dim=CONFIG.get('EMBEDDING_DIM', 16),
                hidden_units=CONFIG.get('HIDDEN_DIMS', [256, 128]),
                dropout=CONFIG.get('DROPOUT', 0.2),
                num_numerical=len(processor.numerical_features)
            ).to(device)

        # 폴드 데이터 준비
        train_cat, train_num, train_tgt, val_cat, val_num, val_tgt = prepare_fold_data(
            train_indices, val_indices, full_categorical, full_numerical, full_targets
        )

        # 데이터 로더 생성
        batch_size = CONFIG.get('BATCH_SIZE', 1024)
        train_loader = DeepFMKFoldDataLoader(
            train_cat, train_num, train_tgt, batch_size=batch_size, shuffle=True, seed=42
        )
        val_loader = DeepFMKFoldDataLoader(
            val_cat, val_num, val_tgt, batch_size=batch_size, shuffle=False, seed=42
        )

        print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # 폴드 훈련
        fold_score, fold_predictions, fold_targets = train_fold(
            model, train_loader, val_loader, device, fold + 1
        )

        # 결과 저장
        oof_predictions[val_indices] = fold_predictions
        models.append(model.state_dict().copy())

        fold_results.append({
            'fold': fold + 1,
            'comp_score': fold_score,
            'val_samples': len(val_indices)
        })

        print(f"   Fold {fold+1} Competition Score: {fold_score:.6f}")

        # 누적 평균 계산
        completed_fold_scores = [r['comp_score'] for r in fold_results]
        running_cv_mean = np.mean(completed_fold_scores)
        running_cv_std = np.std(completed_fold_scores) if len(completed_fold_scores) > 1 else 0

        print(f"   Running CV Mean (after {fold+1} folds): {running_cv_mean:.6f} ± {running_cv_std:.6f}")

        # 메모리 정리
        del train_loader, val_loader, model
        torch.cuda.empty_cache()

    return fold_results, oof_predictions, models


def evaluate_kfold_results(full_targets, oof_predictions, fold_results):
    """K-Fold 결과 평가"""
    print("\nEvaluating K-Fold results...")

    # 개별 폴드 결과
    print("\nIndividual Fold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: CompScore={result['comp_score']:.6f}")

    # 교차검증 메트릭
    fold_scores = [r['comp_score'] for r in fold_results]
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    print(f"\nCross-Validation Results:")
    print(f"  CV Competition Score: {cv_mean:.6f} ± {cv_std:.6f}")

    # Out-of-fold 메트릭
    oof_metrics = calculate_competition_metrics(full_targets.numpy(), oof_predictions)
    oof_comp_score = oof_metrics['comp_score'] if isinstance(oof_metrics, dict) else oof_metrics

    print(f"\nOut-of-Fold Metrics:")
    print(f"  OOF Competition Score: {oof_comp_score:.6f}")

    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'oof_comp_score': oof_comp_score,
        'fold_scores': fold_scores
    }


def save_kfold_models(models, oof_predictions, processor, metadata, results, run_name):
    """K-Fold 모델들과 결과 저장"""
    save_dir = f'checkpoints/deepfm_kfold/{run_name}'
    os.makedirs(save_dir, exist_ok=True)

    # 모델들 저장
    for i, model_state in enumerate(models):
        torch.save({
            'model_state_dict': model_state,
            'fold': i + 1,
            'processor': processor,
            'metadata': metadata,
            'config': CONFIG.copy()  # Dynamic or v2.2 config 저장
        }, f'{save_dir}/fold_{i+1}_model.pth')

    # 예측 결과 저장
    np.save(f'{save_dir}/oof_predictions.npy', oof_predictions)

    # 결과 저장
    with open(f'{save_dir}/kfold_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"K-Fold models saved: {save_dir}")
    return save_dir


def main():
    """메인 훈련 함수"""
    model_type = "Dynamic Embedding (v2.2 based)" if USE_DYNAMIC_EMBEDDING else "v2.2 Original"
    print(f"DeepFM K-Fold Cross Validation Training - {model_type}")
    print(f"Using {'DynamicEmbeddingDeepFM' if USE_DYNAMIC_EMBEDDING else 'OptimizedDeepFM'} Architecture")

    # 시드 고정
    seed = CONFIG.get('SEED', 42)
    seed_everything(seed)
    print(f"Seed fixed: {seed}")

    start_time = time.time()

    # 데이터 로드
    metadata, processor, data_dir = load_prepared_deepfm_data()

    # Run name 설정
    arch_tag = 'dynamic-embedding' if USE_DYNAMIC_EMBEDDING else 'v2.2-original'
    run_name = f'deepfm-{arch_tag}-kfold-{datetime.now().strftime("%m%d_%H%M")}'

    # K-Fold 훈련
    print(f"\nStarting {KFOLD_CONFIG['n_splits']}-Fold training...")
    train_start_time = time.time()

    fold_results, oof_predictions, models = train_deepfm_kfold(metadata, processor, data_dir)

    train_time = time.time() - train_start_time

    # 결과 평가
    # 전체 훈련 타겟 로드 (평가용)
    train_data = torch.load(f'{data_dir}/train_dataset.pt', weights_only=False)
    full_targets = train_data['targets']

    final_results = evaluate_kfold_results(full_targets, oof_predictions, fold_results)

    # 모델 저장
    save_dir = save_kfold_models(models, oof_predictions, processor, metadata, final_results, run_name)

    total_time = time.time() - start_time

    # 최종 결과
    print("\n" + "="*70)
    print("DeepFM K-Fold Training Complete!")
    print("="*70)
    print(f"Final Results:")
    print(f"   CV Competition Score: {final_results['cv_mean']:.6f} ± {final_results['cv_std']:.6f}")
    print(f"   OOF Competition Score: {final_results['oof_comp_score']:.6f}")
    print(f"   Best Fold Score: {max(final_results['fold_scores']):.6f}")
    print(f"   Training Time: {train_time:.2f}s ({train_time/60:.1f}min)")
    print(f"   Total Time: {total_time:.2f}s ({total_time/60:.1f}min)")
    print(f"Models saved: {save_dir}")

    if USE_DYNAMIC_EMBEDDING:
        print(f"Architecture: DeepFM v2.2 + Dynamic Embedding (6×card^0.25)")
    else:
        print(f"Architecture: DeepFM v2.2 Original")

    # 목표 달성 체크
    baseline_score = 0.3507  # v2.2 OOF 기준
    target_score = 0.3537    # +0.003 개선 목표 (Dynamic Embedding)

    if final_results['oof_comp_score'] >= target_score:
        improvement = final_results['oof_comp_score'] - baseline_score
        print(f"TARGET ACHIEVED! OOF Score: {final_results['oof_comp_score']:.6f}")
        print(f"   Improvement vs v2.2 baseline: +{improvement:.6f}")
    elif final_results['oof_comp_score'] > baseline_score:
        improvement = final_results['oof_comp_score'] - baseline_score
        remaining = target_score - final_results['oof_comp_score']
        print(f"IMPROVED! OOF Score: {final_results['oof_comp_score']:.6f}")
        print(f"   Improvement vs v2.2: +{improvement:.6f}")
        print(f"   Remaining to target: +{remaining:.6f}")
    else:
        decline = baseline_score - final_results['oof_comp_score']
        print(f"Lower than baseline. OOF Score: {final_results['oof_comp_score']:.6f}")
        print(f"   Decline vs v2.2: -{decline:.6f}")

    return final_results


if __name__ == "__main__":
    try:
        results = main()
        print(f"\nK-Fold training completed successfully!")
        print(f"OOF Competition Score: {results['oof_comp_score']:.6f}")
    except Exception as e:
        print(f"\nK-Fold training failed: {e}")
        import traceback
        traceback.print_exc()