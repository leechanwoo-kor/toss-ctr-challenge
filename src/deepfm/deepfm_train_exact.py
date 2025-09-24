"""
XGBoost Enhanced와 정확히 동일한 78개 + 시퀀스 9개 피처로 DeepFM 훈련
진정한 Tree vs DeepFM 공정 비교
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
from torch.utils.data import DataLoader, TensorDataset
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.deepfm_core import DeepFM
from src.utils.config import DEEPFM_CONFIG, WANDB_CONFIG
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


class ExactDataLoader:
    """Exact DeepFM 데이터 전용 로더"""

    def __init__(self, data_path, batch_size=4096, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # 수동으로 저장된 데이터 로드
        data = torch.load(data_path, weights_only=False)

        self.categorical_data = data['categorical']
        self.numerical_data = data['numerical']
        self.targets = data['targets']

        self.num_samples = len(self.targets)
        print(f"   Loaded {self.num_samples:,} samples")
        print(f"   Categorical features: {len(self.categorical_data)}")
        print(f"   Numerical shape: {self.numerical_data.shape if self.numerical_data is not None else 'None'}")

    def __iter__(self):
        if self.shuffle:
            # 시드 고정으로 재현 가능한 셔플링
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            indices = torch.randperm(self.num_samples, generator=generator)
        else:
            indices = torch.arange(self.num_samples)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]

            # categorical batch
            batch_categorical = {}
            for col, data in self.categorical_data.items():
                batch_categorical[col] = data[batch_indices]

            # numerical batch
            batch_numerical = None
            if self.numerical_data is not None:
                batch_numerical = self.numerical_data[batch_indices]

            # targets batch
            batch_targets = self.targets[batch_indices]

            yield batch_categorical, batch_numerical, batch_targets

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def load_exact_data():
    """Exact DeepFM 데이터 로드"""
    print("📁 Loading exact DeepFM data...")

    data_dir = 'data/processed/deepfm_exact'

    # 필수 파일 확인
    required_files = ['train_dataset.pt', 'val_dataset.pt', 'metadata.pkl']
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file}. Run 'python -m src.deepfm.deepfm_prepare_data_exact' first.")

    # 메타데이터 로드
    with open(f'{data_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"✅ Metadata loaded:")
    print(f"   Features: {metadata['data_shape']['features']}")
    print(f"   Categorical: {len(metadata['categorical_features'])}")
    print(f"   Numerical: {len(metadata['numerical_features'])}")
    print(f"   Base features: {metadata['comparison_info']['base_features']}")
    print(f"   Sequence features: {metadata['comparison_info']['sequence_features']}")

    # 정확성 검증
    expected_features = 87
    actual_features = metadata['data_shape']['features']
    if actual_features != expected_features:
        raise ValueError(f"Feature count mismatch: expected {expected_features}, got {actual_features}")

    print(f"🎯 Perfect match: {actual_features} features (78 base + 9 sequence)")

    return metadata


def create_exact_deepfm_model(metadata):
    """정확한 DeepFM 모델 생성"""
    print("🧠 Creating exact DeepFM model...")

    feature_dims = metadata['categorical_feature_dims']
    num_numerical_features = len(metadata['numerical_features'])

    print(f"   Categorical features: {len(feature_dims)}")
    print(f"   Categorical dims: {feature_dims}")
    print(f"   Numerical features: {num_numerical_features}")

    # 모델 하이퍼파라미터
    embedding_dim = DEEPFM_CONFIG.get('EMBED_DIM', 16)
    hidden_units = DEEPFM_CONFIG.get('HIDDEN_DIMS', [256, 128])
    dropout = DEEPFM_CONFIG.get('DROPOUT', 0.2)

    # DeepFM 모델 생성 (올바른 파라미터 사용)
    model = DeepFM(
        feature_dims=feature_dims,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        dropout=dropout,
        num_numerical=num_numerical_features
    )

    # GPU 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"   Model created on: {device}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Hidden units: {hidden_units}")
    print(f"   Dropout: {dropout}")


    return model, device


def train_epoch(model, train_loader, criterion, optimizer, device):
    """한 에포크 훈련"""
    model.train()
    total_loss = 0
    total_samples = 0
    batch_count = 0

    for batch_categorical, batch_numerical, batch_targets in train_loader:
        batch_count += 1

        # GPU로 이동
        for key in batch_categorical:
            batch_categorical[key] = batch_categorical[key].to(device)

        if batch_numerical is not None:
            batch_numerical = batch_numerical.to(device)

        batch_targets = batch_targets.to(device).float()

        # 순전파
        optimizer.zero_grad()

        # categorical과 numerical을 올바른 순서로 전달
        outputs = model(batch_categorical, batch_numerical)
        loss = criterion(outputs.squeeze(), batch_targets)

        # 역전파
        loss.backward()

        # 그래디언트 클리핑 (simplified와 동일)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * len(batch_targets)
        total_samples += len(batch_targets)

        if batch_count % 1000 == 0:
            print(f"     Batch {batch_count}: loss = {loss.item():.6f}")

    return total_loss / total_samples


def validate_epoch(model, val_loader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_categorical, batch_numerical, batch_targets in val_loader:
            # GPU로 이동
            for key in batch_categorical:
                batch_categorical[key] = batch_categorical[key].to(device)

            if batch_numerical is not None:
                batch_numerical = batch_numerical.to(device)

            batch_targets = batch_targets.to(device).float()

            # 순전파
            outputs = model(batch_categorical, batch_numerical)
            loss = criterion(outputs.squeeze(), batch_targets)

            total_loss += loss.item() * len(batch_targets)
            total_samples += len(batch_targets)

            # 예측값과 실제값 수집
            predictions = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            targets_np = batch_targets.cpu().numpy()

            all_predictions.extend(predictions)
            all_targets.extend(targets_np)

    # 메트릭 계산
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    metrics = calculate_competition_metrics(all_targets, all_predictions)
    avg_loss = total_loss / total_samples

    return avg_loss, metrics, all_predictions, all_targets


def main():
    print("🎯 Starting Exact DeepFM Training...")
    print("   Goal: Fair comparison with XGBoost Enhanced")
    print("   Features: 78 (XGBoost Enhanced) + 9 (sequence) = 87 total")

    # 시드 고정 (재현 가능한 결과)
    seed = DEEPFM_CONFIG.get('SEED', 42)
    seed_everything(seed)
    print(f"🎲 Seed fixed: {seed}")

    start_time = time.time()

    # 1. 데이터 로드
    metadata = load_exact_data()

    # 2. 모델 생성
    model, device = create_exact_deepfm_model(metadata)

    # 3. 데이터 로더 생성
    batch_size = DEEPFM_CONFIG.get('BATCH_SIZE', 4096)

    print(f"\n📊 Creating data loaders (batch size: {batch_size})...")
    train_loader = ExactDataLoader('data/processed/deepfm_exact/train_dataset.pt', batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = ExactDataLoader('data/processed/deepfm_exact/val_dataset.pt', batch_size=batch_size, shuffle=False, seed=seed)

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # 4. 훈련 설정
    learning_rate = DEEPFM_CONFIG.get('LEARNING_RATE', 0.001)
    num_epochs = DEEPFM_CONFIG.get('EPOCHS', 25)  # 수정: NUM_EPOCHS -> EPOCHS
    weight_decay = DEEPFM_CONFIG.get('WEIGHT_DECAY', 1e-5)
    pos_weight = DEEPFM_CONFIG.get('POS_WEIGHT', 51.6)  # 추가: 클래스 불균형 대응
    early_stopping_patience = DEEPFM_CONFIG.get('EARLY_STOPPING', 5)  # 여기로 이동

    # pos_weight 적용 (매우 중요!)
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # AdamW 사용 (simplified와 동일)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"🎯 Training configuration:")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Pos weight: {pos_weight} (class imbalance fix)")
    print(f"   Epochs: {num_epochs}")
    print(f"   Early stopping: {early_stopping_patience}")

    # 5. WandB 초기화
    run_name = f'deepfm-exact-{datetime.now().strftime("%m%d_%H%M")}'

    wandb.init(
        project=WANDB_CONFIG['PROJECT'],
        entity=WANDB_CONFIG.get('ENTITY', 'leechanwoo'),
        name=run_name,
        tags=['deepfm', 'exact', 'tree-comparison', 'fair-fight']
    )

    # 설정 로깅
    wandb.config.update({
        **DEEPFM_CONFIG,
        'model_type': 'DeepFM_Exact',
        'data_type': 'exact_xgboost_compatible',
        'features_total': metadata['data_shape']['features'],
        'features_base': metadata['comparison_info']['base_features'],
        'features_sequence': metadata['comparison_info']['sequence_features'],
        'comparison_baseline': 'XGBoost Enhanced (78 features)',
        'fairness_level': 'EXACT_MATCH'
    })

    # 6. 훈련 루프
    print(f"\n🚀 Starting training: {num_epochs} epochs")
    print("   Baseline to beat: XGBoost Enhanced ~0.355")

    best_val_score = 0
    best_epoch = 0
    patience_counter = 0

    training_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")

        # 훈련
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # 검증
        val_loss, val_metrics, val_predictions, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )

        # 스케줄러 업데이트
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time

        # 결과 출력
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss: {val_loss:.6f}")
        print(f"   Competition Score: {val_metrics['comp_score']:.6f}")
        print(f"   AUC: {val_metrics['auc']:.6f}")
        print(f"   vs XGBoost Enhanced: {val_metrics['comp_score'] - 0.355:+.6f}")
        print(f"   Epoch Time: {epoch_time:.2f}s")

        # WandB 로깅
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_comp_score': val_metrics['comp_score'],
            'val_auc': val_metrics['auc'],
            'val_ap': val_metrics['ap'],
            'val_wll': val_metrics['wll'],
            'vs_xgboost_enhanced': val_metrics['comp_score'] - 0.355,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        })

        # 최고 성능 모델 저장
        if val_metrics['comp_score'] > best_val_score:
            best_val_score = val_metrics['comp_score']
            best_epoch = epoch
            patience_counter = 0

            # 모델 저장
            checkpoint_dir = f'checkpoints/deepfm_exact/{run_name}'
            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pth')

            # 메타데이터 저장
            best_metadata = {
                'model_state': 'best_model.pth',
                'epoch': epoch + 1,
                'val_metrics': val_metrics,
                'data_info': metadata,
                'model_config': {
                    'categorical_feature_dims': metadata['categorical_feature_dims'],
                    'num_numerical_features': len(metadata['numerical_features']),
                    'embed_dim': DEEPFM_CONFIG.get('EMBED_DIM', 16),
                    'hidden_dims': DEEPFM_CONFIG.get('HIDDEN_DIMS', [256, 128]),
                    'dropout': DEEPFM_CONFIG.get('DROPOUT', 0.2)
                },
                'training_info': {
                    'run_name': run_name,
                    'best_epoch': epoch + 1,
                    'best_score': best_val_score,
                    'comparison_type': 'exact_xgboost_enhanced',
                    'fairness_level': 'EXACT_MATCH'
                }
            }

            with open(f'{checkpoint_dir}/model_metadata.pkl', 'wb') as f:
                pickle.dump(best_metadata, f)

            print(f"   💾 New best model saved! Score: {best_val_score:.6f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⏰ Early stopping triggered after {epoch + 1} epochs")
            break

    training_time = time.time() - training_start_time
    total_time = time.time() - start_time

    # 7. 최종 결과
    print("\n" + "="*70)
    print("✅ Exact DeepFM Training Completed!")
    print("="*70)
    print(f"📊 Final Results:")
    print(f"   Best Epoch: {best_epoch + 1}")
    print(f"   Best Competition Score: {best_val_score:.6f}")
    print(f"   Training Time: {training_time:.2f}s")
    print(f"   Total Time: {total_time:.2f}s")

    # 공정한 비교 결과
    xgboost_score = 0.355
    difference = best_val_score - xgboost_score

    print(f"\n🥊 Fair Fight Results:")
    print(f"   XGBoost Enhanced (78 features): {xgboost_score:.6f}")
    print(f"   DeepFM Exact (87 features): {best_val_score:.6f}")
    print(f"   Difference: {difference:+.6f}")
    print(f"   Improvement: {(difference/xgboost_score*100):+.2f}%")

    if difference > 0.005:
        print("   🏆 DeepFM WINS by significant margin!")
    elif difference > 0:
        print("   🏆 DeepFM WINS by small margin!")
    elif difference > -0.005:
        print("   🤝 TIE - No significant difference")
    else:
        print("   🌳 XGBoost WINS - Tree models still superior")

    print(f"\n📁 Model saved: checkpoints/deepfm_exact/{run_name}")

    # WandB 최종 로깅
    wandb.log({
        'final_best_score': best_val_score,
        'final_best_epoch': best_epoch + 1,
        'total_training_time': training_time,
        'xgboost_enhanced_baseline': xgboost_score,
        'improvement_vs_xgboost': difference,
        'improvement_percentage': (difference/xgboost_score*100)
    })

    wandb.finish()

    return best_val_score, best_epoch, difference


if __name__ == "__main__":
    try:
        best_score, best_epoch, improvement = main()
        print(f"\n🎉 Training completed successfully!")
        print(f"Best score: {best_score:.6f} at epoch {best_epoch + 1}")
        print(f"Improvement over XGBoost Enhanced: {improvement:+.6f}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()