"""
새로운 DeepFMProcessor를 사용한 DeepFM 훈련
완전히 독립적이고 깔끔한 아키텍처
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
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.deepfm_optimized import OptimizedDeepFM
from src.deepfm.deepfm_processor import DeepFMProcessor
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


class DeepFMDataLoader:
    """새로운 DeepFM 데이터 로더"""

    def __init__(self, data_path, batch_size=4096, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # 데이터 로드
        print(f"   Loading data from {data_path}...")
        data = torch.load(data_path, weights_only=False)

        self.categorical_data = data['categorical']
        self.numerical_data = data['numerical']
        self.targets = data['targets']

        self.num_samples = len(self.targets)
        print(f"   Loaded {self.num_samples:,} samples")

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
        outputs = model(batch_categorical, batch_numerical)
        loss = criterion(outputs.squeeze(), batch_targets)

        # 역전파
        loss.backward()
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
    print("🚀 DeepFM Training (New Clean Architecture)")
    print("Using independent DeepFMProcessor - No dependencies!")

    # 시드 고정
    seed = DEEPFM_CONFIG.get('SEED', 42)
    seed_everything(seed)
    print(f"Seed fixed: {seed}")

    start_time = time.time()

    # 1. 메타데이터와 프로세서 로드
    data_dir = "data/processed/deepfm_new"
    print(f"\n📊 Loading DeepFM data from {data_dir}...")

    with open(f"{data_dir}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    processor = DeepFMProcessor.load(f"{data_dir}/processor.pkl")

    print(f"   Features: {metadata['feature_info']['total_features']}")
    print(f"   - Categorical: {metadata['feature_info']['categorical_count']}")
    print(f"   - Numerical: {metadata['feature_info']['numerical_count']}")

    # 2. 모델 생성
    print("\n🧠 Creating DeepFM model...")
    model = OptimizedDeepFM(
        feature_dims=processor.categorical_feature_dims,
        default_embedding_dim=DEEPFM_CONFIG.get('EMBEDDING_DIM', 16),
        hidden_units=DEEPFM_CONFIG.get('HIDDEN_UNITS', [256, 128]),
        dropout=DEEPFM_CONFIG.get('DROPOUT', 0.2),
        num_numerical=len(processor.numerical_features)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model on: {device}")
    print(f"   Total parameters: {total_params:,}")

    # 3. 데이터 로더
    batch_size = DEEPFM_CONFIG.get('BATCH_SIZE', 1024)
    print(f"\n📦 Creating data loaders (batch size: {batch_size})...")

    train_loader = DeepFMDataLoader(f'{data_dir}/train_dataset.pt',
                                   batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = DeepFMDataLoader(f'{data_dir}/val_dataset.pt',
                                 batch_size=batch_size, shuffle=False, seed=seed)

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # 4. 훈련 설정
    learning_rate = DEEPFM_CONFIG.get('LEARNING_RATE', 0.001)
    num_epochs = DEEPFM_CONFIG.get('EPOCHS', 30)
    weight_decay = DEEPFM_CONFIG.get('WEIGHT_DECAY', 1e-5)
    pos_weight = DEEPFM_CONFIG.get('POS_WEIGHT', 51.6)
    early_stopping_patience = DEEPFM_CONFIG.get('EARLY_STOPPING', 5)

    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\n⚙️ Training configuration:")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Pos weight: {pos_weight}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Early stopping: {early_stopping_patience}")

    # 5. WandB 초기화
    run_name = f'deepfm-new-{datetime.now().strftime("%m%d_%H%M")}'
    wandb.init(
        project=WANDB_CONFIG['PROJECT'],
        entity=WANDB_CONFIG.get('ENTITY'),
        name=run_name,
        tags=['deepfm', 'new-architecture', 'clean', 'independent']
    )

    wandb.config.update({
        **DEEPFM_CONFIG,
        'model_type': 'OptimizedDeepFM',
        'processor_type': 'DeepFMProcessor',
        'architecture': 'clean_independent',
        'total_params': total_params,
        'data_source': 'deepfm_new'
    })

    # 6. 훈련 루프
    print(f"\n🚀 Starting training: {num_epochs} epochs")

    best_val_score = 0
    best_epoch = 0
    patience_counter = 0
    training_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # 훈련
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # 검증
        val_loss, val_metrics, val_predictions, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start_time

        # 결과 출력
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss: {val_loss:.6f}")
        print(f"   Competition Score: {val_metrics['comp_score']:.6f}")
        print(f"   AUC: {val_metrics['auc']:.6f}")
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
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        })

        # 최고 성능 모델 저장
        if val_metrics['comp_score'] > best_val_score:
            best_val_score = val_metrics['comp_score']
            best_epoch = epoch
            patience_counter = 0

            # 완전한 체크포인트 저장
            checkpoint_dir = f'checkpoints/deepfm_new/{run_name}'
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'processor': processor,
                'metadata': metadata,
                'config': DEEPFM_CONFIG.copy(),
                'best_comp_score': best_val_score,
                'epoch': best_epoch,
                'run_name': run_name,
                'architecture': 'new_clean'
            }
            torch.save(checkpoint, f'{checkpoint_dir}/best_model.pth')

            print(f"   ✅ New best model saved! Score: {best_val_score:.6f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    training_time = time.time() - training_start_time
    total_time = time.time() - start_time

    # 7. 최종 결과
    print("\n" + "="*70)
    print("🎉 DeepFM Training Complete!")
    print("="*70)
    print(f"📊 Final Results:")
    print(f"   Best Epoch: {best_epoch + 1}")
    print(f"   Best Competition Score: {best_val_score:.6f}")
    print(f"   Training Time: {training_time:.2f}s")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"🏆 Architecture: Clean Independent DeepFM")

    # WandB 최종 로깅
    wandb.log({
        'final_best_score': best_val_score,
        'final_best_epoch': best_epoch + 1,
        'total_training_time': training_time,
        'total_time': total_time
    })

    wandb.finish()
    return best_val_score


if __name__ == "__main__":
    try:
        best_score = main()
        print(f"\n✅ Training completed successfully!")
        print(f"🎯 Best score: {best_score:.6f}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()