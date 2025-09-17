"""
DeepFM 빠른 훈련 스크립트 (전처리된 데이터 사용)
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import wandb

from ..models.deepfm_core import DeepFM
from ..utils.metrics import calculate_competition_metrics
from ..utils.config import DEEPFM_CONFIG, WANDB_CONFIG


def seed_everything(seed):
    """시드 고정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prepared_data():
    """전처리된 데이터 로드"""
    print("📁 Loading prepared data...")

    data_dir = 'data/processed/deepfm'

    # 데이터 파일 존재 확인
    required_files = ['train_dataset.pt', 'val_dataset.pt', 'metadata.pkl', 'processor.pkl']
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            raise FileNotFoundError(f"Required file not found: {file}. Run 'python deepfm_prepare_data.py' first.")

    # 데이터셋 로드 (weights_only=False로 설정)
    train_dataset = torch.load(f'{data_dir}/train_dataset.pt', weights_only=False)
    val_dataset = torch.load(f'{data_dir}/val_dataset.pt', weights_only=False)

    # 메타데이터 로드
    with open(f'{data_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # 프로세서 로드
    with open(f'{data_dir}/processor.pkl', 'rb') as f:
        processor = pickle.load(f)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Feature dimensions: {len(metadata['feature_dims'])}")

    return train_dataset, val_dataset, metadata, processor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에포크 훈련"""
    model.train()
    total_loss = 0
    num_batches = 0

    for categorical_inputs, numerical_inputs, targets in tqdm(dataloader, desc="Training"):
        # GPU로 이동
        categorical_inputs = {k: v.to(device) for k, v in categorical_inputs.items()}
        numerical_inputs = numerical_inputs.to(device)
        targets = targets.to(device)

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


def validate_epoch(model, dataloader, criterion, device, temperature=1.0):
    """검증 (Temperature Scaling 포함)"""
    model.eval()
    total_loss = 0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for categorical_inputs, numerical_inputs, targets in tqdm(dataloader, desc="Validation"):
            # GPU로 이동
            categorical_inputs = {k: v.to(device) for k, v in categorical_inputs.items()}
            numerical_inputs = numerical_inputs.to(device)
            targets = targets.to(device)

            # 예측 (logits 저장)
            outputs = model(categorical_inputs, numerical_inputs)
            loss = criterion(outputs, targets.float())

            total_loss += loss.item()
            all_logits.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    logits = np.array(all_logits)
    targets = np.array(all_targets)

    # Temperature Scaling 적용
    scaled_logits = logits / temperature
    predictions = 1 / (1 + np.exp(-scaled_logits))  # sigmoid with temperature

    # 메트릭 계산
    metrics = calculate_competition_metrics(targets, predictions)

    return avg_loss, metrics, predictions, logits


def find_optimal_temperature(logits, targets):
    """최적 Temperature 찾기"""
    print("🌡️ Finding optimal temperature...")

    temperatures = np.linspace(0.5, 5.0, 50)  # 0.5~5.0 범위에서 50개 시도
    best_temp = 1.0
    best_score = 0

    for temp in temperatures:
        # Temperature 적용
        scaled_logits = logits / temp
        calibrated_preds = 1 / (1 + np.exp(-scaled_logits))

        # Competition Score 계산
        metrics = calculate_competition_metrics(targets, calibrated_preds)
        score = metrics['comp_score']

        if score > best_score:
            best_score = score
            best_temp = temp

    print(f"  Optimal temperature: {best_temp:.3f}")
    print(f"  Competition Score improvement: {best_score:.6f}")

    return best_temp, best_score


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples"""

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # BCE with logits
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=self.pos_weight, reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Sigmoid probability
        pt = torch.sigmoid(inputs)
        # For binary classification: pt = p if target=1, else pt = 1-p
        pt = targets * pt + (1 - targets) * (1 - pt)

        # Focal weight: (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss

        return focal_loss.mean()


def main():
    print("🚀 Starting DeepFM training...")

    # 시드 고정
    seed_everything(DEEPFM_CONFIG['SEED'])

    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 전처리된 데이터 로드
    start_time = time.time()
    train_dataset, val_dataset, metadata, processor = load_prepared_data()
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")

    # 2. WandB 초기화
    run_name = f"deepfm-{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project=WANDB_CONFIG['PROJECT'],
        entity=WANDB_CONFIG['ENTITY'],
        name=run_name,
        config={**DEEPFM_CONFIG, **metadata['config_used']},
        tags=['deepfm', 'training', 'prepared-data'],
        notes='DeepFM training with pre-processed data'
    )

    # 메타데이터 로깅
    wandb.log({
        'data_load_time': load_time,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'num_categorical': metadata['num_categorical'],
        'num_numerical': metadata['num_numerical'],
        'train_ctr': metadata['train_ctr'],
        'val_ctr': metadata['val_ctr']
    })

    # 3. 데이터로더 생성
    print("🔄 Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEEPFM_CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=DEEPFM_CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 4. 모델 초기화
    print("🧠 Initializing DeepFM model...")
    model = DeepFM(
        feature_dims=metadata['feature_dims'],
        embedding_dim=DEEPFM_CONFIG['EMBEDDING_DIM'],
        hidden_units=DEEPFM_CONFIG['HIDDEN_UNITS'],
        dropout=DEEPFM_CONFIG['DROPOUT'],
        num_numerical=metadata['num_numerical']
    ).to(device)

    # 5. 옵티마이저 및 손실함수
    # Focal Loss for advanced class imbalance and hard example mining
    pos_weight = torch.tensor([98.1 / 1.9]).to(device)  # ~51.6
    criterion = FocalLoss(
        alpha=0.75,         # CTR용: positive에 더 집중 (0.25→0.75)
        gamma=1.0,          # CTR용: 덜 극단적 focusing (2.0→1.0)
        pos_weight=pos_weight
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=DEEPFM_CONFIG['LEARNING_RATE'],
        weight_decay=DEEPFM_CONFIG['WEIGHT_DECAY']
    )

    print(f"🎯 CTR-optimized Focal Loss:")
    print(f"  Alpha: 0.75 (strong focus on positive class)")
    print(f"  Gamma: 1.0 (moderate hard example focusing)")
    print(f"  Pos weight: {pos_weight.item():.1f} (address 98.1%/1.9% imbalance)")
    print(f"  CTR-tuned parameters for better precision/recall balance")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. 훈련 루프
    print("🏃 Starting training...")
    training_start = time.time()

    best_auc = 0
    best_comp_score = 0  # 대회 점수 추적
    patience_counter = 0
    train_losses = []
    val_aucs = []

    for epoch in range(DEEPFM_CONFIG['EPOCHS']):
        epoch_start = time.time()

        # 훈련
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # 검증 및 Temperature Scaling
        val_loss, val_metrics, val_predictions, val_logits = validate_epoch(model, val_loader, criterion, device)

        # 타겟 데이터 수집 (간단한 방법)
        val_targets = []
        for _, _, targets in val_loader:
            val_targets.extend(targets.cpu().numpy())
        val_targets = np.array(val_targets)

        # Temperature Scaling 최적화
        optimal_temp, calibrated_score = find_optimal_temperature(val_logits, val_targets)

        # 최종 점수 결정
        val_auc = val_metrics['auc']
        original_score = val_metrics['comp_score']

        if calibrated_score > original_score:
            val_comp_score = calibrated_score
            print(f"  🌡️ Temperature scaling improved: {original_score:.6f} → {calibrated_score:.6f} (temp={optimal_temp:.3f})")
        else:
            val_comp_score = original_score
            optimal_temp = 1.0

        val_aucs.append(val_auc)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{DEEPFM_CONFIG['EPOCHS']}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val AUC: {val_auc:.6f}")
        print(f"  Competition Score: {val_comp_score:.6f}")
        print(f"  AP: {val_metrics['ap']:.6f}, WLL: {val_metrics['wll']:.6f}")
        print(f"  Time: {epoch_time:.2f}s")

        # WandB 로깅 (대회 채점 기준 포함)
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_comp_score': val_comp_score,        # 🆕 대회 최종 점수
            'val_ap': val_metrics['ap'],             # 🆕 Average Precision
            'val_wll': val_metrics['wll'],           # 🆕 Weighted LogLoss
            'val_logloss': val_metrics['logloss'],
            'epoch_time': epoch_time
        })

        # 최고 성능 모델 저장 (대회 점수 기준)
        if val_comp_score > best_comp_score:
            best_auc = val_auc
            best_comp_score = val_comp_score
            patience_counter = 0

            # 실험별 모델 저장 디렉토리 생성
            exp_dir = f"models/deepfm/{run_name}"
            os.makedirs(exp_dir, exist_ok=True)
            model_path = f"{exp_dir}/best_model.pth"

            torch.save({
                'model_state_dict': model.state_dict(),
                'config': DEEPFM_CONFIG,
                'metadata': metadata,
                'processor': processor,
                'epoch': epoch + 1,
                'best_auc': best_auc,
                'best_comp_score': best_comp_score,
                'best_metrics': val_metrics,
                'timestamp': datetime.now().isoformat()
            }, model_path)

            print(f"  💾 New best model saved: {model_path}")
            print(f"      Competition Score: {best_comp_score:.6f} (AUC: {best_auc:.6f})")
        else:
            patience_counter += 1

        # 조기 종료
        if patience_counter >= DEEPFM_CONFIG['EARLY_STOPPING']:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print()

    training_time = time.time() - training_start

    # 7. 최종 결과
    total_time = time.time() - start_time

    print("\n" + "="*50)
    print("🎉 DeepFM Training Completed!")
    print("="*50)
    print(f"Data Load Time: {load_time:.2f}s")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Best Competition Score: {best_comp_score:.6f}")
    print(f"Best Validation AUC: {best_auc:.6f}")
    print(f"Model saved: {model_path}")

    # 성능 비교 정보
    print(f"\n📈 Performance Summary:")
    print(f"Speed improvement: ~{(12*60 + 30) / total_time:.1f}x faster")  # 기존 처리 시간 대비
    print(f"Best Competition Score: {best_comp_score:.6f}")
    print(f"Best AUC achieved: {best_auc:.6f}")

    # 최종 메트릭 로깅
    wandb.log({
        'best_auc': best_auc,
        'best_comp_score': best_comp_score,
        'training_time': training_time,
        'total_time': total_time,
        'epochs_trained': epoch + 1
    })

    wandb.finish()

    # 메모리 정리
    del model, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()