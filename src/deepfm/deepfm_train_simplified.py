"""
간소화된 DeepFM 훈련 스크립트
트리 모델과 동일한 피처를 사용하여 공정한 성능 비교
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prepared_data():
    """간소화된 전처리 데이터 로드"""
    print("📁 Loading simplified prepared data...")

    data_dir = 'data/processed/deepfm_simplified'

    # 데이터 파일 존재 확인
    required_files = ['train_dataset.pt', 'val_dataset.pt', 'metadata.pkl', 'processor.pkl']
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            raise FileNotFoundError(f"Simplified data not found: {file}. Run 'python deepfm_prepare_data_simplified.py' first.")

    # 데이터셋 로드
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
    print(f"Feature dimensions: {metadata['data_shape']['features']}")

    # 피처 그룹 정보 출력
    if 'feature_groups' in metadata:
        groups = metadata['feature_groups']
        print("🎯 Tree-compatible features loaded:")
        for group_name, features in groups.items():
            if features:
                print(f"   - {group_name}: {len(features)} features")

    return train_dataset, val_dataset, metadata, processor



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

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss

        return focal_loss.mean()


def main():
    print("🚀 Starting Simplified DeepFM training...")
    print("   🎯 Using Tree-compatible features for fair comparison")

    # 시드 고정
    seed_everything(DEEPFM_CONFIG['SEED'])

    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 간소화된 전처리 데이터 로드
    start_time = time.time()
    train_dataset, val_dataset, metadata, processor = load_prepared_data()
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")

    # 2. WandB 초기화
    run_name = f"deepfm-simplified-{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project=WANDB_CONFIG['PROJECT'],
        entity=WANDB_CONFIG['ENTITY'],
        name=run_name,
        config={**DEEPFM_CONFIG, **metadata['config_used']},
        tags=['deepfm', 'simplified', 'tree-compatible', 'fair-comparison'],
        notes='Simplified DeepFM training with Tree-model features for fair comparison'
    )

    # 메타데이터 로깅
    wandb.log({
        'data_load_time': load_time,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'total_features': metadata['data_shape']['features'],
        'categorical_features': len(metadata['categorical_features']),
        'numerical_features': len(metadata['numerical_features']),
        'enhancement_type': metadata.get('enhancement_type', 'tree_compatible')
    })

    # 피처 그룹 정보 로깅
    if 'feature_groups' in metadata:
        groups = metadata['feature_groups']
        for group_name, features in groups.items():
            wandb.log({f'features_{group_name}': len(features)})

    # 3. 데이터 로더 생성
    print("🔄 Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEEPFM_CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DEEPFM_CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 4. 모델 초기화
    print("🧠 Initializing Simplified DeepFM model...")

    model = DeepFM(
        feature_dims=metadata['categorical_feature_dims'],
        embedding_dim=DEEPFM_CONFIG['EMBEDDING_DIM'],
        hidden_units=DEEPFM_CONFIG['HIDDEN_DIMS'],
        dropout=DEEPFM_CONFIG['DROPOUT'],
        num_numerical=len(metadata['numerical_features'])
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"🎯 Fair comparison setup:")
    print(f"   - Categorical features: {len(metadata['categorical_features'])} (same as Tree)")
    print(f"   - Numerical features: {len(metadata['numerical_features'])} (EDA + sequence)")
    print(f"   - Feature engineering: Identical to Tree model")

    # 5. 손실 함수 및 옵티마이저 설정 (일반 BCE Loss 사용)
    pos_weight = torch.tensor([DEEPFM_CONFIG['POS_WEIGHT']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("🎯 Tree-compatible BCE Loss:")
    print(f"  Pos weight: {DEEPFM_CONFIG['POS_WEIGHT']:.1f} (class imbalance)")
    print(f"  Loss type: BCEWithLogitsLoss (simplified from Focal Loss)")
    print(f"  Feature parity: Same as Tree model for fair comparison")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=DEEPFM_CONFIG['LEARNING_RATE'],
        weight_decay=DEEPFM_CONFIG['WEIGHT_DECAY']
    )

    # ReduceLROnPlateau: CTR 예측에서 표준적으로 사용
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.7
    )

    # 6. 훈련
    print("🏃 Starting tree-compatible training...")
    best_score = 0
    patience_counter = 0

    # 체크포인트 디렉토리 생성
    checkpoint_dir = f"checkpoints/deepfm_simplified/deepfm-simplified-{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(DEEPFM_CONFIG['EPOCHS']):
        # 훈련
        model.train()
        train_loss = 0
        train_steps = 0

        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (categorical_features, numerical_features, targets) in enumerate(train_pbar):
            categorical_features = {k: v.to(device) for k, v in categorical_features.items()}
            numerical_features = numerical_features.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(categorical_features, numerical_features).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            # 진행률 업데이트
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_train_loss = train_loss / train_steps

        # 검증
        model.eval()
        val_loss = 0
        val_steps = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for categorical_features, numerical_features, targets in val_pbar:
                categorical_features = {k: v.to(device) for k, v in categorical_features.items()}
                numerical_features = numerical_features.to(device)
                targets = targets.to(device).float()

                outputs = model(categorical_features, numerical_features).squeeze()
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_steps += 1

                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

        avg_val_loss = val_loss / val_steps
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        # 간단한 sigmoid 적용 (Temperature Scaling 제거)
        probabilities = torch.sigmoid(all_outputs).numpy()
        metrics = calculate_competition_metrics(all_targets.numpy(), probabilities)
        competition_score = metrics['comp_score']

        # AUC 계산
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_score = roc_auc_score(all_targets.numpy(), probabilities)
        ap_score = average_precision_score(all_targets.numpy(), probabilities)

        # WLL 계산 (Weighted Log Loss)
        epsilon = 1e-15
        probabilities_clipped = np.clip(probabilities, epsilon, 1 - epsilon)
        pos_weight = 51.6  # from config
        wll = -(pos_weight * all_targets.numpy() * np.log(probabilities_clipped) +
                (1 - all_targets.numpy()) * np.log(1 - probabilities_clipped)).mean()

        # 결과 출력
        print(f"Epoch {epoch + 1}/{DEEPFM_CONFIG['EPOCHS']}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Val AUC: {auc_score:.6f}")
        print(f"  Competition Score: {competition_score:.6f}")
        print(f"  AP: {ap_score:.6f}, WLL: {wll:.6f}")
        print(f"  Time: {time.time() - start_time:.2f}s")

        # WandB 로깅
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_auc': auc_score,
            'competition_score': competition_score,
            'average_precision': ap_score,
            'weighted_log_loss': wll,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # 모델 저장
        if competition_score > best_score:
            best_score = competition_score
            patience_counter = 0

            # 모델 저장
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_score': best_score,
                'config': DEEPFM_CONFIG,
                'metadata': metadata
            }, f"{checkpoint_dir}/best_model.pth")

            print(f"  💾 New best model saved: {checkpoint_dir}/best_model.pth")
            print(f"      Competition Score: {competition_score:.6f} (AUC: {auc_score:.6f})")
        else:
            patience_counter += 1

        # 학습률 스케줄링 (ReduceLROnPlateau)
        scheduler.step(competition_score)

        # Early stopping
        if patience_counter >= DEEPFM_CONFIG['PATIENCE']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("✅ Simplified DeepFM training completed!")
    print(f"🏆 Best Competition Score: {best_score:.6f}")
    print(f"🎯 Fair comparison ready - compare with Tree model performance")

    wandb.log({'final_best_score': best_score})
    wandb.finish()

    return best_score


if __name__ == "__main__":
    best_score = main()