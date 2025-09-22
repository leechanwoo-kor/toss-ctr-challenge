"""
DeepFM 향상된 훈련 스크립트 (DeepFM 특화 피처 사용)
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
    """Enhanced 전처리된 데이터 로드"""
    print("📁 Loading enhanced prepared data...")

    data_dir = 'data/processed/deepfm_enhanced'

    # 데이터 파일 존재 확인
    required_files = ['train_dataset.pt', 'val_dataset.pt', 'metadata.pkl', 'processor.pkl']
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            raise FileNotFoundError(f"Enhanced data not found: {file}. Run 'python deepfm_prepare_data_enhanced.py' first.")

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

    # 향상된 피처 정보 출력
    if 'feature_importance_hints' in metadata:
        hints = metadata['feature_importance_hints']
        print("💡 Enhanced features loaded:")
        for category, features in hints.items():
            if features:
                print(f"   - {category}: {len(features)} features")

    return train_dataset, val_dataset, metadata, processor


def temperature_scaling(logits, labels):
    """온도 스케일링을 통한 보정"""
    print("🌡️ Finding optimal temperature...")

    # 온도 후보값들
    temperatures = np.arange(0.1, 5.1, 0.02)
    best_score = -1
    best_temp = 1.0

    for temp in temperatures:
        scaled_probs = torch.sigmoid(logits / temp).cpu().numpy()
        metrics = calculate_competition_metrics(labels, scaled_probs)
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
    print("🚀 Starting Enhanced DeepFM training...")
    print("   🔧 Using DeepFM-optimized features")

    # 시드 고정
    seed_everything(DEEPFM_CONFIG['SEED'])

    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Enhanced 전처리된 데이터 로드
    start_time = time.time()
    train_dataset, val_dataset, metadata, processor = load_prepared_data()
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")

    # 2. WandB 초기화
    run_name = f"deepfm-enhanced-{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project=WANDB_CONFIG['PROJECT'],
        entity=WANDB_CONFIG['ENTITY'],
        name=run_name,
        config={**DEEPFM_CONFIG, **metadata['config_used']},
        tags=['deepfm', 'enhanced', 'training', 'prepared-data'],
        notes='Enhanced DeepFM training with optimized features'
    )

    # 메타데이터 로깅
    wandb.log({
        'data_load_time': load_time,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'total_features': metadata['data_shape']['features'],
        'categorical_features': len(metadata['categorical_features']),
        'numerical_features': len(metadata['numerical_features']),
        'enhancement_type': metadata.get('enhancement_type', 'deepfm_optimized')
    })

    # Enhanced 피처 정보 로깅
    if 'feature_importance_hints' in metadata:
        hints = metadata['feature_importance_hints']
        for category, features in hints.items():
            wandb.log({f'features_{category}': len(features)})

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
    print("🧠 Initializing Enhanced DeepFM model...")

    # Enhanced 설정 적용
    model_config = DEEPFM_CONFIG.copy()

    # 피처 차원 정보가 있으면 활용
    if 'feature_dims' in metadata and metadata['feature_dims']:
        print("💡 Using optimized embedding dimensions")
        feature_dims = metadata['feature_dims']
        # 임베딩 차원 평균 계산 (동적 조정)
        avg_embed_dim = np.mean([info['embed_dim'] for info in feature_dims.values()])
        model_config['EMBED_DIM'] = max(8, min(50, int(avg_embed_dim)))
        print(f"   Optimized embedding dimension: {model_config['EMBED_DIM']}")

    model = DeepFM(
        feature_dims=metadata['categorical_feature_dims'],
        embedding_dim=model_config['EMBED_DIM'],
        hidden_units=model_config['HIDDEN_DIMS'],
        dropout=model_config['DROPOUT'],
        num_numerical=len(metadata['numerical_features'])
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 5. 손실 함수 및 옵티마이저 설정
    pos_weight = torch.tensor([DEEPFM_CONFIG['POS_WEIGHT']]).to(device)
    criterion = FocalLoss(
        alpha=DEEPFM_CONFIG['FOCAL_ALPHA'],
        gamma=DEEPFM_CONFIG['FOCAL_GAMMA'],
        pos_weight=pos_weight
    )

    print("🎯 Enhanced CTR-optimized Focal Loss:")
    print(f"  Alpha: {DEEPFM_CONFIG['FOCAL_ALPHA']} (enhanced positive class focus)")
    print(f"  Gamma: {DEEPFM_CONFIG['FOCAL_GAMMA']} (hard example focusing)")
    print(f"  Pos weight: {DEEPFM_CONFIG['POS_WEIGHT']:.1f} (address class imbalance)")
    print(f"  Enhanced feature engineering for better precision/recall")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=DEEPFM_CONFIG['LEARNING_RATE'],
        weight_decay=DEEPFM_CONFIG['WEIGHT_DECAY']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.5
    )

    # 6. 훈련
    print("🏃 Starting enhanced training...")
    best_score = 0
    patience_counter = 0

    # 체크포인트 디렉토리 생성
    checkpoint_dir = f"checkpoints/deepfm_enhanced/deepfm-enhanced-{datetime.now().strftime('%m%d_%H%M')}"
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

        # 온도 스케일링 적용
        optimal_temp, competition_score = temperature_scaling(all_outputs, all_targets.numpy())

        if competition_score > best_score:
            temperature_info = f"🌡️ Temperature scaling improved: {best_score:.6f} → {competition_score:.6f} (temp={optimal_temp:.3f})"
            print(f"  {temperature_info}")

        # 메트릭 계산
        probabilities = torch.sigmoid(all_outputs / optimal_temp).numpy()
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
            'optimal_temperature': optimal_temp,
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
                'optimal_temperature': optimal_temp,
                'config': model_config,
                'metadata': metadata
            }, f"{checkpoint_dir}/best_model.pth")

            print(f"  💾 New best model saved: {checkpoint_dir}/best_model.pth")
            print(f"      Competition Score: {competition_score:.6f} (AUC: {auc_score:.6f})")
        else:
            patience_counter += 1

        # 학습률 스케줄링
        scheduler.step(competition_score)

        # Early stopping
        if patience_counter >= DEEPFM_CONFIG['PATIENCE']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("✅ Enhanced DeepFM training completed!")
    print(f"🏆 Best Competition Score: {best_score:.6f}")

    wandb.log({'final_best_score': best_score})
    wandb.finish()


if __name__ == "__main__":
    main()