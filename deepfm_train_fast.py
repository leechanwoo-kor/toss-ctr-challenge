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
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import wandb

from src.models.deepfm_core import DeepFM
from src.utils.metrics import calculate_competition_metrics
from src.utils.config import DEEPFM_CONFIG, WANDB_CONFIG


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

    # 데이터셋 로드
    train_dataset = torch.load(f'{data_dir}/train_dataset.pt')
    val_dataset = torch.load(f'{data_dir}/val_dataset.pt')

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


def validate_epoch(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for categorical_inputs, numerical_inputs, targets in tqdm(dataloader, desc="Validation"):
            # GPU로 이동
            categorical_inputs = {k: v.to(device) for k, v in categorical_inputs.items()}
            numerical_inputs = numerical_inputs.to(device)
            targets = targets.to(device)

            # 예측
            outputs = model(categorical_inputs, numerical_inputs)
            loss = criterion(outputs, targets.float())

            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.array(all_preds)
    targets = np.array(all_targets)

    # 메트릭 계산
    metrics = calculate_competition_metrics(targets, predictions)

    return avg_loss, metrics, predictions


def main():
    print("🚀 Starting DeepFM fast training...")

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
    run_name = f"deepfm-fast-{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project=WANDB_CONFIG['PROJECT'],
        entity=WANDB_CONFIG['ENTITY'],
        name=run_name,
        config={**DEEPFM_CONFIG, **metadata['config_used']},
        tags=['deepfm', 'fast-training', 'prepared-data'],
        notes='DeepFM fast training with pre-processed data'
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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=DEEPFM_CONFIG['LEARNING_RATE'],
        weight_decay=DEEPFM_CONFIG['WEIGHT_DECAY']
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. 훈련 루프
    print("🏃 Starting training...")
    training_start = time.time()

    best_auc = 0
    patience_counter = 0
    train_losses = []
    val_aucs = []

    for epoch in range(DEEPFM_CONFIG['EPOCHS']):
        epoch_start = time.time()

        # 훈련
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # 검증
        val_loss, val_metrics, val_predictions = validate_epoch(model, val_loader, criterion, device)
        val_auc = val_metrics['auc']
        val_aucs.append(val_auc)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{DEEPFM_CONFIG['EPOCHS']}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val AUC: {val_auc:.6f}")
        print(f"  Time: {epoch_time:.2f}s")

        # WandB 로깅
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_logloss': val_metrics['logloss'],
            'epoch_time': epoch_time
        })

        # 최고 성능 모델 저장
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            # 모델 저장
            os.makedirs('models', exist_ok=True)
            model_path = f'models/deepfm_fast_{datetime.now().strftime("%m%d_%H%M")}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': DEEPFM_CONFIG,
                'metadata': metadata,
                'epoch': epoch + 1,
                'best_auc': best_auc
            }, model_path)

            print(f"  💾 New best model saved: {model_path}")
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
    print("🎉 DeepFM Fast Training Completed!")
    print("="*50)
    print(f"Data Load Time: {load_time:.2f}s")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Best Validation AUC: {best_auc:.6f}")
    print(f"Model saved: {model_path}")

    # 성능 비교 정보
    print(f"\n📈 Performance Summary:")
    print(f"Speed improvement: ~{(12*60 + 30) / total_time:.1f}x faster")  # 기존 처리 시간 대비
    print(f"Best AUC achieved: {best_auc:.6f}")

    # 최종 메트릭 로깅
    wandb.log({
        'best_auc': best_auc,
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