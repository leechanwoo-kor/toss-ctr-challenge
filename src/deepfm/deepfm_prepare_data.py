"""
DeepFM 데이터 준비 스크립트
피처 전처리와 데이터 분할을 수행하고 결과를 저장
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split

from ..models.feature_processor import TrainingFeatureProcessor
from ..models.deepfm_core import DeepFMDataset
from ..utils.config import DEEPFM_CONFIG


def main():
    print("🔧 Starting DeepFM data preparation...")

    start_time = time.time()

    # 1. 데이터 로드
    print("📁 Loading raw data...")
    all_train = pd.read_parquet("data/train.parquet", engine="pyarrow")
    print(f"Full training data loaded: {all_train.shape}")

    # 샘플링 설정 - 전체 데이터 사용 (Optuna 포함)
    sample_size = DEEPFM_CONFIG['SAMPLE_SIZE']

    if sample_size:
        all_train = all_train.sample(n=sample_size, random_state=DEEPFM_CONFIG['SEED'])
        print(f"Sampled to: {all_train.shape}")
    else:
        print("🔥 Using full dataset for accurate hyperparameter optimization")

    print(f"Training data ready: {all_train.shape}")

    # 2. 피처 전처리
    print("🔧 Processing features...")
    processor = TrainingFeatureProcessor()
    processed_df = processor.fit_transform(all_train.copy())

    print(f"Features processed: {processed_df.shape}")
    print(f"Categorical features: {len(processor.categorical_features)}")
    print(f"Numerical features: {len(processor.numerical_features)}")

    # 메모리 정리
    del all_train
    gc.collect()

    # 3. 훈련/검증 분할
    print("✂️ Splitting data...")
    X = processed_df.drop(['clicked'], axis=1)
    y = processed_df['clicked']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=DEEPFM_CONFIG['SEED'], stratify=y
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Train CTR: {y_train.mean():.4f}, Val CTR: {y_val.mean():.4f}")

    # 메모리 정리
    del processed_df, X, y
    gc.collect()

    # 4. DeepFM Dataset 생성
    print("🔄 Creating DeepFM datasets...")

    # 훈련 데이터셋 생성
    train_dataset = DeepFMDataset(
        pd.concat([X_train, y_train], axis=1),
        processor,
        target_col='clicked',
        has_target=True
    )

    # 검증 데이터셋 생성
    val_dataset = DeepFMDataset(
        pd.concat([X_val, y_val], axis=1),
        processor,
        target_col='clicked',
        has_target=True
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # 메모리 정리
    del X_train, X_val, y_train, y_val
    gc.collect()

    # 5. 데이터 저장
    print("💾 Saving prepared data...")

    os.makedirs('data/processed/deepfm', exist_ok=True)

    # 데이터셋 저장
    torch.save(train_dataset, 'data/processed/deepfm/train_dataset.pt')
    torch.save(val_dataset, 'data/processed/deepfm/val_dataset.pt')

    # 프로세서 저장
    with open('data/processed/deepfm/processor.pkl', 'wb') as f:
        pickle.dump(processor, f)

    # 메타데이터 저장
    metadata = {
        'feature_dims': processor.feature_dims,
        'categorical_features': processor.categorical_features,
        'numerical_features': processor.numerical_features,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'train_ctr': float(train_dataset.targets.mean()),
        'val_ctr': float(val_dataset.targets.mean()),
        'num_categorical': len(processor.categorical_features),
        'num_numerical': len(processor.numerical_features),
        'config_used': DEEPFM_CONFIG.copy()
    }

    with open('data/processed/deepfm/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    preparation_time = time.time() - start_time

    # 6. 요약 출력
    print("\n" + "="*50)
    print("✅ DeepFM Data Preparation Completed!")
    print("="*50)
    print(f"Preparation time: {preparation_time:.2f} seconds")
    print(f"Train samples: {metadata['train_size']:,}")
    print(f"Val samples: {metadata['val_size']:,}")
    print(f"Categorical features: {metadata['num_categorical']}")
    print(f"Numerical features: {metadata['num_numerical']}")
    print(f"Train CTR: {metadata['train_ctr']:.4f}")
    print(f"Val CTR: {metadata['val_ctr']:.4f}")

    print(f"\nFeature dimensions:")
    for feat, dim in processor.feature_dims.items():
        print(f"  {feat}: {dim}")

    print(f"\nSaved to: data/processed/deepfm/")
    print("Files created:")
    print("  - train_dataset.pt")
    print("  - val_dataset.pt")
    print("  - processor.pkl")
    print("  - metadata.pkl")

    # 파일 크기 정보
    train_size = os.path.getsize('data/processed/deepfm/train_dataset.pt') / 1024**3
    val_size = os.path.getsize('data/processed/deepfm/val_dataset.pt') / 1024**3
    print(f"\nFile sizes:")
    print(f"  Train dataset: {train_size:.2f} GB")
    print(f"  Val dataset: {val_size:.2f} GB")
    print(f"  Total: {train_size + val_size:.2f} GB")

    print("\n🚀 Ready for DeepFM fast training!")


if __name__ == "__main__":
    main()