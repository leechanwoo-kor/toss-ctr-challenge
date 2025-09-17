"""
LightGBM 데이터 준비 스크립트
피처 전처리와 데이터 분할을 수행하고 결과를 저장
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from ..models.feature_processor import TreeBasedFeatureProcessor
from ..utils.config import LIGHTGBM_CONFIG


def main():
    print("🌳 Starting LightGBM data preparation...")

    start_time = time.time()

    # 1. 데이터 로드
    print("📁 Loading raw data...")
    all_train = pd.read_parquet("data/train.parquet", engine="pyarrow")
    print(f"Full training data loaded: {all_train.shape}")

    # 전체 데이터 사용 (트리는 큰 데이터에서 더 강력)
    print("🌳 Using full dataset for optimal tree performance")
    print(f"Training data ready: {all_train.shape}")

    # 2. 피처 전처리 (트리 최적화)
    print("🌳 Processing features for LightGBM...")
    processor = TreeBasedFeatureProcessor()

    # 타겟 분리
    X = all_train.drop('clicked', axis=1)
    y = all_train['clicked'].values

    # 피처 전처리 적용
    X_processed = processor.fit_transform(X.copy())

    print(f"Features processed: {X_processed.shape}")
    print(f"Categorical features: {len(processor.categorical_features)}")
    print(f"Numerical features: {len(processor.numerical_features)}")

    # 3. 데이터 분할
    print("✂️ Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Train CTR: {y_train.mean():.4f}, Val CTR: {y_val.mean():.4f}")

    # 4. 메모리 최적화 (int64 -> int32, float64 -> float32)
    print("💾 Optimizing memory usage...")

    for col in X_train.columns:
        if X_train[col].dtype == 'int64':
            X_train[col] = X_train[col].astype('int32')
            X_val[col] = X_val[col].astype('int32')
        elif X_train[col].dtype == 'float64':
            X_train[col] = X_train[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')

    # 5. 데이터 저장
    print("💾 Saving prepared data...")
    os.makedirs('data/processed/lightgbm', exist_ok=True)

    # NumPy 형태로 저장 (LightGBM 최적화)
    np.save('data/processed/lightgbm/X_train.npy', X_train.values)
    np.save('data/processed/lightgbm/X_val.npy', X_val.values)
    np.save('data/processed/lightgbm/y_train.npy', y_train)
    np.save('data/processed/lightgbm/y_val.npy', y_val)

    # 피처명과 프로세서 저장
    with open('data/processed/lightgbm/feature_names.pkl', 'wb') as f:
        pickle.dump(list(X_train.columns), f)

    with open('data/processed/lightgbm/processor.pkl', 'wb') as f:
        pickle.dump(processor, f)

    # 메타데이터 저장
    metadata = {
        'train_shape': X_train.shape,
        'val_shape': X_val.shape,
        'num_categorical': len(processor.categorical_features),
        'num_numerical': len(processor.numerical_features),
        'train_ctr': y_train.mean(),
        'val_ctr': y_val.mean(),
        'config_used': LIGHTGBM_CONFIG.copy()
    }

    with open('data/processed/lightgbm/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    # 메모리 정리
    del X_processed, X_train, X_val, y_train, y_val, all_train
    gc.collect()

    prep_time = time.time() - start_time

    # 파일 크기 정보
    train_size = os.path.getsize('data/processed/lightgbm/X_train.npy') / 1024**3
    val_size = os.path.getsize('data/processed/lightgbm/X_val.npy') / 1024**3

    print("\n" + "="*50)
    print("✅ LightGBM Data Preparation Completed!")
    print("="*50)
    print(f"Preparation time: {prep_time:.2f} seconds")
    print(f"Train samples: {metadata['train_shape'][0]:,}")
    print(f"Val samples: {metadata['val_shape'][0]:,}")
    print(f"Features: {metadata['train_shape'][1]}")
    print(f"Categorical features: {metadata['num_categorical']}")
    print(f"Numerical features: {metadata['num_numerical']}")
    print(f"Train CTR: {metadata['train_ctr']:.4f}")
    print(f"Val CTR: {metadata['val_ctr']:.4f}")

    print(f"\nSaved to: data/processed/lightgbm/")
    print(f"Files created:")
    print(f"  - X_train.npy, X_val.npy")
    print(f"  - y_train.npy, y_val.npy")
    print(f"  - processor.pkl, metadata.pkl")

    print(f"\nFile sizes:")
    print(f"  Train dataset: {train_size:.2f} GB")
    print(f"  Val dataset: {val_size:.2f} GB")
    print(f"  Total: {train_size + val_size:.2f} GB")

    print("\n🚀 Ready for LightGBM training!")


if __name__ == "__main__":
    main()