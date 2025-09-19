"""
DeepFM 향상된 데이터 준비 스크립트
DeepFM 특화 피처 엔지니어링을 적용한 데이터 전처리
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split

from ..models.enhanced_deepfm_processor import EnhancedDeepFMFeatureProcessor
from ..models.deepfm_core import DeepFMDataset
from ..utils.config import DEEPFM_CONFIG


def main():
    print("🚀 Starting Enhanced DeepFM data preparation...")
    print("   📊 Using DeepFM-optimized feature engineering")

    start_time = time.time()

    # 1. 데이터 로드
    print("📁 Loading raw data...")
    all_train = pd.read_parquet("data/train.parquet", engine="pyarrow")
    print(f"Full training data loaded: {all_train.shape}")

    # 샘플링 설정
    sample_size = DEEPFM_CONFIG['SAMPLE_SIZE']

    if sample_size:
        all_train = all_train.sample(n=sample_size, random_state=DEEPFM_CONFIG['SEED'])
        print(f"Sampled to: {all_train.shape}")
    else:
        print("🔥 Using full dataset for enhanced feature engineering")

    print(f"Training data ready: {all_train.shape}")

    # 2. Enhanced 피처 전처리
    print("🔧 Processing features with DeepFM enhancements...")
    processor = EnhancedDeepFMFeatureProcessor()
    processed_df = processor.fit_transform(all_train.copy())

    print(f"Features processed: {processed_df.shape}")
    print(f"Categorical features: {len(processor.categorical_features)}")
    print(f"Numerical features: {len(processor.numerical_features)}")

    # 피처 중요도 힌트 출력
    importance_hints = processor.get_feature_importance_hints()
    print("💡 Feature importance hints:")
    for category, features in importance_hints.items():
        if features:
            print(f"   {category}: {len(features)} features")

    # 메모리 정리
    del all_train
    gc.collect()

    # 3. 타겟과 피처 분리
    print("✂️ Separating features and target...")
    X = processed_df.drop('clicked', axis=1)
    y = processed_df['clicked'].values

    print(f"Final feature shape: {X.shape}")
    print(f"CTR: {y.mean():.4f}")

    # 4. 훈련/검증 분할
    print("🔄 Splitting train/validation data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=DEEPFM_CONFIG['SEED'], stratify=y
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Train CTR: {y_train.mean():.4f}, Val CTR: {y_val.mean():.4f}")

    # 5. DeepFM Dataset 생성
    print("🧠 Creating DeepFM datasets...")

    # 범주형 피처 정보 준비
    categorical_features = processor.categorical_features
    categorical_feature_dims = {}

    for feat in categorical_features:
        if feat in X_train.columns:
            unique_vals = pd.concat([X_train[feat], X_val[feat]]).nunique()
            categorical_feature_dims[feat] = unique_vals + 1  # +1 for unknown

    print(f"Categorical feature dimensions: {len(categorical_feature_dims)}")

    # Dataset 생성
    train_dataset = DeepFMDataset(
        X_train, y_train, categorical_features, categorical_feature_dims
    )
    val_dataset = DeepFMDataset(
        X_val, y_val, categorical_features, categorical_feature_dims
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # 6. 메타데이터 저장
    metadata = {
        'processor': processor,
        'categorical_features': categorical_features,
        'categorical_feature_dims': categorical_feature_dims,
        'numerical_features': processor.numerical_features,
        'feature_dims': getattr(processor, 'feature_dims', {}),
        'data_shape': {
            'train': X_train.shape,
            'val': X_val.shape,
            'features': len(X_train.columns)
        },
        'ctr_stats': {
            'train_ctr': y_train.mean(),
            'val_ctr': y_val.mean(),
            'total_ctr': y.mean()
        },
        'config_used': DEEPFM_CONFIG.copy(),
        'feature_importance_hints': importance_hints,
        'enhancement_type': 'deepfm_optimized'
    }

    # 7. 데이터 저장
    save_dir = "data/processed/deepfm_enhanced"
    os.makedirs(save_dir, exist_ok=True)

    print("💾 Saving processed data...")

    # Dataset 저장
    torch.save(train_dataset, f"{save_dir}/train_dataset.pt")
    torch.save(val_dataset, f"{save_dir}/val_dataset.pt")

    # 메타데이터 저장
    with open(f"{save_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # 프로세서 저장
    with open(f"{save_dir}/processor.pkl", "wb") as f:
        pickle.dump(processor, f)

    total_time = time.time() - start_time

    print("✅ Enhanced DeepFM data preparation completed!")
    print(f"📊 Processing summary:")
    print(f"   - Total time: {total_time:.2f} seconds")
    print(f"   - Features created: {X.shape[1]}")
    print(f"   - Categorical features: {len(categorical_features)}")
    print(f"   - Numerical features: {len(processor.numerical_features)}")
    print(f"   - Enhancement features: {sum(len(v) for v in importance_hints.values())}")
    print(f"   - Train samples: {len(train_dataset):,}")
    print(f"   - Val samples: {len(val_dataset):,}")
    print(f"📁 Data saved to: {save_dir}")

    # 피처 분석 정보 출력
    print("\n🔍 Feature analysis:")
    print(f"   - Embedding combinations: {len(importance_hints.get('embedding_friendly', []))}")
    print(f"   - Interaction features: {len(importance_hints.get('interaction_features', []))}")
    print(f"   - Temporal features: {len(importance_hints.get('temporal_features', []))}")
    print(f"   - Popularity features: {len(importance_hints.get('popularity_features', []))}")

    return metadata


if __name__ == "__main__":
    metadata = main()