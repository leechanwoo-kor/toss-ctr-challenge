"""
간소화된 DeepFM 데이터 준비 스크립트
트리 모델과 동일한 피처를 사용하여 공정한 비교 수행
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split

from ..models.simplified_deepfm_processor import SimplifiedDeepFMProcessor
from ..models.deepfm_core import DeepFMDataset
from ..utils.config import DEEPFM_CONFIG


def main():
    print("🚀 Starting Simplified DeepFM data preparation...")
    print("   🎯 Using Tree-model compatible features for fair comparison")

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
        print("🔥 Using full dataset for simplified feature engineering")

    print(f"Training data ready: {all_train.shape}")

    # 2. 간소화된 피처 전처리 (트리 모델과 동일)
    print("🔧 Processing features with Tree-compatible approach...")
    processor = SimplifiedDeepFMProcessor()
    processed_df = processor.fit_transform(all_train.copy())

    print(f"Features processed: {processed_df.shape}")
    print(f"Categorical features: {len(processor.categorical_features)}")
    print(f"Numerical features: {len(processor.numerical_features)}")

    # 피처 그룹 정보 출력
    feature_groups = processor.get_feature_importance_groups()
    print("📊 Feature groups (Tree-compatible):")
    for group_name, features in feature_groups.items():
        if features:
            print(f"   {group_name}: {len(features)} features")

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

    # 범주형 피처 차원 정보 준비
    categorical_features = processor.categorical_features
    categorical_feature_dims = {}

    for feat in categorical_features:
        if feat in X_train.columns:
            unique_vals = pd.concat([X_train[feat], X_val[feat]]).nunique()
            categorical_feature_dims[feat] = unique_vals + 1  # +1 for unknown

    print(f"Categorical feature dimensions: {len(categorical_feature_dims)}")

    # Dataset 생성 (target이 포함된 전체 DataFrame 사용)
    # 훈련/검증 데이터에 target 추가
    X_train_with_target = X_train.copy()
    X_train_with_target['clicked'] = y_train

    X_val_with_target = X_val.copy()
    X_val_with_target['clicked'] = y_val

    train_dataset = DeepFMDataset(
        X_train_with_target, processor, target_col='clicked', has_target=True
    )
    val_dataset = DeepFMDataset(
        X_val_with_target, processor, target_col='clicked', has_target=True
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # 6. 메타데이터 저장
    metadata = {
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
        'feature_groups': feature_groups,
        'enhancement_type': 'tree_compatible',
        'comparison_info': {
            'purpose': 'Fair comparison with Tree models',
            'features_used': 'Same as EnhancedTreeFeatureProcessor',
            'categorical_count': len(categorical_features),
            'numerical_count': len(processor.numerical_features)
        }
    }

    # 7. 데이터 저장
    save_dir = "data/processed/deepfm_simplified"
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

    print("✅ Simplified DeepFM data preparation completed!")
    print(f"📊 Processing summary:")
    print(f"   - Total time: {total_time:.2f} seconds")
    print(f"   - Features created: {X.shape[1]}")
    print(f"   - Categorical features: {len(categorical_features)} (same as Tree)")
    print(f"   - Numerical features: {len(processor.numerical_features)} (EDA + sequence)")
    print(f"   - Train samples: {len(train_dataset):,}")
    print(f"   - Val samples: {len(val_dataset):,}")
    print(f"📁 Data saved to: {save_dir}")

    # 비교 정보 출력
    print("\n🎯 Tree-DeepFM comparison setup:")
    print(f"   - Purpose: Fair comparison between Tree and DeepFM models")
    print(f"   - Feature parity: Using identical feature engineering")
    print(f"   - Only difference: Model architecture (Tree vs FM+Deep)")
    print(f"   - Categorical features: {len(categorical_features)} (no day_of_week)")
    print(f"   - EDA features: {len(feature_groups.get('eda_inventory_features', []) + feature_groups.get('eda_user_features', []))}")
    print(f"   - Sequence features: {len(feature_groups.get('sequence_features', []))}")

    return metadata


if __name__ == "__main__":
    metadata = main()