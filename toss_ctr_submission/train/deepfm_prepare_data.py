"""
새로운 DeepFMProcessor를 사용한 DeepFM 데이터 준비
완전히 독립적이고 깔끔한 아키텍처
"""

import os
import sys
import time
import gc
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.deepfm.deepfm_processor import DeepFMProcessor
from src.utils.config import DEEPFM_CONFIG


def create_deepfm_datasets(X_train, X_val, y_train, y_val, processor, save_dir):
    """DeepFM용 데이터셋 생성 및 저장"""
    print("\nCreating DeepFM datasets...")

    def convert_to_torch_tensors(X, y):
        """DataFrame을 DeepFM 모델용 텐서로 변환"""
        categorical_data = {}

        # 범주형 데이터 변환
        for col in processor.categorical_features:
            if col in X.columns:
                categorical_data[col] = torch.tensor(X[col].values, dtype=torch.long)

        # 수치형 데이터 변환
        numerical_data = None
        if processor.numerical_features:
            numerical_cols = [col for col in processor.numerical_features if col in X.columns]
            if numerical_cols:
                numerical_data = torch.tensor(X[numerical_cols].values, dtype=torch.float32)

        # 타겟 변환
        targets = torch.tensor(y, dtype=torch.float32)

        return categorical_data, numerical_data, targets

    # 훈련 데이터셋 변환
    print("   Converting training data...")
    train_categorical, train_numerical, train_targets = convert_to_torch_tensors(X_train, y_train)

    # 검증 데이터셋 변환
    print("   Converting validation data...")
    val_categorical, val_numerical, val_targets = convert_to_torch_tensors(X_val, y_val)

    # 데이터셋 저장
    os.makedirs(save_dir, exist_ok=True)

    print("   Saving datasets...")
    torch.save({
        'categorical': train_categorical,
        'numerical': train_numerical,
        'targets': train_targets
    }, f"{save_dir}/train_dataset.pt")

    torch.save({
        'categorical': val_categorical,
        'numerical': val_numerical,
        'targets': val_targets
    }, f"{save_dir}/val_dataset.pt")

    print(f"Datasets saved to {save_dir}")
    return train_categorical, train_numerical, train_targets


def main():
    print("DeepFM Data Preparation (New Architecture)")
    print("Using independent DeepFMProcessor - No XGBoost dependency!")

    start_time = time.time()

    try:
        # 1. 데이터 로드
        print("\nLoading raw data...")
        train_data = pd.read_parquet("data/train.parquet", engine="pyarrow")
        print(f"Training data loaded: {train_data.shape}")

        # 샘플링 (필요시)
        sample_size = DEEPFM_CONFIG.get("SAMPLE_SIZE")
        if sample_size and sample_size < len(train_data):
            print(f"Sampling to {sample_size:,} rows...")
            train_data = train_data.sample(
                n=sample_size,
                random_state=DEEPFM_CONFIG.get("SEED", 42)
            )
            print(f"Sampled data shape: {train_data.shape}")

        # 2. 프로세서 생성 및 데이터 처리
        print("\nCreating and fitting DeepFMProcessor...")
        processor = DeepFMProcessor()
        processed_df = processor.fit_transform(train_data)

        print(f"Data processing complete!")
        print(f"   Original shape: {train_data.shape}")
        print(f"   Processed shape: {processed_df.shape}")

    except Exception as e:
        raise RuntimeError(f"Data processing failed: {e}") from e

    # 메모리 정리
    del train_data
    gc.collect()

    # 3. 타겟과 피처 분리
    print("\nSeparating features and target...")
    if 'clicked' not in processed_df.columns:
        raise ValueError("Target column 'clicked' not found!")

    X = processed_df.drop("clicked", axis=1)
    y = processed_df["clicked"].values

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"CTR: {y.mean():.4f}")

    # 4. 훈련/검증 분할
    print("\nSplitting data...")
    test_size = DEEPFM_CONFIG.get("TEST_SIZE", 0.2)
    random_seed = DEEPFM_CONFIG.get("SEED", 42)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Train CTR: {y_train.mean():.4f}, Val CTR: {y_val.mean():.4f}")

    # 메모리 정리
    del processed_df, X, y
    gc.collect()

    # 5. DeepFM 텐서 데이터셋 생성
    save_dir = "data/processed/deepfm_new"
    train_categorical, train_numerical, train_targets = create_deepfm_datasets(
        X_train, X_val, y_train, y_val, processor, save_dir
    )

    # 6. 메타데이터 생성
    print("\nCreating metadata...")
    metadata = {
        'data_shape': {
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'features': len(processor.categorical_features) + len(processor.numerical_features)
        },
        'categorical_features': processor.categorical_features,
        'numerical_features': processor.numerical_features,
        'categorical_feature_dims': processor.categorical_feature_dims,
        'feature_info': {
            'total_features': len(processor.categorical_features) + len(processor.numerical_features),
            'categorical_count': len(processor.categorical_features),
            'numerical_count': len(processor.numerical_features)
        },
        'config': DEEPFM_CONFIG.copy(),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # 7. 메타데이터와 프로세서 저장
    with open(f"{save_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)

    processor.save(f"{save_dir}/processor.pkl")

    # 8. 최종 결과
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("DeepFM Data Preparation Complete!")
    print("="*70)
    print(f"Data Summary:")
    print(f"   Training samples: {len(y_train):,}")
    print(f"   Validation samples: {len(y_val):,}")
    print(f"   Total features: {metadata['feature_info']['total_features']}")
    print(f"   - Categorical: {metadata['feature_info']['categorical_count']}")
    print(f"   - Numerical: {metadata['feature_info']['numerical_count']}")
    print(f"Processing time: {total_time:.1f}s")
    print(f"Saved to: {save_dir}/")
    print("Ready for DeepFM training!")

    return metadata


if __name__ == "__main__":
    try:
        metadata = main()
        print(f"\nSuccess! DeepFM data preparation completed.")
        print(f"Next step: Run DeepFM training")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()