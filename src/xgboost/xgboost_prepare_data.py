"""
XGBoost 데이터 준비 스크립트
LightGBM과 동일한 전처리된 데이터를 XGBoost 형태로 변환
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from ..models.feature_processor import TrainingFeatureProcessor
from ..utils.config import XGBOOST_CONFIG


def main():
    print("🚀 Starting XGBoost data preparation...")

    start_time = time.time()

    # LightGBM 데이터가 있으면 재사용, 없으면 새로 생성
    lightgbm_data_dir = 'data/processed/lightgbm'
    xgboost_data_dir = 'data/processed/xgboost'

    if os.path.exists(f'{lightgbm_data_dir}/X_train.npy'):
        print("♻️  Reusing LightGBM processed data...")

        # LightGBM 데이터 복사
        os.makedirs(xgboost_data_dir, exist_ok=True)

        # 데이터 파일들 복사
        for file_name in ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy',
                         'feature_names.pkl', 'processor.pkl']:
            src_path = f'{lightgbm_data_dir}/{file_name}'
            dst_path = f'{xgboost_data_dir}/{file_name}'

            if file_name.endswith('.npy'):
                data = np.load(src_path)
                np.save(dst_path, data)
            else:
                with open(src_path, 'rb') as f:
                    data = pickle.load(f)
                with open(dst_path, 'wb') as f:
                    pickle.dump(data, f)

        # 메타데이터 업데이트 (XGBoost config 반영)
        with open(f'{lightgbm_data_dir}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        metadata['config_used'] = XGBOOST_CONFIG.copy()

        with open(f'{xgboost_data_dir}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print(f"✅ Data copied from LightGBM preparation")

    else:
        print("🔧 Creating new processed data...")

        # 1. 데이터 로드
        print("📁 Loading raw data...")
        all_train = pd.read_parquet("data/train.parquet", engine="pyarrow")
        print(f"Full training data loaded: {all_train.shape}")

        # 2. 피처 전처리
        print("🔧 Processing features...")
        processor = TrainingFeatureProcessor()

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

        # 4. 메모리 최적화
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
        os.makedirs(xgboost_data_dir, exist_ok=True)

        # NumPy 형태로 저장
        np.save(f'{xgboost_data_dir}/X_train.npy', X_train.values)
        np.save(f'{xgboost_data_dir}/X_val.npy', X_val.values)
        np.save(f'{xgboost_data_dir}/y_train.npy', y_train)
        np.save(f'{xgboost_data_dir}/y_val.npy', y_val)

        # 피처명과 프로세서 저장
        with open(f'{xgboost_data_dir}/feature_names.pkl', 'wb') as f:
            pickle.dump(list(X_train.columns), f)

        with open(f'{xgboost_data_dir}/processor.pkl', 'wb') as f:
            pickle.dump(processor, f)

        # 메타데이터 저장
        metadata = {
            'train_shape': X_train.shape,
            'val_shape': X_val.shape,
            'num_categorical': len(processor.categorical_features),
            'num_numerical': len(processor.numerical_features),
            'train_ctr': y_train.mean(),
            'val_ctr': y_val.mean(),
            'config_used': XGBOOST_CONFIG.copy()
        }

        with open(f'{xgboost_data_dir}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        # 메모리 정리
        del X_processed, X_train, X_val, y_train, y_val, all_train
        gc.collect()

    prep_time = time.time() - start_time

    # 최종 정보 출력
    with open(f'{xgboost_data_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print("\n" + "="*50)
    print("✅ XGBoost Data Preparation Completed!")
    print("="*50)
    print(f"Preparation time: {prep_time:.2f} seconds")
    print(f"Train samples: {metadata['train_shape'][0]:,}")
    print(f"Val samples: {metadata['val_shape'][0]:,}")
    print(f"Features: {metadata['train_shape'][1]}")
    print(f"Categorical features: {metadata['num_categorical']}")
    print(f"Numerical features: {metadata['num_numerical']}")
    print(f"Train CTR: {metadata['train_ctr']:.4f}")
    print(f"Val CTR: {metadata['val_ctr']:.4f}")

    print(f"\nSaved to: {xgboost_data_dir}/")
    print(f"Files created:")
    print(f"  - X_train.npy, X_val.npy")
    print(f"  - y_train.npy, y_val.npy")
    print(f"  - processor.pkl, metadata.pkl")

    print("\n🚀 Ready for XGBoost training!")


if __name__ == "__main__":
    main()