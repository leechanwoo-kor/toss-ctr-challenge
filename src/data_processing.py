import os
import gc
import time
import pandas as pd
from pathlib import Path
import pickle

from sklearn.preprocessing import LabelEncoder

# --- 설정 ---
DATA_DIR = Path('../data')
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = Path('../models')

TRAIN_PATH = DATA_DIR / 'train.parquet'

OUTPUT_X = PROCESSED_DIR / 'X_processed.parquet'
OUTPUT_Y = PROCESSED_DIR / 'y_processed.npy'
OUTPUT_ENCODERS = MODEL_DIR / 'label_encoders.pkl'


def main():
    """데이터 전처리 파이프라인"""
    start_time = time.time()
    print("🚀 Starting data processing pipeline...")

    # 디렉토리 생성
    PROCESSED_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. 데이터 로딩
    print(f"\n📦 Loading data from {TRAIN_PATH}...")
    train_df = pd.read_parquet(TRAIN_PATH)
    if 'seq' in train_df.columns:
        train_df = train_df.drop('seq', axis=1)
    print(f"Train data shape: {train_df.shape}")

    # 2. 피처 엔지니어링 및 전처리
    print("\n🔧 Preprocessing and feature engineering...")
    
    CATEGORICAL_FEATURES = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    CONTINUOUS_FEATURES = [col for col in train_df.columns if col.startswith(('feat_', 'history_', 'l_feat_')) and col not in CATEGORICAL_FEATURES]

    y = train_df['clicked'].values
    X_df = train_df.drop('clicked', axis=1)

    label_encoders = {}
    for feature in CATEGORICAL_FEATURES:
        if feature in X_df.columns:
            X_df[feature] = X_df[feature].fillna('missing')
            value_counts = X_df[feature].value_counts()
            top_values = value_counts.head(50000).index
            X_df[feature] = X_df[feature].apply(lambda x: x if x in top_values else 'rare')
            le = LabelEncoder()
            X_df[feature] = le.fit_transform(X_df[feature])
            label_encoders[feature] = le
            print(f"Encoded {feature}: {len(le.classes_)} unique values")

    for feature in CONTINUOUS_FEATURES:
        if feature in X_df.columns:
            X_df[feature] = X_df[feature].fillna(0)

    print("✅ Preprocessing complete")
    
    # 3. 결과 저장
    print("\n💾 Saving processed data...")
    
    # 처리된 피처(X)와 타겟(y) 저장
    X_df.to_parquet(OUTPUT_X, index=False)
    np.save(OUTPUT_Y, y)
    print(f"Processed features saved to: {OUTPUT_X}")
    print(f"Processed target saved to: {OUTPUT_Y}")

    # 라벨 인코더 저장
    with open(OUTPUT_ENCODERS, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved to: {OUTPUT_ENCODERS}")

    print(f"\n✅ Data processing pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    import numpy as np # main 실행 시에만 numpy 임포트
    main()
