import os
import gc
import time
import pandas as pd
from pathlib import Path
import pickle

from sklearn.preprocessing import LabelEncoder

# --- 설정 ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'

TRAIN_PATH = DATA_DIR / 'train.parquet'
TEST_PATH = DATA_DIR / 'test.parquet'

OUTPUT_X = PROCESSED_DIR / 'X_processed.parquet'
OUTPUT_Y = PROCESSED_DIR / 'y_processed.npy'
OUTPUT_ENCODERS = MODEL_DIR / 'label_encoders.pkl'

CATEGORICAL_FEATURES = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']

def preprocess(df: pd.DataFrame, is_train: bool = True, encoders: dict = None) -> tuple[pd.DataFrame, dict | None]:
    """데이터프레임에 대한 전처리를 수행하는 중앙 함수"""
    print(f"\n🔧 Preprocessing data... (is_train={is_train})")
    
    # 연속형 피처 정의
    continuous_features = [col for col in df.columns if col.startswith(('feat_', 'history_', 'l_feat_')) and col not in CATEGORICAL_FEATURES]

    # 범주형 피처 처리
    if is_train:
        # 학습 시: LabelEncoder를 새로 만들고 fit_transform 수행
        label_encoders = {}
        for feature in CATEGORICAL_FEATURES:
            if feature in df.columns:
                df[feature] = df[feature].fillna('missing')
                value_counts = df[feature].value_counts()
                top_values = value_counts.head(50000).index
                df[feature] = df[feature].apply(lambda x: x if x in top_values else 'rare')
                
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature])
                label_encoders[feature] = le
                print(f"  Encoded {feature}: {len(le.classes_)} unique values")
        encoders_to_return = label_encoders
    else:
        # 추론 시: 기존에 학습된 encoder를 사용
        if encoders is None:
            raise ValueError("Encoders must be provided for test data processing.")
        for feature in CATEGORICAL_FEATURES:
            if feature in df.columns and feature in encoders:
                encoder = encoders[feature]
                known_classes = set(encoder.classes_)
                
                df[feature] = df[feature].fillna('missing')
                df[feature] = df[feature].apply(lambda x: x if x in known_classes else 'rare')
                df[feature] = encoder.transform(df[feature])
        encoders_to_return = None

    # 연속형 피처 결측치 처리
    for feature in continuous_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)

    print("✅ Preprocessing complete")
    return df, encoders_to_return

def main():
    """학습 데이터 전처리 파이프라인 (기존 역할)"""
    start_time = time.time()
    print("🚀 Starting data processing pipeline for TRAINING data...")

    # --- 0. 경로 및 디렉토리 설정 ---
    PROCESSED_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # --- 1. 원본 데이터 검증 및 로딩 ---
    print(f"\n[Step 1/3] 📦 Validating and loading data...")
    if not TRAIN_PATH.exists():
        print(f"❌ Error: Source data file not found at {TRAIN_PATH}")
        return

    train_df = pd.read_parquet(TRAIN_PATH)
    print(f"Original train data shape: {train_df.shape}")
    
    if 'seq' in train_df.columns:
        train_df = train_df.drop('seq', axis=1)
        print("Dropped 'seq' column.")

    # --- 2. 전처리 실행 ---
    y = train_df['clicked'].values
    X_df = train_df.drop('clicked', axis=1)
    
    X_processed, label_encoders = preprocess(X_df, is_train=True)
    
    # --- 3. 결과 저장 ---
    print("\n[Step 3/3] 💾 Saving processed data and encoders...")
    
    X_processed.to_parquet(OUTPUT_X, index=False)
    np.save(OUTPUT_Y, y)
    print(f"Processed features saved to: {OUTPUT_X}")
    print(f"Processed target saved to: {OUTPUT_Y}")

    with open(OUTPUT_ENCODERS, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved to: {OUTPUT_ENCODERS}")

    print(f"\n✅ Data processing pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()