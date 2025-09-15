import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
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
    """데이터프레임에 대한 고도화된 전처리를 수행하는 중앙 함수"""
    print(f"\nStarting advanced preprocessing... (is_train={is_train})")
    
    # --- 1. '프로필 정보 없음' 피처 생성 (콜드 스타트 처리) ---
    print("  - Step 1: Creating 'is_profile_missing' feature for cold start users...")
    df['is_profile_missing'] = df['gender'].isnull().astype(int)

    # --- 2. 기본 범주형 피처 처리 ---
    print("  - Step 2: Processing basic categorical features...")
    if is_train:
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
        encoders_to_return = label_encoders
    else:
        if encoders is None: raise ValueError("Encoders must be provided for test data.")
        for feature in CATEGORICAL_FEATURES:
            if feature in df.columns and feature in encoders:
                encoder = encoders[feature]
                known_classes = set(encoder.classes_)
                df[feature] = df[feature].fillna('missing')
                df[feature] = df[feature].apply(lambda x: x if x in known_classes else 'rare')
                df[feature] = encoder.transform(df[feature])
        encoders_to_return = None

    # --- 3. 동적 연속형 피처 결측치 처리 ---
    print("  - Step 3: Applying dynamic imputation for continuous features...")
    continuous_features = [col for col in df.columns if col.startswith(('feat_', 'history_', 'l_feat_'))]
    for feature in continuous_features:
        if df[feature].isnull().sum() > 0:
            if (df[feature] == 0).sum() > 0:
                df[feature] = df[feature].fillna(0)
            else:
                median_val = df[feature].median()
                df[feature] = df[feature].fillna(median_val)

    # --- 4. 신규 피처 엔지니어링 ---
    print("  - Step 4: Engineering new features...")
    new_cols = {}

    # 4-1. 교차 피처
    new_cols['inventory_id_x_age_group'] = pd.factorize(df['inventory_id'].astype(str) + '_' + df['age_group'].astype(str))[0]

    # 4-2. 통계 피처
    new_cols['inventory_id_mean_history_a_1'] = df.groupby('inventory_id')['history_a_1'].transform('mean')

    # 4-3. 임베딩 기반 피처 (내적)
    feat_a_cols = [f'feat_a_{i}' for i in range(1, 19)]
    feat_b_cols = [f'feat_b_{i}' for i in range(1, 7)]
    if all(c in df.columns for c in feat_a_cols) and all(c in df.columns for c in feat_b_cols):
        print("    - Creating embedding dot product feature (feat_a vs feat_b)")
        feat_b_tiled = np.tile(df[feat_b_cols].values, (1, 3))
        new_cols['user_ad_embedding_dot'] = np.sum(df[feat_a_cols].values * feat_b_tiled, axis=1)
    
    # 생성된 피처들을 한 번에 합치기
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    
    print("\nAdvanced preprocessing complete.")
    return df, encoders_to_return

def main():
    """학습 데이터 전처리 파이프라인"""
    start_time = time.time()
    print("Starting data processing pipeline for TRAINING data...")
    PROCESSED_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print(f"\n[Step 1/3] Loading data...")
    if not TRAIN_PATH.exists():
        print(f"Error: Source data file not found at {TRAIN_PATH}")
        return
    train_df = pd.read_parquet(TRAIN_PATH)
    if 'seq' in train_df.columns: train_df = train_df.drop('seq', axis=1)

    print("\n[Step 2/3] Preprocessing and Engineering Features...")
    y = train_df['clicked'].values
    X_df = train_df.drop('clicked', axis=1)
    X_processed, label_encoders = preprocess(X_df, is_train=True)
    
    print(f"\n[Step 3/3] Saving processed data and encoders...")
    X_processed.to_parquet(OUTPUT_X, index=False)
    np.save(OUTPUT_Y, y)
    print(f"  - Processed features saved to: {OUTPUT_X} (shape: {X_processed.shape})")
    print(f"  - Processed target saved to: {OUTPUT_Y}")
    with open(OUTPUT_ENCODERS, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"  - Label encoders saved to: {OUTPUT_ENCODERS}")

    print(f"\nData processing pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
