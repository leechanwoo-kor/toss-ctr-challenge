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

OUTPUT_X = PROCESSED_DIR / 'X_processed2.parquet'
OUTPUT_Y = PROCESSED_DIR / 'y_processed2.npy'
OUTPUT_ENCODERS = MODEL_DIR / 'label_encoders2.pkl'

CATEGORICAL_FEATURES = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']

def preprocess(df: pd.DataFrame, is_train: bool = True, encoders: dict = None) -> tuple[pd.DataFrame, dict | None]:
    """데이터프레임에 대한 고도화된 전처리를 수행하는 중앙 함수"""
    print(f"\nStarting advanced preprocessing... (is_train={is_train})")
    print(f"  - Initial memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Optimize data types first to reduce memory usage
    print("  - Step 0: Optimizing data types...")
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # --- 1. '프로필 정보 없음' 피처 생성 (콜드 스타트 처리) ---
    print("  - Step 1: Creating 'is_profile_missing' feature for cold start users...")
    df['is_profile_missing'] = df['gender'].isnull().astype('uint8')

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
    print(f"    Found {len(continuous_features)} continuous features to process")

    for i, feature in enumerate(continuous_features):
        if i % 10 == 0:
            print(f"    Processing feature {i+1}/{len(continuous_features)}: {feature}")

        null_count = df[feature].isnull().sum()
        if null_count > 0:
            if (df[feature] == 0).sum() > 0:
                # Use loc for more memory-efficient assignment
                df.loc[df[feature].isnull(), feature] = 0
            else:
                median_val = df[feature].median()
                df.loc[df[feature].isnull(), feature] = median_val

    # --- 4. 신규 피처 엔지니어링 ---
    print("  - Step 4: Engineering new features...")
    new_cols = {}

    # 4-1. 교차 피처
    new_cols['inventory_id_x_age_group'] = pd.factorize(df['inventory_id'].astype(str) + '_' + df['age_group'].astype(str))[0]

    # 4-2. 통계 피처
    new_cols['inventory_id_mean_history_a_1'] = df.groupby('inventory_id')['history_a_1'].transform('mean')

    # 4-3. Sequence 기반 피처 생성
    if 'seq' in df.columns:
        print("    - Creating sequence-based features...")
        # 시퀀스 길이
        new_cols['seq_length'] = df['seq'].str.count(',') + 1
        
        # 시퀀스 내 고유 값 개수
        new_cols['seq_unique_count'] = df['seq'].apply(lambda x: len(set(x.split(','))) if pd.notna(x) else 0)
        
        # 시퀀스 다양성 (고유값/전체길이)
        new_cols['seq_diversity'] = new_cols['seq_unique_count'] / new_cols['seq_length']
        new_cols['seq_diversity'] = new_cols['seq_diversity'].fillna(0)
        
        # 첫 번째와 마지막 행동
        new_cols['seq_first_action'] = df['seq'].str.split(',').str[0].astype('category').cat.codes
        new_cols['seq_last_action'] = df['seq'].str.split(',').str[-1].astype('category').cat.codes
        
        # 시퀀스에서 가장 빈번한 행동
        def get_most_frequent(seq_str):
            if pd.isna(seq_str): return 0
            actions = seq_str.split(',')
            return max(set(actions), key=actions.count) if actions else '0'
        
        new_cols['seq_most_frequent'] = df['seq'].apply(get_most_frequent).astype('category').cat.codes

    # 4-4. 임베딩 기반 피처 (내적) - Memory optimized
    feat_a_cols = [f'feat_a_{i}' for i in range(1, 19)]
    feat_b_cols = [f'feat_b_{i}' for i in range(1, 7)]
    if all(c in df.columns for c in feat_a_cols) and all(c in df.columns for c in feat_b_cols):
        print("    - Creating embedding dot product feature (feat_a vs feat_b)")
        # Process in smaller chunks to avoid memory issues
        chunk_size = 50000
        embedding_dots = []
        for i in range(0, len(df), chunk_size):
            end_idx = min(i + chunk_size, len(df))
            feat_a_chunk = df[feat_a_cols].iloc[i:end_idx].values
            feat_b_chunk = df[feat_b_cols].iloc[i:end_idx].values
            feat_b_tiled = np.tile(feat_b_chunk, (1, 3))
            chunk_dot = np.sum(feat_a_chunk * feat_b_tiled, axis=1)
            embedding_dots.append(chunk_dot)
        new_cols['user_ad_embedding_dot'] = np.concatenate(embedding_dots)

    # 생성된 피처들을 한 번에 합치기
    if new_cols:
        for col_name, col_data in new_cols.items():
            df[col_name] = col_data
    
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
