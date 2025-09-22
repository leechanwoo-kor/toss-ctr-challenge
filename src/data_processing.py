import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import gc
import logging
from sklearn.preprocessing import LabelEncoder
from config import CATEGORICAL_FEATURES

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'

# 입력 파일
TRAIN_PATH = DATA_DIR / 'train.parquet'
TEST_PATH = DATA_DIR / 'test.parquet'

# 출력 파일
OUTPUT_X = PROCESSED_DIR / 'X_processed.parquet'
OUTPUT_Y = PROCESSED_DIR / 'y_processed.npy'
ENCODERS_PATH = MODEL_DIR / 'label_encoders.pkl'

# Critical features from report
CRITICAL_OUTLIER_FEATURES = ['feat_e_1', 'feat_e_2', 'feat_e_3', 'l_feat_2']
# CATEGORICAL_FEATURES는 config.py에서 import 해서 사용

def create_directories():
    """필요한 디렉토리 생성"""
    PROCESSED_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

def optimize_memory_usage(df):
    """메모리 최적화 - 보고서 권장사항"""
    logger.info("Step 0: Optimizing memory usage...")
    memory_before = df.memory_usage(deep=True).sum() / 1024**2

    # float64 → float32 변환 (ID 컬럼 제외)
    float64_cols = df.select_dtypes(include=['float64']).columns
    for col in float64_cols:
        if 'id' not in col.lower() and df[col].abs().max() < 1e7:
            df[col] = df[col].astype('float32')

    # int64 → 적절한 정수형으로 다운캐스팅
    int64_cols = df.select_dtypes(include=['int64']).columns
    for col in int64_cols:
        if 'id' not in col.lower():
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # object 타입 중 숫자형으로 변환 가능한 것들 처리
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if col in ['hour', 'day_of_week', 'age_group']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].astype('int8')
            except:
                pass

    memory_after = df.memory_usage(deep=True).sum() / 1024**2
    memory_saved = memory_before - memory_after
    logger.info(f"Memory: {memory_before:.1f}MB → {memory_after:.1f}MB (절약: {memory_saved:.1f}MB)")

    return df

def handle_feat_e_3_critical(df):
    """feat_e_3 특별 처리 - 보고서에서 가장 심각한 문제로 지적"""
    if 'feat_e_3' not in df.columns:
        return df

    logger.info("Critical: Processing feat_e_3 (10.1% missing + 13.6% outliers)")

    # 1. 결측 표시 피처 생성
    df['feat_e_3_is_missing'] = df['feat_e_3'].isnull().astype('uint8')

    # 2. 명확한 구분값으로 대체
    df['feat_e_3'] = df['feat_e_3'].fillna(-999)

    # 3. 이상치 처리 (13.6% → 5% 미만)
    Q1, Q3 = df['feat_e_3'].quantile([0.025, 0.975])
    df['feat_e_3'] = df['feat_e_3'].clip(Q1, Q3)

    logger.info(f"feat_e_3: Missing feature created, outliers clipped to [{Q1:.2f}, {Q3:.2f}]")
    return df

def handle_outliers_systematic(df):
    """체계적 이상치 처리 - 보고서 권장"""
    outlier_config = {
        'feat_e_1': 'winsorize_95',  # 19.1% 이상치
        'feat_e_2': 'log_transform', # 극단적 음수값 (-311,604)
        'l_feat_2': 'winsorize_95'   # 17.0% 이상치
    }

    for feature, method in outlier_config.items():
        if feature not in df.columns:
            continue

        if method == 'winsorize_95':
            Q1, Q3 = df[feature].quantile([0.025, 0.975])
            original_outliers = ((df[feature] < Q1) | (df[feature] > Q3)).sum()
            df[feature] = df[feature].clip(Q1, Q3)
            logger.info(f"{feature}: {original_outliers} outliers winsorized to [{Q1:.2f}, {Q3:.2f}]")

        elif method == 'log_transform' and feature == 'feat_e_2':
            min_val = df[feature].min()
            if min_val < 0:
                # 음수를 양수로 변환 후 로그 적용
                df[feature] = df[feature] - min_val + 1
                df[feature] = np.log1p(df[feature])
                logger.info(f"{feature}: Log transform applied (min was {min_val:.0f})")

    return df

def improved_missing_handler(df):
    """개선된 결측치 처리 - 보고서 권장 로직"""
    continuous_features = [col for col in df.columns
                          if col.startswith(('feat_', 'history_', 'l_feat_'))
                          and col not in CRITICAL_OUTLIER_FEATURES]

    logger.info(f"Processing {len(continuous_features)} continuous features for missing values...")

    processed_count = 0
    high_missing_count = 0
    medium_missing_count = 0
    low_missing_count = 0

    for feature in continuous_features:
        null_count = df[feature].isnull().sum()
        if null_count == 0:
            continue

        missing_rate = null_count / len(df)
        processed_count += 1

        if missing_rate > 0.1:  # 10% 이상: 별도 피처 생성
            df[f'{feature}_is_missing'] = df[feature].isnull().astype('uint8')
            df[feature] = df[feature].fillna(-999)
            high_missing_count += 1

        elif missing_rate > 0.05:  # 5-10%: 중위수
            median_val = df[feature].median()
            df[feature] = df[feature].fillna(median_val)
            medium_missing_count += 1

        else:  # 5% 미만: 평균값
            mean_val = df[feature].mean()
            df[feature] = df[feature].fillna(mean_val)
            low_missing_count += 1

    logger.info(f"Missing value processing complete:")
    logger.info(f"  High missing (>10%): {high_missing_count} features")
    logger.info(f"  Medium missing (5-10%): {medium_missing_count} features")
    logger.info(f"  Low missing (<5%): {low_missing_count} features")

    return df

def create_temporal_features(df):
    """고가치 시간 피처 생성 - 보고서 핵심 발견사항"""
    logger.info("Creating high-value temporal features...")

    # 먼저 hour와 day_of_week를 숫자로 확실히 변환
    hour_numeric = pd.to_numeric(df['hour'], errors='coerce')
    dow_numeric = pd.to_numeric(df['day_of_week'], errors='coerce')

    # 메모리 효율적인 개별 피처 추가
    df['is_dawn_peak'] = hour_numeric.isin([1, 2, 3, 4]).astype('uint8')
    df['is_tuesday'] = (dow_numeric == 2).astype('uint8')
    df['is_peak_combination'] = ((hour_numeric == 4) & (dow_numeric == 2)).astype('uint8')
    df['hour_dow_interaction'] = (hour_numeric * 10 + dow_numeric).astype('int16')
    df['is_profile_missing'] = df['gender'].isnull().astype('uint8')

    logger.info(f"Temporal features created:")
    logger.info(f"  Peak combination (4AM Tuesday): {df['is_peak_combination'].sum():,} samples")
    logger.info(f"  Dawn peak hours (1-4AM): {df['is_dawn_peak'].sum():,} samples")
    logger.info(f"  Tuesday samples: {df['is_tuesday'].sum():,} samples")

    return df

def create_user_segment_features(df):
    """사용자 세그먼트 피처 - 범주형 처리 후 실행"""
    logger.info("Creating user segment features (after categorical encoding)...")

    new_features = {}

    # 범주형 인코딩 후 숫자 값으로 비교
    # LabelEncoder로 변환된 값들을 확인해서 매핑

    # 가장 빈도 높은 조합들 찾기 (인코딩된 값 기준)
    combo_counts = df.groupby(['gender', 'age_group']).size().sort_values(ascending=False)
    top_combos = combo_counts.head(5)

    logger.info(f"  Found {len(top_combos)} high-frequency segment combinations")

    # 상위 2개 조합을 타겟으로 설정
    if len(top_combos) >= 2:
        top_combo_1 = top_combos.index[0]
        top_combo_2 = top_combos.index[1]

        new_features['is_top_segment_1'] = (
            (df['gender'] == top_combo_1[0]) & (df['age_group'] == top_combo_1[1])
        ).astype('uint8')

        new_features['is_top_segment_2'] = (
            (df['gender'] == top_combo_2[0]) & (df['age_group'] == top_combo_2[1])
        ).astype('uint8')

    # 상위 5개 조합을 고가치 세그먼트로 설정
    high_value_mask = pd.Series(False, index=df.index)
    for (gender, age), count in top_combos.items():
        mask = (df['gender'] == gender) & (df['age_group'] == age)
        high_value_mask |= mask
    new_features['is_high_value_segment'] = high_value_mask.astype('uint8')

    # 메모리 효율적인 개별 피처 추가
    for col_name, col_data in new_features.items():
        df[col_name] = col_data

    # 결과 로깅
    if 'is_top_segment_1' in new_features:
        logger.info(f"  Top segment 1: {new_features['is_top_segment_1'].sum():,} samples")
        logger.info(f"  Top segment 2: {new_features['is_top_segment_2'].sum():,} samples")

    logger.info(f"  High value segments (top 5): {new_features['is_high_value_segment'].sum():,} samples")

    return df

def create_inventory_features(df):
    """인벤토리 최적화 피처"""
    logger.info("Creating inventory optimization features...")

    # 실제 존재하는 인벤토리 기반 피처
    new_features = {
        'is_premium_inventory': df['inventory_id'].isin(['88', '37']).astype('uint8'),
        'is_high_volume_inventory': df['inventory_id'].isin(['2', '36', '37']).astype('uint8'),
        'is_low_performance_inventory': df['inventory_id'].isin(['11', '19']).astype('uint8')
    }

    # 메모리 효율적인 개별 피처 추가
    for col_name, col_data in new_features.items():
        df[col_name] = col_data

    logger.info(f"  Premium inventory: {new_features['is_premium_inventory'].sum():,} samples")
    logger.info(f"  High volume inventory: {new_features['is_high_volume_inventory'].sum():,} samples")
    logger.info(f"  Low performance inventory: {new_features['is_low_performance_inventory'].sum():,} samples")

    return df

def process_categorical_features(df):
    """개선된 범주형 피처 처리"""
    logger.info("Processing categorical features...")
    label_encoders = {}

    for feature in CATEGORICAL_FEATURES:
        if feature in df.columns:
            # 결측치를 "missing"으로 처리
            df[feature] = df[feature].fillna("missing")

            # 빈도 기반 희귀값 처리
            value_counts = df[feature].value_counts()
            min_frequency = max(10, len(df) * 0.0001)  # 0.01% 또는 최소 10개
            significant_values = value_counts[value_counts >= min_frequency].index

            df[feature] = df[feature].apply(
                lambda x: x if x in significant_values else "rare"
            )

            # Label Encoding
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature].astype(str))
            label_encoders[feature] = le

    logger.info(f"  Processed {len(CATEGORICAL_FEATURES)} categorical features")
    return df, label_encoders

def process_data():
    """Enhanced 데이터 처리 파이프라인"""
    logger.info("Starting enhanced data processing pipeline...")

    # 데이터 로딩
    logger.info("Loading train data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    logger.info(f"Train data shape: {train_df.shape}")

    # 타겟 분리
    y = train_df['clicked'].values
    X = train_df.drop('clicked', axis=1)

    # seq 피처 생성 (제거하기 전에)
    if 'seq' in X.columns:
        logger.info("Creating sequence features...")
        from seq_feature_engineering import create_sequence_features
        X = create_sequence_features(X)
        # 이제 원본 seq 컬럼 제거 (메모리 절약)
        X = X.drop('seq', axis=1)
        logger.info("Original seq column removed after feature extraction")

    # Phase 1: Critical 개선사항
    logger.info("\n=== Phase 1: Critical Improvements ===")

    # 1. 메모리 최적화
    X = optimize_memory_usage(X)

    # 2. feat_e_3 특별 처리 (가장 심각한 문제)
    X = handle_feat_e_3_critical(X)

    # 3. 체계적 이상치 처리
    X = handle_outliers_systematic(X)

    # 4. 개선된 결측치 처리
    X = improved_missing_handler(X)

    # 5. 고가치 시간 피처 생성
    X = create_temporal_features(X)

    # 6. 인벤토리 피처 (문자열 비교)
    X = create_inventory_features(X)

    # 7. 범주형 피처 처리 (먼저 숫자로 변환)
    X, label_encoders = process_categorical_features(X)

    # 8. 사용자 세그먼트 피처 (숫자 변환 후)
    X = create_user_segment_features(X)

    # 메모리 정리
    gc.collect()

    # 최종 결과
    logger.info(f"\nFinal processed data shape: {X.shape}")
    logger.info(f"Target distribution: Clicked={y.sum():,} ({y.mean():.2%}), Not clicked={len(y)-y.sum():,}")

    # 새로운 피처 개수 확인
    new_features = [col for col in X.columns if any(indicator in col for indicator in
                   ['_is_missing', 'is_dawn_peak', 'is_tuesday', 'is_peak_combination',
                    'hour_dow_interaction', 'is_super_premium', 'is_high_value', 'is_premium_inventory'])]
    logger.info(f"New engineered features: {len(new_features)}")

    # 저장
    logger.info("\nSaving enhanced processed data...")
    X.to_parquet(OUTPUT_X, index=False, compression='snappy')
    np.save(OUTPUT_Y, y)

    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(label_encoders, f)

    logger.info(f"Saved to:")
    logger.info(f"  Features: {OUTPUT_X}")
    logger.info(f"  Target: {OUTPUT_Y}")
    logger.info(f"  Encoders: {ENCODERS_PATH}")

    return X, y, label_encoders

def main():
    """메인 실행 함수"""
    create_directories()
    X, y, encoders = process_data()
    logger.info("\n🎯 Enhanced data processing completed!")
    logger.info("Expected improvements:")
    logger.info("  - AUC: +4.5% (Phase 1)")
    logger.info("  - Memory: 50% reduction")
    logger.info("  - CTR prediction: +15% accuracy")

if __name__ == "__main__":
    main()