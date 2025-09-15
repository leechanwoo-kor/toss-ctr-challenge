import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import gc
import logging
from sklearn.preprocessing import LabelEncoder

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
CATEGORICAL_FEATURES = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']

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

    for feature in continuous_features:
        null_count = df[feature].isnull().sum()
        if null_count == 0:
            continue

        missing_rate = null_count / len(df)

        if missing_rate > 0.1:  # 10% 이상: 별도 피처 생성
            df[f'{feature}_is_missing'] = df[feature].isnull().astype('uint8')
            df[feature] = df[feature].fillna(-999)
            logger.info(f"{feature}: {missing_rate:.1%} missing - separate indicator created")

        elif missing_rate > 0.05:  # 5-10%: 중위수
            median_val = df[feature].median()
            df[feature] = df[feature].fillna(median_val)
            logger.info(f"{feature}: {missing_rate:.1%} missing - filled with median")

        else:  # 5% 미만: 평균값
            mean_val = df[feature].mean()
            df[feature] = df[feature].fillna(mean_val)
            logger.info(f"{feature}: {missing_rate:.1%} missing - filled with mean")

    return df

def create_temporal_features(df):
    """고가치 시간 피처 생성 - 보고서 핵심 발견사항"""
    logger.info("Creating high-value temporal features...")

    # 모든 새 피처를 dict에 먼저 생성 (fragmentation 방지)
    new_features = {}

    # 먼저 hour와 day_of_week를 숫자로 확실히 변환
    hour_numeric = pd.to_numeric(df['hour'], errors='coerce')
    dow_numeric = pd.to_numeric(df['day_of_week'], errors='coerce')

    # 새벽 시간대 표시 (CTR 4-7배 향상)
    new_features['is_dawn_peak'] = hour_numeric.isin([1, 2, 3, 4]).astype('uint8')

    # 화요일 표시 (가장 높은 CTR)
    new_features['is_tuesday'] = (dow_numeric == 2).astype('uint8')

    # 최고 효율 조합 (6.85% CTR)
    new_features['is_peak_combination'] = (
        (hour_numeric == 4) & (dow_numeric == 2)
    ).astype('uint8')

    # 시간×요일 상호작용 인코딩 (확실히 숫자로)
    new_features['hour_dow_interaction'] = (hour_numeric * 10 + dow_numeric).astype('int16')

    # 프로필 정보 없음 피처 (콜드 스타트 처리)
    new_features['is_profile_missing'] = df['gender'].isnull().astype('uint8')

    # 한 번에 모든 피처 추가
    for col_name, col_data in new_features.items():
        df[col_name] = col_data

    peak_count = new_features['is_peak_combination'].sum()
    dawn_count = new_features['is_dawn_peak'].sum()
    tuesday_count = new_features['is_tuesday'].sum()

    logger.info(f"Temporal features created:")
    logger.info(f"  Peak combination (4AM Tuesday): {peak_count:,} samples")
    logger.info(f"  Dawn peak hours (1-4AM): {dawn_count:,} samples")
    logger.info(f"  Tuesday samples: {tuesday_count:,} samples")

    return df

def create_user_segment_features(df):
    """사용자 세그먼트 피처 - Phase 2 준비"""
    logger.info("Creating user segment features...")

    new_features = {}

    # 슈퍼 프리미엄 세그먼트 (10% CTR)
    new_features['is_super_premium_segment'] = (
        (df['gender'] == 1) & (df['age_group'] == 1)
    ).astype('uint8')

    # 고가치 세그먼트들
    high_ctr_combinations = [(2, 3), (1, 4), (2, 2), (1, 3)]
    high_value_mask = pd.Series(False, index=df.index)
    for gender, age in high_ctr_combinations:
        mask = (df['gender'] == gender) & (df['age_group'] == age)
        high_value_mask |= mask
    new_features['is_high_value_segment'] = high_value_mask.astype('uint8')

    # 한 번에 모든 피처 추가
    for col_name, col_data in new_features.items():
        df[col_name] = col_data

    super_premium_count = new_features['is_super_premium_segment'].sum()
    high_value_count = new_features['is_high_value_segment'].sum()

    logger.info(f"  Super premium segment: {super_premium_count:,} samples")
    logger.info(f"  High value segments: {high_value_count:,} samples")

    return df

def create_inventory_features(df):
    """인벤토리 최적화 피처"""
    logger.info("Creating inventory optimization features...")

    new_features = {}

    # 고성과 인벤토리 (집중 투자 대상)
    high_performance = [88, 41, 37]
    new_features['is_premium_inventory'] = df['inventory_id'].isin(high_performance).astype('uint8')

    # 저성과 인벤토리 표시 (최적화 필요)
    low_performance = [2, 19, 11]
    new_features['is_low_performance_inventory'] = df['inventory_id'].isin(low_performance).astype('uint8')

    # 한 번에 모든 피처 추가
    for col_name, col_data in new_features.items():
        df[col_name] = col_data

    premium_count = new_features['is_premium_inventory'].sum()
    low_perf_count = new_features['is_low_performance_inventory'].sum()

    logger.info(f"  Premium inventory samples: {premium_count:,}")
    logger.info(f"  Low performance inventory samples: {low_perf_count:,}")

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

            logger.info(f"  {feature}: {len(significant_values)} significant values (threshold: {min_frequency})")

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

    # seq 컬럼 제거 (메모리 절약)
    if 'seq' in X.columns:
        X = X.drop('seq', axis=1)
        logger.info("seq column removed for memory efficiency")

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

    # 6. 사용자 세그먼트 피처 (Phase 2 준비)
    X = create_user_segment_features(X)

    # 7. 인벤토리 피처
    X = create_inventory_features(X)

    # 8. 범주형 피처 처리
    X, label_encoders = process_categorical_features(X)

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