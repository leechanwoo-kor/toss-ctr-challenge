"""
Current Preprocessing Analysis for DeepFM
현재 전처리 방식 분석 + 데이터 품질 분석을 결합
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def analyze_current_preprocessing():
    """현재 DeepFM에서 사용하는 전처리 방식 분석"""
    print("🔍 Current DeepFM Preprocessing Analysis")
    print("="*60)

    print("\n📋 Current Preprocessing Steps:")
    print("1. 범주형 피처 처리:")
    print("   • 결측값 → 'unknown' 문자열로 대체")
    print("   • LabelEncoder로 숫자 변환")
    print("   • 피처: gender, age_group, inventory_id, day_of_week, hour")

    print("\n2. 수치형 피처 처리:")
    print("   • 결측값 → 0으로 대체")
    print("   • float32로 변환")
    print("   • 112개 피처 (seq 제외한 나머지 모든 피처)")

    print("\n3. 시퀀스 피처 처리:")
    print("   • 문자열 파싱: np.fromstring(seq_str, sep=',')")
    print("   • 통계 변환: [length, mean, std, min, max, sum]")
    print("   • 파생 피처: [range, mean_per_len]")
    print("   • 에러/빈값 → 모두 0으로 대체")

    print("\n4. 데이터 타입:")
    print("   • 범주형: LongTensor (임베딩용)")
    print("   • 수치형: FloatTensor (float32)")

def analyze_data_with_current_method():
    """현재 방식으로 전처리할 때의 데이터 품질 분석"""
    print("\n🚀 Data Quality Analysis with Current Preprocessing")
    print("="*70)

    # 데이터 로드
    print("📂 Loading train dataset...")
    train_df = pd.read_parquet("data/train.parquet", engine="pyarrow")
    print(f"   Shape: {train_df.shape}")

    # 1. 현재 방식으로 결측치 분석
    print("\n📊 Missing Values Analysis (Current Method)")
    print("-"*50)

    categorical_features = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    exclude_cols = {'clicked', 'seq', 'ID'}
    numerical_features = [col for col in train_df.columns if col not in exclude_cols and col not in categorical_features]

    print(f"📋 Feature Classification:")
    print(f"   Categorical: {len(categorical_features)} features")
    print(f"   Numerical: {len(numerical_features)} features")
    print(f"   Sequence: 1 feature (seq)")

    # 범주형 결측치 분석
    print(f"\n🔍 Categorical Features Missing Values:")
    for col in categorical_features:
        if col in train_df.columns:
            missing_count = train_df[col].isnull().sum()
            missing_percent = missing_count / len(train_df) * 100
            unique_after = train_df[col].fillna('unknown').astype(str).nunique()

            print(f"   {col}:")
            print(f"      Missing: {missing_count:,} ({missing_percent:.2f}%)")
            print(f"      Unique (after unknown): {unique_after}")

            # 'unknown'이 추가되면서 카디널리티 증가 확인
            if missing_count > 0:
                print(f"      ⚠️  'unknown' category will be added!")

    # 수치형 결측치 분석
    print(f"\n🔍 Numerical Features Missing Values:")
    numerical_missing = train_df[numerical_features].isnull().sum()
    numerical_missing = numerical_missing[numerical_missing > 0].sort_values(ascending=False)

    if len(numerical_missing) > 0:
        print(f"   {len(numerical_missing)} numerical features have missing values:")
        for col, count in numerical_missing.head(10).items():
            percent = count / len(train_df) * 100
            print(f"      {col}: {count:,} ({percent:.2f}%) → will be filled with 0")

            # 0으로 채우는 것이 적절한지 확인
            col_stats = train_df[col].describe()
            if col_stats['min'] >= 0:
                print(f"         ✅ Min value: {col_stats['min']:.2f} (0 filling seems reasonable)")
            else:
                print(f"         ⚠️  Min value: {col_stats['min']:.2f} (0 might not be optimal)")
    else:
        print("   ✅ No missing values in numerical features!")

    # 시퀀스 피처 분석
    analyze_sequence_processing(train_df)

    # 이상치 분석 (현재 방식으로는 처리 안함)
    analyze_outliers_current_method(train_df, numerical_features)

def analyze_sequence_processing(df):
    """시퀀스 처리 방식 분석"""
    print(f"\n🔍 Sequence Processing Analysis")
    print("-"*50)

    if 'seq' not in df.columns:
        print("❌ No sequence column found")
        return

    # 샘플링 (메모리 절약)
    sample_size = min(50000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    print(f"📊 Analyzing {sample_size:,} samples...")

    # 현재 방식으로 처리
    parse_success = 0
    parse_errors = 0
    empty_sequences = 0
    valid_sequences = []

    for seq_str in tqdm(sample_df['seq'].astype(str), desc="Analyzing sequence processing"):
        if seq_str and seq_str != 'nan' and seq_str != '':
            try:
                seq_array = np.fromstring(seq_str, sep=",", dtype=np.float32)
                if len(seq_array) > 0:
                    parse_success += 1
                    valid_sequences.append(len(seq_array))
                else:
                    empty_sequences += 1
            except:
                parse_errors += 1
        else:
            empty_sequences += 1

    total = len(sample_df)
    print(f"\n📊 Sequence Processing Results:")
    print(f"   Parse Success: {parse_success:,} ({parse_success/total*100:.2f}%)")
    print(f"   Parse Errors: {parse_errors:,} ({parse_errors/total*100:.2f}%)")
    print(f"   Empty/Null: {empty_sequences:,} ({empty_sequences/total*100:.2f}%)")

    if valid_sequences:
        valid_sequences = np.array(valid_sequences)
        print(f"\n📊 Valid Sequence Statistics:")
        print(f"   Length Mean: {np.mean(valid_sequences):.2f}")
        print(f"   Length Std: {np.std(valid_sequences):.2f}")
        print(f"   Length Range: [{np.min(valid_sequences)}, {np.max(valid_sequences)}]")

    # 에러/빈값이 많으면 문제
    error_rate = (parse_errors + empty_sequences) / total * 100
    if error_rate > 10:
        print(f"   ⚠️  High error rate: {error_rate:.2f}%")
        print(f"   💡 Consider improving sequence parsing logic")

def analyze_outliers_current_method(df, numerical_features):
    """현재 방식(이상치 처리 안함)에서의 이상치 영향 분석"""
    print(f"\n🔍 Outlier Analysis (Current: No Treatment)")
    print("-"*50)

    print("📊 Current outlier handling: NO TREATMENT")
    print("   → All extreme values are kept as-is")

    # 샘플 피처들에 대해 이상치 영향 분석
    sample_features = numerical_features[:10]  # 처음 10개만

    outlier_impact = []

    for col in sample_features:
        if col in df.columns and df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
            # IQR 방법으로 이상치 탐지
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_percent = len(outliers) / len(df) * 100

            # 극값 영향
            value_range = df[col].max() - df[col].min()
            std_dev = df[col].std()

            outlier_impact.append({
                'feature': col,
                'outlier_percent': outlier_percent,
                'value_range': value_range,
                'std_dev': std_dev,
                'max_value': df[col].max(),
                'min_value': df[col].min()
            })

    # 이상치가 많은 피처들
    high_outlier_features = [item for item in outlier_impact if item['outlier_percent'] > 5]

    if high_outlier_features:
        print(f"\n⚠️  Features with high outlier rates (>5%):")
        for item in high_outlier_features[:5]:
            print(f"   {item['feature']}:")
            print(f"      Outliers: {item['outlier_percent']:.2f}%")
            print(f"      Range: [{item['min_value']:.2f}, {item['max_value']:.2f}]")
            print(f"      Std Dev: {item['std_dev']:.2f}")

    # 극값 범위 피처들
    extreme_range_features = [item for item in outlier_impact if item['max_value'] / (abs(item['min_value']) + 1e-8) > 100]

    if extreme_range_features:
        print(f"\n⚠️  Features with extreme value ranges:")
        for item in extreme_range_features[:5]:
            ratio = item['max_value'] / (abs(item['min_value']) + 1e-8)
            print(f"   {item['feature']}: range ratio {ratio:.2f}")

def suggest_improvements():
    """현재 방식 기반 개선 제안"""
    print(f"\n💡 Preprocessing Improvement Suggestions")
    print("="*60)

    print(f"\n🔧 1. Missing Value Handling:")
    print(f"   Current: categorical→'unknown', numerical→0")
    print(f"   Suggestions:")
    print(f"   • Use median/mode instead of 0 for numerical features")
    print(f"   • Consider missing indicator features")
    print(f"   • Analyze missing patterns before imputation")

    print(f"\n🔧 2. Outlier Treatment:")
    print(f"   Current: No treatment")
    print(f"   Suggestions:")
    print(f"   • Apply log transformation for skewed features")
    print(f"   • Clip extreme values (99th percentile)")
    print(f"   • Use robust scaling instead of simple fillna(0)")

    print(f"\n🔧 3. Sequence Processing:")
    print(f"   Current: Basic statistics (8 features)")
    print(f"   Suggestions:")
    print(f"   • Add percentiles (Q25, Q50, Q75)")
    print(f"   • Include trend analysis")
    print(f"   • Handle parsing errors more gracefully")

    print(f"\n🔧 4. Categorical Encoding:")
    print(f"   Current: LabelEncoder")
    print(f"   Suggestions:")
    print(f"   • Consider frequency-based encoding")
    print(f"   • Use target encoding for high-cardinality features")
    print(f"   • Handle unknown categories in test data better")

    print(f"\n🔧 5. Feature Scaling:")
    print(f"   Current: None (just float32 conversion)")
    print(f"   Suggestions:")
    print(f"   • StandardScaler or RobustScaler for numerical features")
    print(f"   • MinMaxScaler for bounded features")

def run_comprehensive_analysis():
    """전체 분석 실행"""
    print("🚀 Comprehensive Current Preprocessing + Data Quality Analysis")
    print("="*80)

    # 1. 현재 전처리 방식 설명
    analyze_current_preprocessing()

    # 2. 현재 방식으로 데이터 분석
    analyze_data_with_current_method()

    # 3. 개선 제안
    suggest_improvements()

    print(f"\n✅ Analysis Complete!")
    print(f"💡 Use these insights to improve DeepFM preprocessing pipeline")

if __name__ == "__main__":
    run_comprehensive_analysis()