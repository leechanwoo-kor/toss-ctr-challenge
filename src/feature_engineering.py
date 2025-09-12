import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import gc

# --- 설정 ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

INPUT_X = PROCESSED_DIR / 'X_processed.parquet'
INPUT_Y = PROCESSED_DIR / 'y_processed.npy'
OUTPUT_X = PROCESSED_DIR / 'X_engineered.parquet'
OUTPUT_Y = PROCESSED_DIR / 'y_engineered.npy'

def create_interaction_features(df, categorical_cols, continuous_cols, max_interactions=20):
    """피처 교차항 생성"""
    print(f"Creating interaction features...")
    
    # 1. 범주형 × 범주형 교차항 (상위 중요 조합만)
    important_cat_pairs = [
        ('gender', 'age_group'),
        ('inventory_id', 'hour'), 
        ('day_of_week', 'hour'),
        ('gender', 'inventory_id'),
        ('age_group', 'inventory_id')
    ]
    
    for col1, col2 in important_cat_pairs:
        if col1 in df.columns and col2 in df.columns:
            interaction_name = f'{col1}_X_{col2}'
            df[interaction_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
            # 라벨 인코딩 (빈도수 기준)
            value_counts = df[interaction_name].value_counts()
            # 상위 1000개만 유지, 나머지는 'rare'
            top_values = value_counts.head(1000).index
            df[interaction_name] = df[interaction_name].apply(lambda x: x if x in top_values else 'rare')
            # 숫자로 인코딩
            unique_vals = df[interaction_name].unique()
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            df[interaction_name] = df[interaction_name].map(val_to_idx)
            print(f"  Created {interaction_name}: {len(unique_vals)} unique values")
    
    # 2. 범주형 × 연속형 교차항 (통계 기반)
    for cat_col in categorical_cols[:3]:  # 상위 3개 범주형만
        if cat_col not in df.columns:
            continue
            
        for cont_col in continuous_cols[:5]:  # 상위 5개 연속형만
            if cont_col not in df.columns:
                continue
                
            # 범주별 연속형 변수의 통계량
            grouped = df.groupby(cat_col)[cont_col]
            
            # 평균
            mean_name = f'{cat_col}_mean_{cont_col}'
            df[mean_name] = df[cat_col].map(grouped.mean())
            
            # 표준편차
            std_name = f'{cat_col}_std_{cont_col}'  
            df[std_name] = df[cat_col].map(grouped.std()).fillna(0)
            
            print(f"  Created {mean_name} and {std_name}")
    
    # 3. 연속형 피처 조합 (곱셈, 나눗셈)
    important_cont_pairs = []
    for i, cont_cols_subset in enumerate([continuous_cols[:5]]):  # 상위 5개만
        for col1, col2 in combinations(cont_cols_subset, 2):
            if col1 in df.columns and col2 in df.columns and len(important_cont_pairs) < 10:
                # 곱셈
                mult_name = f'{col1}_mult_{col2}'
                df[mult_name] = df[col1] * df[col2]
                
                # 안전한 나눗셈
                div_name = f'{col1}_div_{col2}'
                df[div_name] = df[col1] / (df[col2] + 1e-8)
                
                important_cont_pairs.extend([mult_name, div_name])
                print(f"  Created {mult_name} and {div_name}")
    
    return df

def create_statistical_features(df, categorical_cols, target=None):
    """통계 기반 피처 생성"""
    print(f"Creating statistical features...")
    
    if target is None:
        return df
    
    # 각 범주별 클릭율 (타겟 인코딩)
    for col in categorical_cols[:5]:  # 상위 5개만
        if col not in df.columns:
            continue
            
        # 전체 평균 클릭율
        global_mean = target.mean()
        
        # 범주별 클릭율 계산 (스무딩 적용)
        category_stats = pd.DataFrame({col: df[col], 'target': target}).groupby(col)['target'].agg(['count', 'mean'])
        
        # 베이지안 스무딩 (최소 20개 샘플 필요)
        alpha = 20
        category_stats['smoothed_mean'] = (category_stats['mean'] * category_stats['count'] + global_mean * alpha) / (category_stats['count'] + alpha)
        
        # 피처로 추가
        feature_name = f'{col}_click_rate'
        df[feature_name] = df[col].map(category_stats['smoothed_mean']).fillna(global_mean)
        print(f"  Created {feature_name}")
    
    return df

def create_time_features(df):
    """시간 기반 파생 피처"""
    print(f"Creating time-based features...")
    
    if 'hour' in df.columns:
        # 시간대 구분
        df['hour_period'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
        df['hour_period'] = df['hour_period'].cat.codes
        
        # 피크 시간대
        df['is_peak_hour'] = ((df['hour'] >= 11) & (df['hour'] <= 13)) | ((df['hour'] >= 18) & (df['hour'] <= 21))
        df['is_peak_hour'] = df['is_peak_hour'].astype(int)
        
        print("  Created hour_period and is_peak_hour")
    
    if 'day_of_week' in df.columns:
        # 주말 여부
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        print("  Created is_weekend")
    
    return df

def main():
    """피처 엔지니어링 메인 파이프라인"""
    print("Starting feature engineering pipeline...")
    
    # 데이터 로드
    print("Loading processed data...")
    X_df = pd.read_parquet(INPUT_X)
    y = np.load(INPUT_Y)
    
    print(f"Original shape: {X_df.shape}")
    
    # 피처 타입 구분
    CATEGORICAL_FEATURES = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    CONTINUOUS_FEATURES = [col for col in X_df.columns if col.startswith(('feat_', 'history_', 'l_feat_')) and col not in CATEGORICAL_FEATURES]
    
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Continuous features: {len(CONTINUOUS_FEATURES)}")
    
    # 1. 교차항 피처 생성
    X_df = create_interaction_features(X_df, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES)
    gc.collect()
    
    # 2. 통계 피처 생성 (타겟 정보 사용)
    X_df = create_statistical_features(X_df, CATEGORICAL_FEATURES, y)
    gc.collect()
    
    # 3. 시간 기반 피처 생성
    X_df = create_time_features(X_df)
    gc.collect()
    
    print(f"\nFinal engineered shape: {X_df.shape}")
    print(f"Added features: {X_df.shape[1] - pd.read_parquet(INPUT_X).shape[1]}")
    
    # 저장
    print("Saving engineered features...")
    X_df.to_parquet(OUTPUT_X, index=False)
    np.save(OUTPUT_Y, y)  # 타겟은 그대로
    
    print(f"Saved to: {OUTPUT_X}")
    print("Feature engineering completed!")

if __name__ == '__main__':
    main()