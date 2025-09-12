import time
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import gc
from pathlib import Path
from itertools import combinations

# --- 설정 ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'

TEST_PATH = DATA_DIR / 'test.parquet'
ENCODERS_PATH = MODEL_DIR / 'label_encoders.pkl'

# --- 피처 엔지니어링 함수들 (훈련과 동일) ---
def create_interaction_features(df, categorical_cols, continuous_cols):
    """피처 교차항 생성 (훈련과 동일한 로직)"""
    print("  Creating interaction features...")
    
    # 1. 범주형 × 범주형 교차항
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
    
    # 2. 범주형 × 연속형 교차항 (통계 기반)
    for cat_col in categorical_cols[:3]:  # 상위 3개 범주형만
        if cat_col not in df.columns:
            continue
            
        for cont_col in continuous_cols[:5]:  # 상위 5개 연속형만
            if cont_col not in df.columns:
                continue
                
            # 범주별 연속형 변수의 통계량 (전체 평균으로 대체)
            grouped = df.groupby(cat_col)[cont_col]
            
            # 평균
            mean_name = f'{cat_col}_mean_{cont_col}'
            df[mean_name] = df[cat_col].map(grouped.mean()).fillna(df[cont_col].mean())
            
            # 표준편차
            std_name = f'{cat_col}_std_{cont_col}'  
            df[std_name] = df[cat_col].map(grouped.std()).fillna(df[cont_col].std())
    
    # 3. 연속형 피처 조합
    important_cont_pairs = []
    for cont_cols_subset in [continuous_cols[:5]]:  # 상위 5개만
        for col1, col2 in combinations(cont_cols_subset, 2):
            if col1 in df.columns and col2 in df.columns and len(important_cont_pairs) < 10:
                # 곱셈
                mult_name = f'{col1}_mult_{col2}'
                df[mult_name] = df[col1] * df[col2]
                
                # 안전한 나눗셈
                div_name = f'{col1}_div_{col2}'
                df[div_name] = df[col1] / (df[col2] + 1e-8)
                
                important_cont_pairs.extend([mult_name, div_name])
    
    return df

def create_statistical_features(df, categorical_cols, global_click_rate=0.019075):
    """통계 기반 피처 생성 (테스트 데이터용)"""
    print("  Creating statistical features...")
    
    # 각 범주별 클릭율 (전체 평균으로 대체)
    for col in categorical_cols[:5]:  # 상위 5개만
        if col not in df.columns:
            continue
            
        # 피처명
        feature_name = f'{col}_click_rate'
        # 전체 평균 클릭율로 초기화 (실제로는 훈련 데이터의 통계를 사용해야 함)
        df[feature_name] = global_click_rate
    
    return df

def create_time_features(df):
    """시간 기반 파생 피처"""
    print("  Creating time features...")
    
    if 'hour' in df.columns:
        # 시간대 구분
        df['hour_period'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
        df['hour_period'] = df['hour_period'].cat.codes
        
        # 피크 시간대
        df['is_peak_hour'] = ((df['hour'] >= 11) & (df['hour'] <= 13)) | ((df['hour'] >= 18) & (df['hour'] <= 21))
        df['is_peak_hour'] = df['is_peak_hour'].astype(int)
    
    if 'day_of_week' in df.columns:
        # 주말 여부
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df

def preprocess_test_data(X_test, label_encoders):
    """테스트 데이터에 학습 시와 동일한 전처리를 적용"""
    print("\nPreprocessing test data with feature engineering...")
    
    CATEGORICAL_FEATURES = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    CONTINUOUS_FEATURES = [col for col in X_test.columns if col.startswith(('feat_', 'history_', 'l_feat_')) and col not in CATEGORICAL_FEATURES]

    # 1. 기본 범주형 피처 인코딩
    for feature in CATEGORICAL_FEATURES:
        if feature in X_test.columns and feature in label_encoders:
            print(f"    Encoding {feature}...")
            encoder = label_encoders[feature]
            known_classes = set(encoder.classes_)

            # 결측치 처리
            X_test[feature] = X_test[feature].fillna('missing')

            # 학습 시 없었던 새로운 값 처리 (rare 또는 최빈값으로 대체)
            X_test[feature] = X_test[feature].apply(lambda x: x if x in known_classes else 'rare')
            
            # 변환
            X_test[feature] = encoder.transform(X_test[feature])

    # 2. 연속형 피처 결측치 처리
    for feature in CONTINUOUS_FEATURES:
        if feature in X_test.columns:
            X_test[feature] = X_test[feature].fillna(0)
    
    # 3. 피처 엔지니어링 적용
    print("  Applying feature engineering...")
    
    # 교차항 피처 생성
    X_test = create_interaction_features(X_test, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES)
    gc.collect()
    
    # 통계 기반 피처 생성 (타겟 인코딩)
    X_test = create_statistical_features(X_test, CATEGORICAL_FEATURES)
    gc.collect()
    
    # 시간 기반 피처 생성
    X_test = create_time_features(X_test)
    gc.collect()
    
    print("Preprocessing with feature engineering complete")
    print(f"Final feature count: {X_test.shape[1]}")
    return X_test

# --- 메인 추론 로직 ---
def main():
    """메인 추론 파이프라인"""
    start_time = time.time()
    print("Starting inference pipeline with engineered features...")
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    # 1. 최신 튜닝된 모델 파일 찾기
    model_files = list(MODEL_DIR.glob('xgboost-tuned_*_best_model.json'))
    if not model_files:
        # fallback to engineered model
        model_files = list(MODEL_DIR.glob('xgboost-engineered_*_best_model.json'))
        if not model_files:
            model_files = list(MODEL_DIR.glob('*_best_model.json'))
    
    if not model_files:
        raise FileNotFoundError("No model file found in models directory")
    
    # 가장 최신 모델 선택 (파일명의 타임스탬프 기준)
    model_path = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Using model: {model_path.name}")
    
    # 해당 모델의 결과 파일 찾기
    results_path = model_path.with_name(model_path.name.replace('_best_model.json', '_results.json'))
    
    # 2. 모델 및 인코더 로드
    print("\nLoading model and encoders...")
    model = xgb.Booster()
    model.load_model(str(model_path))
    print("  Model loaded.")

    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    print("  Label encoders loaded.")

    # 3. 테스트 데이터 로드
    print(f"\nLoading test data from {TEST_PATH}...")
    test_df = pd.read_parquet(TEST_PATH)
    
    test_ids = test_df['ID'].copy()
    X_test = test_df.drop('ID', axis=1)
    if 'seq' in X_test.columns:
        X_test = X_test.drop('seq', axis=1)
    print(f"Test data shape: {X_test.shape}")

    # 4. 전처리 (피처 엔지니어링 포함)
    X_test_processed = preprocess_test_data(X_test, label_encoders)

    # 5. 예측
    print("\nMaking predictions...")
    dtest = xgb.DMatrix(X_test_processed)
    predictions = model.predict(dtest)
    print("Predictions generated.")

    # 6. 제출 파일 생성
    print("\nCreating submission file...")
    submission_df = pd.DataFrame({'ID': test_ids, 'clicked': predictions})

    # 파일명에 CV 점수 추가
    try:
        with open(results_path, 'r') as f:
            results_data = json.load(f)
            cv_score = results_data['cv_score_mean']
        print(f"CV Score: {cv_score:.6f}")
    except FileNotFoundError:
        cv_score = 0.0 # 결과 파일이 없으면 0.0
        print("No results file found, using CV score 0.0")

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    model_name = model_path.stem.replace('_best_model', '')
    filename = f'submission_{model_name}_cv{cv_score:.6f}_{timestamp}.csv'
    submission_path = SUBMISSIONS_DIR / filename

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")
    print(f"Submission stats:")
    print(f"  - Shape: {submission_df.shape}")
    print(f"  - Prediction range: {predictions.min():.6f} ~ {predictions.max():.6f}")
    print(f"  - Mean prediction: {predictions.mean():.6f}")
    print(f"\nInference pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()