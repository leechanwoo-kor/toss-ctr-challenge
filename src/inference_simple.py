"""
단순화된 추론 파이프라인
NVIDIA Merlin 접근법 기반 - 최소한의 전처리
성능: 0.352247 기반 모델 사용
"""
import pandas as pd
import numpy as np
import pickle
import time
import json
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from utils import print_memory, clear_memory
from config import (
    TEST_PATH, ENCODERS_PATH, SUBMISSIONS_DIR,
    MODEL_SHARED_DIR, CATEGORICAL_FEATURES
)

def get_core_features():
    """NVIDIA Merlin 방식: 117개 핵심 피처만 사용"""
    # Use categorical features from config
    categorical_features = CATEGORICAL_FEATURES.copy()

    # CONTINUOUS COLUMNS (112개 - 원본 피처들만)
    continuous_features = (
        [f'feat_a_{i}' for i in range(1, 19)] +  # 18
        [f'feat_b_{i}' for i in range(1, 7)] +   # 6
        [f'feat_c_{i}' for i in range(1, 9)] +   # 8
        [f'feat_d_{i}' for i in range(1, 7)] +   # 6
        [f'feat_e_{i}' for i in range(1, 11)] +  # 10
        [f'history_a_{i}' for i in range(1, 8)] +  # 7
        [f'history_b_{i}' for i in range(1, 31)] + # 30
        [f'l_feat_{i}' for i in range(1, 28)]      # 27
    )

    core_features = categorical_features + continuous_features
    return core_features

def find_best_simple_model():
    """MODEL_SHARED_DIR에서 최고 성능 simple 모델 찾기"""
    print("[MODEL_SEARCH] Searching for best simple model...")

    # simple 모델 파일들 찾기
    model_files = list(MODEL_SHARED_DIR.glob('simple_model_*.json'))
    if not model_files:
        # 일반 모델 파일들도 찾기
        model_files = list(MODEL_SHARED_DIR.glob('*.json'))
        model_files = [f for f in model_files if not f.name.endswith('_results.json') and not f.name.endswith('_params.json')]

    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODEL_SHARED_DIR}")

    # 가장 최신 모델 선택 (또는 파일명에서 CV 점수가 가장 높은 것)
    best_model = None
    best_score = 0.0

    for model_file in model_files:
        try:
            # 파일명에서 CV 점수 추출 시도
            if '_cv' in model_file.name:
                cv_part = model_file.name.split('_cv')[1].split('.')[0]
                score = float(cv_part)
                if score > best_score:
                    best_score = score
                    best_model = model_file
            else:
                # CV 점수가 없으면 최신 파일 사용
                if best_model is None:
                    best_model = max(model_files, key=lambda p: p.stat().st_mtime)
        except:
            continue

    if best_model is None:
        best_model = max(model_files, key=lambda p: p.stat().st_mtime)

    print(f"  Selected model: {best_model}")
    if best_score > 0:
        print(f"  CV Score: {best_score:.6f}")

    return best_model

def process_categorical_features_inference(df, label_encoders):
    """Test 데이터용 범주형 피처 처리"""
    print("  [CATEGORICAL] Processing with trained encoders...")

    for feature in CATEGORICAL_FEATURES:
        if feature in df.columns and feature in label_encoders:
            encoder = label_encoders[feature]
            known_classes = set(encoder.classes_)

            # 결측치를 "missing"으로 처리
            df[feature] = df[feature].fillna("missing")

            # 학습 시 보지 못한 값들을 안전하게 처리
            # "rare"가 학습 데이터에 있는지 확인
            if "rare" in known_classes:
                # "rare"가 있으면 unknown 값들을 "rare"로 변환
                df[feature] = df[feature].apply(
                    lambda x: x if x in known_classes else "rare"
                )
            else:
                # "rare"가 없으면 가장 빈도가 높은 값으로 대체
                most_common = encoder.classes_[0]  # 첫 번째는 보통 가장 빈도가 높음
                df[feature] = df[feature].apply(
                    lambda x: x if x in known_classes else most_common
                )
                print(f"    Warning: Using '{most_common}' for unknown values in {feature}")

            # Label Encoding 적용
            df[feature] = encoder.transform(df[feature].astype(str))

    print(f"    Processed {len(CATEGORICAL_FEATURES)} categorical features")
    return df

def simple_test_preprocessing(df):
    """단순한 테스트 데이터 전처리"""
    print("  [PREPROCESSING] Applying simple preprocessing...")

    # 1. 메모리 최적화
    float64_cols = df.select_dtypes(include=['float64']).columns
    for col in float64_cols:
        if 'id' not in col.lower():
            df[col] = df[col].astype('float32')

    int64_cols = df.select_dtypes(include=['int64']).columns
    for col in int64_cols:
        if 'id' not in col.lower():
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # 2. 결측값 처리 (NVIDIA Merlin 방식: 0으로 채우기)
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"    Filling {missing_count:,} missing values with 0")
        df = df.fillna(0)

    return df

def simple_inference():
    """단순화된 추론 파이프라인"""
    print("="*60)
    print("[SIMPLE_INFERENCE] NVIDIA Merlin Style Inference")
    print("="*60)
    print("Goal: Generate predictions using simple 117-feature model")
    print("Method: Minimal preprocessing, optimal performance")

    start_time = time.time()
    print_memory("Initial")

    # 디렉토리 생성
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 모델 로드
    model_path = find_best_simple_model()
    print(f"\n[MODEL_LOADING] Loading model...")
    model = xgb.Booster()
    model.load_model(str(model_path))
    print(f"  Model loaded from: {model_path}")

    # 2. 인코더 로드
    print(f"\n[ENCODERS] Loading label encoders...")
    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    print(f"  Encoders loaded from: {ENCODERS_PATH}")

    # 3. 테스트 데이터 로드
    print(f"\n[DATA_LOADING] Loading test data...")
    test_df = pd.read_parquet(TEST_PATH)
    print(f"  Test data shape: {test_df.shape}")
    print_memory("After loading")

    # ID 분리
    test_ids = test_df['ID'].copy()
    X_test = test_df.drop('ID', axis=1)

    # seq 컬럼 제거 (사용하지 않음)
    if 'seq' in X_test.columns:
        X_test = X_test.drop('seq', axis=1)
        print("    Removed 'seq' column")

    # 4. 핵심 피처만 선택
    print(f"\n[FEATURE_SELECTION] Selecting core features...")
    core_features = get_core_features()

    # 존재하는 피처만 선택
    available_features = [f for f in core_features if f in X_test.columns]
    missing_features = [f for f in core_features if f not in X_test.columns]

    if missing_features:
        print(f"  Warning: Missing {len(missing_features)} features in test data")

    X_test_core = X_test[available_features].copy()
    print(f"  Selected {len(available_features)} available features")
    print(f"  Test core shape: {X_test_core.shape}")

    # 메모리 정리
    del X_test, test_df
    clear_memory()

    # 5. 단순 전처리
    X_test_core = simple_test_preprocessing(X_test_core)

    # 6. 범주형 피처 처리
    X_test_core = process_categorical_features_inference(X_test_core, label_encoders)

    print(f"  Final test shape: {X_test_core.shape}")
    print_memory("After preprocessing")

    # 7. 예측
    print(f"\n[PREDICTION] Making predictions...")
    dtest = xgb.DMatrix(X_test_core)
    predictions = model.predict(dtest)
    print(f"  Generated {len(predictions):,} predictions")

    # 예측값 통계
    print(f"  Prediction stats:")
    print(f"    Min: {predictions.min():.6f}")
    print(f"    Max: {predictions.max():.6f}")
    print(f"    Mean: {predictions.mean():.6f}")
    print(f"    Std: {predictions.std():.6f}")

    # 8. 제출 파일 생성
    print(f"\n[SUBMISSION] Creating submission file...")
    submission_df = pd.DataFrame({'ID': test_ids, 'clicked': predictions})

    # CV 점수 추출 (파일명에서)
    try:
        if '_cv' in model_path.name:
            cv_score = float(model_path.name.split('_cv')[1].split('.')[0])
        else:
            cv_score = 0.0
    except:
        cv_score = 0.0

    # 제출 파일명 생성
    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    if cv_score > 0:
        filename = f'submission_simple_{timestamp}_cv{cv_score:.4f}.csv'
    else:
        filename = f'submission_simple_{timestamp}.csv'

    submission_path = SUBMISSIONS_DIR / filename
    submission_df.to_csv(submission_path, index=False)

    print(f"  Submission saved to: {submission_path}")
    print(f"  Submission shape: {submission_df.shape}")

    # 제출 파일 검증
    print(f"\n[VALIDATION] Validating submission...")
    print(f"  ID count: {submission_df['ID'].nunique():,}")
    print(f"  Prediction range: [{submission_df['clicked'].min():.6f}, {submission_df['clicked'].max():.6f}]")
    print(f"  Missing values: {submission_df.isnull().sum().sum()}")

    print(f"\n[TIMING] Inference completed in {time.time() - start_time:.1f}s")
    print_memory("Final")

    return submission_path, cv_score

def main():
    """메인 실행 함수"""
    submission_path, cv_score = simple_inference()

    print("\n" + "="*60)
    print("[SUCCESS] Simple inference completed!")
    print(f"Submission file: {submission_path}")
    if cv_score > 0:
        print(f"Expected performance: CV {cv_score:.4f}")
    print("Ready for competition submission!")
    print("="*60)

    return submission_path

if __name__ == '__main__':
    main()