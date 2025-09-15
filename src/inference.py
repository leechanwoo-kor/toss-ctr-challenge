import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
import json
import xgboost as xgb

from data_processing import (
    optimize_memory_usage,
    handle_feat_e_3_critical,
    handle_outliers_systematic,
    improved_missing_handler,
    create_temporal_features,
    create_inventory_features,
    create_user_segment_features
)
from config import (
    TEST_PATH, ENCODERS_PATH, SUBMISSIONS_DIR,
    MODEL_EXPERIMENTS_DIR, MODEL_PRODUCTION_DIR,
    CATEGORICAL_FEATURES
)

# 추론 전용 설정

# CATEGORICAL_FEATURES는 config에서 가져옴

def process_categorical_features_inference(df, label_encoders):
    """Test 데이터용 범주형 피처 처리"""
    print("    Processing categorical features with trained encoders...")

    for feature in CATEGORICAL_FEATURES:
        if feature in df.columns and feature in label_encoders:
            encoder = label_encoders[feature]
            known_classes = set(encoder.classes_)

            # 결측치를 "missing"으로 처리
            df[feature] = df[feature].fillna("missing")

            # 학습 시 보지 못한 값들을 "rare"로 처리
            df[feature] = df[feature].apply(
                lambda x: x if x in known_classes else "rare"
            )

            # Label Encoding 적용
            df[feature] = encoder.transform(df[feature].astype(str))

    print(f"    Processed {len(CATEGORICAL_FEATURES)} categorical features")
    return df

# --- 메인 추론 로직 ---
def find_best_model():
    """구조화된 저장소에서 최고 성능 모델 찾기"""
    print("Searching for best model in structured storage...")

    # 1. Production 모델 우선 찾기
    production_models = list(MODEL_PRODUCTION_DIR.glob('**/*.json'))
    if production_models:
        model_path = max(production_models, key=lambda p: p.stat().st_mtime)
        print(f"Found production model: {model_path}")
        return model_path

    # 2. Experiments 폴더에서 최신 모델 찾기
    experiment_dirs = [d for d in MODEL_EXPERIMENTS_DIR.iterdir() if d.is_dir()]
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in {MODEL_EXPERIMENTS_DIR}")

    # 최신 실험 디렉토리 찾기
    latest_exp_dir = max(experiment_dirs, key=lambda d: d.stat().st_mtime)

    # 실험 디렉토리에서 모델 파일 찾기
    model_files = list(latest_exp_dir.glob('*.json'))
    model_files = [f for f in model_files if not f.name.endswith('results.json')]

    if not model_files:
        raise FileNotFoundError(f"No model file found in {latest_exp_dir}")

    model_path = model_files[0]  # 실험 디렉토리에는 하나의 모델만 있음
    print(f"Found experiment model: {model_path}")
    return model_path

def get_model_results(model_path):
    """모델에 대응하는 결과 파일 찾기"""
    # 모델과 같은 디렉토리에서 results.json 찾기
    results_path = model_path.parent / 'results.json'
    return results_path if results_path.exists() else None

def main():
    """메인 추론 파이프라인"""
    start_time = time.time()
    print("Starting inference pipeline...")
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    # 1. 구조화된 저장소에서 최고 모델 찾기
    model_path = find_best_model()
    results_path = get_model_results(model_path)
    
    # 2. 모델 및 인코더 로드
    print("\nLoading model and encoders...")
    model = xgb.Booster()
    model.load_model(str(model_path))
    print("  Model loaded.")

    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    print("  Label encoders loaded.")

    # 3. 테스트 데이터 로드 및 전처리
    print(f"\nLoading and preprocessing test data from {TEST_PATH}...")
    test_df = pd.read_parquet(TEST_PATH)
    
    test_ids = test_df['ID'].copy()
    X_test = test_df.drop('ID', axis=1)
    if 'seq' in X_test.columns:
        X_test = X_test.drop('seq', axis=1)
    
    # Enhanced preprocessing pipeline (same as training)
    print("  Applying enhanced preprocessing...")

    # 1. Memory optimization
    X_test = optimize_memory_usage(X_test)

    # 2. feat_e_3 critical handling
    X_test = handle_feat_e_3_critical(X_test)

    # 3. Systematic outlier handling
    X_test = handle_outliers_systematic(X_test)

    # 4. Improved missing value handling
    X_test = improved_missing_handler(X_test)

    # 5. Temporal features
    X_test = create_temporal_features(X_test)

    # 6. Inventory features
    X_test = create_inventory_features(X_test)

    # 7. Categorical features (using trained encoders)
    X_test = process_categorical_features_inference(X_test, label_encoders)

    # 8. User segment features (after categorical encoding)
    X_test = create_user_segment_features(X_test)

    print(f"  Processed test data shape: {X_test.shape}")

    # 4. 예측
    print("\nMaking predictions...")
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)
    print("Predictions generated.")

    # 5. 제출 파일 생성
    print("\nCreating submission file...")
    submission_df = pd.DataFrame({'ID': test_ids, 'clicked': predictions})

    # 5. CV 점수 로드
    if results_path and results_path.exists():
        try:
            with open(results_path, 'r') as f:
                cv_score = json.load(f)['cv_score_mean']
            print(f"CV Score from results: {cv_score:.6f}")
        except (FileNotFoundError, KeyError):
            cv_score = 0.0
            print("Could not find CV score, setting to 0.0")
    else:
        cv_score = 0.0
        print("No results file found, setting CV score to 0.0")

    # 제출 파일명 생성 (실험 디렉토리 이름 포함)
    experiment_name = model_path.parent.name
    filename = f'submission-{experiment_name}-cv{cv_score:.4f}.csv'
    submission_path = SUBMISSIONS_DIR / filename

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")
    print(f"\nInference pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
