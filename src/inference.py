import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
import json
import xgboost as xgb

from data_processing import preprocess

# --- 설정 ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'

TEST_PATH = DATA_DIR / 'test.parquet'
ENCODERS_PATH = MODEL_DIR / 'label_encoders.pkl'

# --- 메인 추론 로직 ---
def main():
    """메인 추론 파이프라인"""
    start_time = time.time()
    print("Starting inference pipeline...")
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    # 1. 최신 모델 파일 찾기 (새로운 명명 규칙 적용)
    # 우선순위: 튜닝된 모델 > 베이스라인 모델
    model_files = list(MODEL_DIR.glob('xgb-v*-tuned-*.json'))
    if not model_files:
        model_files = list(MODEL_DIR.glob('xgb-v*-baseline-*.json'))
    if not model_files:
        model_files = list(MODEL_DIR.glob('*.json')) # Fallback

    if not model_files:
        raise FileNotFoundError(f"No model file found in {MODEL_DIR}")
    
    model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"Using model: {model_path.name}")
    
    # 해당 모델의 결과 파일 찾기
    results_path = model_path.with_name(model_path.name.replace('.json', '_results.json'))
    
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
    
    X_test_processed, _ = preprocess(X_test, is_train=False, encoders=label_encoders)

    # 4. 예측
    print("\nMaking predictions...")
    dtest = xgb.DMatrix(X_test_processed)
    predictions = model.predict(dtest)
    print("Predictions generated.")

    # 5. 제출 파일 생성
    print("\nCreating submission file...")
    submission_df = pd.DataFrame({'ID': test_ids, 'clicked': predictions})

    try:
        with open(results_path, 'r') as f:
            cv_score = json.load(f)['cv_score_mean']
        print(f"CV Score from results: {cv_score:.6f}")
    except (FileNotFoundError, KeyError):
        cv_score = 0.0
        print("Could not find CV score, setting to 0.0")

    # 새로운 제출 파일명 규칙
    model_name_stem = model_path.stem
    filename = f'submission-{model_name_stem}-cv{cv_score:.4f}.csv'
    submission_path = SUBMISSIONS_DIR / filename

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")
    print(f"\nInference pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
