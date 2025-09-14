import time
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import glob
from pathlib import Path

# 중앙 전처리 함수 import
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
    print("🚀 Starting inference pipeline...")
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    # 1. 최신 모델 파일 찾기
    model_files = list(MODEL_DIR.glob('*-basic_features-*_best_model.json'))
    if not model_files:
        model_files = list(MODEL_DIR.glob('*_best_model.json'))
    
    if not model_files:
        raise FileNotFoundError("No model file found in models directory")
    
    model_path = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Using model: {model_path.name}")
    
    results_path = model_path.with_name(model_path.name.replace('_best_model.json', '_results.json'))
    
    # 2. 모델 및 인코더 로드
    print("\n📦 Loading model and encoders...")
    model = xgb.Booster()
    model.load_model(str(model_path))
    print("  Model loaded.")

    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    print("  Label encoders loaded.")

    # 3. 테스트 데이터 로드
    print(f"\n📦 Loading test data from {TEST_PATH}...")
    test_df = pd.read_parquet(TEST_PATH)
    
    test_ids = test_df['ID'].copy()
    X_test = test_df.drop('ID', axis=1)
    if 'seq' in X_test.columns:
        X_test = X_test.drop('seq', axis=1)
    print(f"Test data shape: {X_test.shape}")

    # 4. 중앙 전처리 함수를 사용하여 전처리
    X_test_processed, _ = preprocess(X_test, is_train=False, encoders=label_encoders)

    # 5. 예측
    print("\n🔮 Making predictions...")
    dtest = xgb.DMatrix(X_test_processed)
    predictions = model.predict(dtest)
    print("✅ Predictions generated.")

    # 6. 제출 파일 생성
    print("\n📄 Creating submission file...")
    submission_df = pd.DataFrame({'ID': test_ids, 'clicked': predictions})

    try:
        with open(results_path, 'r') as f:
            cv_score = json.load(f)['cv_score_mean']
        print(f"CV Score: {cv_score:.6f}")
    except (FileNotFoundError, KeyError):
        cv_score = 0.0
        print("No results file or score found, using CV score 0.0")

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    model_name = model_path.stem.replace('_best_model', '')
    filename = f'submission_{model_name}_cv{cv_score:.6f}_{timestamp}.csv'
    submission_path = SUBMISSIONS_DIR / filename

    submission_df.to_csv(submission_path, index=False)
    print(f"✅ Submission file saved to: {submission_path}")
    print(f"\n✅ Inference pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()