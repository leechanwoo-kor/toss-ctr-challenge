import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from datetime import datetime

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Constants
DATA_DIR = "data/processed/xgboost"
CHECKPOINT_DIR = "checkpoints/xgboost"
SUBMISSION_DIR = "submissions/xgboost"
TEST_DATA_PATH = "data/test.parquet"
SAMPLE_SUBMISSION_PATH = "data/sample_submission.csv"


def load_kfold_models(kfold_dir):
    """K-fold 모델들 로드"""
    models = []
    model_files = [
        f
        for f in os.listdir(kfold_dir)
        if f.startswith("xgboost_fold_") and f.endswith(".json")
    ]
    model_files.sort()

    print(f"Found {len(model_files)} fold models")

    for model_file in model_files:
        model_path = os.path.join(kfold_dir, model_file)
        model = xgb.Booster()
        model.load_model(model_path)
        models.append(model)
        print(f"  Loaded: {model_file}")

    return models


def load_test_data():
    """테스트 데이터 로드 및 전처리"""
    print("Loading test data...")

    try:
        # 원본 테스트 데이터 로드
        test = pd.read_parquet(TEST_DATA_PATH)
        test_ids = test["ID"].copy() if "ID" in test.columns else None

        if "ID" in test.columns:
            test = test.drop(columns=["ID"])

        print(f"Test data shape: {test.shape}")

        # 프로세서와 피처명 로드
        with open(f"{DATA_DIR}/processor.pkl", "rb") as f:
            processor = pickle.load(f)

        with open(f"{DATA_DIR}/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        print("Processing test data...")

        # XGBoostFeatureProcessor의 transform 사용 (일관된 전처리)
        test_processed = processor.transform(test)

        # 피처 순서 맞추기
        test_processed = test_processed[feature_names]

        print(f"Test data processed: {test_processed.shape}")

        return test_processed, test_ids, feature_names

    except Exception as e:
        raise RuntimeError(f"Failed to load and process test data: {e}")


def predict_kfold_ensemble():
    """K-fold 앙상블 예측"""
    print("Starting K-Fold ensemble prediction...")

    try:
        # 최신 K-fold 모델 디렉토리 찾기
        if not os.path.exists(CHECKPOINT_DIR):
            raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

        kfold_dirs = [d for d in os.listdir(CHECKPOINT_DIR) if "kfold" in d]

        if not kfold_dirs:
            raise FileNotFoundError("No K-fold models found. Run K-fold training first.")

        # 최신 K-fold 디렉토리
        latest_kfold = max(
            kfold_dirs, key=lambda x: os.path.getctime(os.path.join(CHECKPOINT_DIR, x))
        )
        kfold_dir = os.path.join(CHECKPOINT_DIR, latest_kfold)

        print(f"Using K-fold models from: {kfold_dir}")

        # 모델들 로드
        models = load_kfold_models(kfold_dir)

        # 테스트 데이터 로드
        test_processed, test_ids, feature_names = load_test_data()

        # 예측 생성 (메모리 효율적)
        print("Generating ensemble predictions...")
        predictions = np.zeros(len(test_processed))

        # DMatrix를 한 번만 생성 (메모리 효율성)
        dtest = xgb.DMatrix(test_processed.values, feature_names=feature_names)

        for i, model in enumerate(models):
            print(f"  Predicting with fold {i+1}/{len(models)}...")

            # 예측 및 집계
            fold_pred = model.predict(dtest)
            predictions += fold_pred / len(models)

        print(f"Ensemble predictions generated: {len(predictions):,}")

        # 예측 결과 정규화
        predictions = np.clip(predictions, 0, 1)
        predictions = np.nan_to_num(predictions, nan=0.5)

        # Submission 파일 생성
        print("Creating submission file...")
        os.makedirs(SUBMISSION_DIR, exist_ok=True)

        # 제출 데이터 생성
        if os.path.exists(SAMPLE_SUBMISSION_PATH):
            submit = pd.read_csv(SAMPLE_SUBMISSION_PATH)
            submit["clicked"] = predictions
        else:
            # 샘플이 없으면 직접 생성
            if test_ids is not None:
                submit = pd.DataFrame({"ID": test_ids, "clicked": predictions})
            else:
                submit = pd.DataFrame(
                    {"ID": range(len(predictions)), "clicked": predictions}
                )

        # 파일명 생성 및 저장
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_path = f"{SUBMISSION_DIR}/xgboost_kfold_{timestamp}.csv"
        submit.to_csv(output_path, index=False)

        # 결과 출력
        print(f"\nSubmission saved: {output_path}")
        print(f"Prediction stats:")
        print(f"  Count: {len(predictions):,}")
        print(f"  Min: {predictions.min():.6f}")
        print(f"  Max: {predictions.max():.6f}")
        print(f"  Mean: {predictions.mean():.6f}")
        print(f"  Std: {predictions.std():.6f}")

        return predictions, output_path

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")


def main():
    """메인 실행 함수"""
    try:
        predictions, output_path = predict_kfold_ensemble()
        print("\nK-Fold ensemble prediction completed successfully!")
        return predictions, output_path
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    main()
