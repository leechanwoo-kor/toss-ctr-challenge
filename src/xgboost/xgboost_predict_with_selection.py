"""
XGBoost K-Fold Ensemble Prediction with Feature Selection
Feature selection으로 학습된 모델용 예측 스크립트
"""

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


def load_selected_features(kfold_dir):
    """선택된 피처 리스트 로드"""
    selected_features_path = os.path.join(kfold_dir, "selected_features.txt")

    if not os.path.exists(selected_features_path):
        raise FileNotFoundError(
            f"selected_features.txt not found in {kfold_dir}\n"
            "This script is for feature-selected models only.\n"
            "Use 'xgboost_predict.py' for full-feature models."
        )

    print(f"✅ Loading selected features from: selected_features.txt")
    with open(selected_features_path, "r") as f:
        selected_features = [line.strip() for line in f if line.strip()]

    print(f"   Found {len(selected_features)} selected features")
    return selected_features


def load_kfold_models(kfold_dir):
    """K-fold 모델들 로드"""
    models = []
    model_files = [
        f
        for f in os.listdir(kfold_dir)
        if f.startswith("xgboost_fold_") and f.endswith(".json")
    ]
    model_files.sort()

    if not model_files:
        raise FileNotFoundError(f"No XGBoost fold models found in {kfold_dir}")

    print(f"\n📦 Loading {len(model_files)} fold models...")

    for model_file in model_files:
        model_path = os.path.join(kfold_dir, model_file)
        model = xgb.Booster()
        model.load_model(model_path)
        models.append(model)
        print(f"   ✓ Loaded: {model_file}")

    return models


def load_and_process_test_data(selected_features):
    """테스트 데이터 로드 및 선택된 피처로 필터링"""
    print(f"\n📁 Loading and processing test data...")

    try:
        # 원본 테스트 데이터 로드
        test = pd.read_parquet(TEST_DATA_PATH)
        test_ids = test["ID"].copy() if "ID" in test.columns else None

        if "ID" in test.columns:
            test = test.drop(columns=["ID"])

        print(f"   Original test data shape: {test.shape}")

        # 프로세서 로드
        with open(f"{DATA_DIR}/processor.pkl", "rb") as f:
            processor = pickle.load(f)

        print(f"   Processing test data with XGBoostFeatureProcessor...")

        # 전처리 (전체 115개 피처 생성)
        test_processed = processor.transform(test)

        print(f"   Processed data shape: {test_processed.shape}")

        # 선택된 피처로 필터링
        print(f"\n🎯 Applying feature selection...")
        available_features = [f for f in selected_features if f in test_processed.columns]
        missing_features = [f for f in selected_features if f not in test_processed.columns]

        if missing_features:
            print(f"   ⚠️  Warning: {len(missing_features)} features not found in test data")
            if len(missing_features) <= 5:
                print(f"   Missing: {missing_features}")
            else:
                print(f"   Missing: {missing_features[:5]}... (and {len(missing_features)-5} more)")

        # 선택된 피처만 추출 (순서 유지!)
        test_processed = test_processed[available_features]

        print(f"   ✓ Filtered to {len(available_features)} selected features")
        print(f"   Final test data shape: {test_processed.shape}")

        return test_processed, test_ids, available_features

    except Exception as e:
        raise RuntimeError(f"Failed to load and process test data: {e}")


def predict_kfold_ensemble(kfold_dir=None):
    """K-fold 앙상블 예측 (feature selection 모델용)"""
    print("🚀 XGBoost K-Fold Ensemble Prediction (Feature Selection)")
    print("=" * 70)

    try:
        # 체크포인트 디렉토리 찾기
        if kfold_dir is None:
            if not os.path.exists(CHECKPOINT_DIR):
                raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

            # feature selection 모델 찾기 (이름에 'selected' 포함)
            all_kfold_dirs = [d for d in os.listdir(CHECKPOINT_DIR) if "kfold" in d.lower()]
            selected_dirs = [d for d in all_kfold_dirs if "selected" in d.lower()]

            if not selected_dirs:
                print("\n⚠️  No feature-selected models found!")
                print("   Available K-fold directories:")
                for d in all_kfold_dirs[:5]:
                    print(f"      - {d}")
                raise FileNotFoundError(
                    "No feature-selected models found.\n"
                    "Use 'xgboost_predict.py' for full-feature models."
                )

            # 최신 feature-selected 모델 사용
            latest_kfold = max(
                selected_dirs,
                key=lambda x: os.path.getctime(os.path.join(CHECKPOINT_DIR, x))
            )
            kfold_dir = os.path.join(CHECKPOINT_DIR, latest_kfold)
        else:
            latest_kfold = os.path.basename(kfold_dir)

        print(f"📂 Using checkpoint: {latest_kfold}")

        # 선택된 피처 로드
        selected_features = load_selected_features(kfold_dir)

        # 모델 로드
        models = load_kfold_models(kfold_dir)

        # 테스트 데이터 로드 및 전처리
        test_processed, test_ids, feature_names = load_and_process_test_data(selected_features)

        # 예측 생성
        print(f"\n🔮 Generating ensemble predictions...")
        predictions = np.zeros(len(test_processed))

        # DMatrix 생성 (feature names 명시적 지정)
        dtest = xgb.DMatrix(test_processed.values, feature_names=feature_names)

        for i, model in enumerate(models):
            print(f"   Fold {i+1}/{len(models)}: predicting...", end="")

            # 예측 및 앙상블
            fold_pred = model.predict(dtest)
            predictions += fold_pred / len(models)

            print(f" ✓ (mean: {fold_pred.mean():.6f})")

        print(f"\n   ✓ Ensemble predictions generated: {len(predictions):,} samples")

        # 예측 결과 정규화
        predictions = np.clip(predictions, 0, 1)
        predictions = np.nan_to_num(predictions, nan=0.5)

        # Submission 파일 생성
        print(f"\n💾 Creating submission file...")
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

        # 파일명 생성 (피처 개수 포함)
        timestamp = datetime.now().strftime("%m%d_%H%M")
        n_features = len(feature_names)
        output_path = f"{SUBMISSION_DIR}/xgboost_kfold_{n_features}f_{timestamp}.csv"

        submit.to_csv(output_path, index=False)

        # 결과 출력
        print("\n" + "=" * 70)
        print("✅ Prediction Complete!")
        print("=" * 70)
        print(f"📁 Submission file: {output_path}")
        print(f"\n📊 Prediction Statistics:")
        print(f"   Samples: {len(predictions):,}")
        print(f"   Min: {predictions.min():.6f}")
        print(f"   Max: {predictions.max():.6f}")
        print(f"   Mean (CTR): {predictions.mean():.6f}")
        print(f"   Std: {predictions.std():.6f}")
        print(f"\n🎯 Feature Selection Info:")
        print(f"   Used {n_features} selected features")
        print(f"   Model: {latest_kfold}")
        print("=" * 70)

        return predictions, output_path

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="XGBoost K-Fold prediction with feature selection"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (auto-detect if not provided)",
    )

    args = parser.parse_args()

    try:
        predictions, output_path = predict_kfold_ensemble(args.checkpoint)
        print("\n✅ K-Fold ensemble prediction completed successfully!")
        print(f"📤 Ready to submit: {output_path}")
        return predictions, output_path
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        return None, None


if __name__ == "__main__":
    main()
