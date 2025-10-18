"""
Toss CTR Challenge - Final Inference Script
Private Score 재현을 위한 추론 코드

최종 제출: XGBoost v6 (65%) + DeepFM v2.2 (35%) 앙상블
결과 파일: ensemble_xgb_deepfm_65_35_1012_1728.csv

Author: [Your Name]
Date: 2025-10-12
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
import xgboost as xgb
from datetime import datetime

# 프로젝트 루트 경로 설정 (상대 경로)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.models.deepfm_optimized import OptimizedDeepFM
from src.deepfm.deepfm_processor import DeepFMProcessor
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# 설정 (Random Seed 고정)
# ============================================================================

RANDOM_SEED = 42

def set_random_seed(seed=42):
    """재현성을 위한 랜덤 시드 고정"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(RANDOM_SEED)

# ============================================================================
# 경로 설정 (상대 경로)
# ============================================================================

# 데이터 경로 (parquet 형식)
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "data/test.parquet")

# 모델 가중치 경로 (제출 패키지 내 간소화된 구조)
XGBOOST_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/xgboost")
DEEPFM_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/deepfm")
DEEPFM_PROCESSOR_PATH = os.path.join(PROJECT_ROOT, "data/processed/deepfm_new/processor.pkl")

# 출력 경로
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "submissions/final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 앙상블 가중치
XGBOOST_WEIGHT = 0.65
DEEPFM_WEIGHT = 0.35

# ============================================================================
# 1. XGBoost v6 예측
# ============================================================================

def predict_xgboost(test_data_path, checkpoint_dir):
    """
    XGBoost v6 K-Fold 모델로 예측 생성

    Args:
        test_data_path: 테스트 데이터 경로
        checkpoint_dir: XGBoost 체크포인트 디렉토리

    Returns:
        predictions: numpy array [N,] - 예측 확률
        ids: numpy array [N,] - ID 컬럼
    """
    print("=" * 70)
    print("1. XGBoost v6 Prediction")
    print("=" * 70)

    # 테스트 데이터 로드
    print(f"Loading test data from: {test_data_path}")
    test_df = pd.read_parquet(test_data_path)
    print(f"Test samples: {len(test_df):,}")

    # ID 저장
    test_ids = test_df['ID'].values

    # 피처 준비 (XGBoost 학습 시와 동일한 전처리)
    exclude_cols = ['ID', 'clicked', 'seq']
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]

    X_test = test_df[feature_cols].copy()

    # Categorical features를 numeric으로 변환 (LabelEncoding)
    categorical_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    print(f"Encoding {len(categorical_cols)} categorical features...")
    for col in categorical_cols:
        if col in X_test.columns:
            X_test[col] = pd.Categorical(X_test[col]).codes

    # K-Fold 모델 로드 (JSON 형식)
    print(f"Loading XGBoost models from: {checkpoint_dir}")
    fold_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('xgboost_fold_') and f.endswith('.json')])
    print(f"Found {len(fold_files)} fold models")

    fold_predictions = []

    for fold_file in fold_files:
        fold_path = os.path.join(checkpoint_dir, fold_file)
        print(f"Loading {fold_file}...")

        # JSON 형식으로 로드
        model = xgb.Booster()
        model.load_model(fold_path)

        dtest = xgb.DMatrix(X_test)
        pred = model.predict(dtest)
        fold_predictions.append(pred)
        print(f"  Predictions: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")

    # 평균
    xgb_predictions = np.mean(fold_predictions, axis=0)

    print(f"XGBoost ensemble: min={xgb_predictions.min():.4f}, max={xgb_predictions.max():.4f}, mean={xgb_predictions.mean():.4f}")
    print()

    return xgb_predictions, test_ids


# ============================================================================
# 2. DeepFM v2.2 예측
# ============================================================================

def predict_deepfm(test_data_path, checkpoint_dir, processor_path):
    """
    DeepFM v2.2 K-Fold 모델로 예측 생성

    Args:
        test_data_path: 테스트 데이터 경로
        checkpoint_dir: DeepFM 체크포인트 디렉토리
        processor_path: DeepFM 프로세서 경로

    Returns:
        predictions: numpy array [N,] - 예측 확률
    """
    print("=" * 70)
    print("2. DeepFM v2.2 Prediction")
    print("=" * 70)

    # 프로세서 로드
    print(f"Loading DeepFM processor from: {processor_path}")
    processor = DeepFMProcessor.load(processor_path)

    # 테스트 데이터 로드 및 전처리
    print(f"Loading and preprocessing test data...")
    test_df = pd.read_parquet(test_data_path)

    # DeepFM 전처리
    test_categorical, test_numerical = processor.transform(test_df, is_train=False)

    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # K-Fold 모델 로드 및 예측
    fold_predictions = []

    # 체크포인트 디렉토리에서 모든 fold 모델 찾기
    fold_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('fold_') and f.endswith('_model.pth')])
    print(f"Found {len(fold_files)} fold models")

    for fold_file in fold_files:
        fold_path = os.path.join(checkpoint_dir, fold_file)
        print(f"Loading {fold_file}...")

        # 체크포인트 로드
        checkpoint = torch.load(fold_path, map_location=device)

        # 모델 생성
        model = OptimizedDeepFM(
            feature_dims=processor.categorical_feature_dims,
            default_embedding_dim=16,
            hidden_units=[256, 128],
            dropout=0.25,
            num_numerical=len(processor.numerical_features)
        ).to(device)

        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 배치 예측
        batch_size = 1024
        predictions = []

        with torch.no_grad():
            for start_idx in range(0, len(test_categorical[list(test_categorical.keys())[0]]), batch_size):
                end_idx = min(start_idx + batch_size, len(test_categorical[list(test_categorical.keys())[0]]))

                # 배치 데이터 준비
                batch_categorical = {}
                for col, data in test_categorical.items():
                    batch_categorical[col] = data[start_idx:end_idx].to(device)

                batch_numerical = None
                if test_numerical is not None:
                    batch_numerical = test_numerical[start_idx:end_idx].to(device)

                # 예측
                outputs = model(batch_categorical, batch_numerical)
                batch_predictions = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                predictions.extend(batch_predictions)

        fold_predictions.append(np.array(predictions))
        print(f"  Predictions: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")

        # 메모리 정리
        del model
        torch.cuda.empty_cache()

    # 모든 fold의 평균
    deepfm_predictions = np.mean(fold_predictions, axis=0)

    print(f"DeepFM ensemble: min={deepfm_predictions.min():.4f}, max={deepfm_predictions.max():.4f}, mean={deepfm_predictions.mean():.4f}")
    print()

    return deepfm_predictions


# ============================================================================
# 3. 앙상블
# ============================================================================

def ensemble_predictions(xgb_preds, deepfm_preds, xgb_weight=0.65, deepfm_weight=0.35):
    """
    XGBoost와 DeepFM 예측을 가중 평균으로 앙상블

    Args:
        xgb_preds: XGBoost 예측
        deepfm_preds: DeepFM 예측
        xgb_weight: XGBoost 가중치
        deepfm_weight: DeepFM 가중치

    Returns:
        ensemble_preds: 앙상블 예측
    """
    print("=" * 70)
    print("3. Ensemble")
    print("=" * 70)

    print(f"Ensemble weights: XGBoost={xgb_weight:.0%}, DeepFM={deepfm_weight:.0%}")

    ensemble_preds = xgb_weight * xgb_preds + deepfm_weight * deepfm_preds

    print(f"Ensemble result: min={ensemble_preds.min():.4f}, max={ensemble_preds.max():.4f}, mean={ensemble_preds.mean():.4f}")
    print()

    return ensemble_preds


# ============================================================================
# 4. 제출 파일 생성
# ============================================================================

def create_submission(ids, predictions, output_path):
    """
    제출 파일 생성

    Args:
        ids: ID 컬럼
        predictions: 예측 확률
        output_path: 출력 파일 경로
    """
    print("=" * 70)
    print("4. Creating Submission File")
    print("=" * 70)

    submission = pd.DataFrame({
        'ID': ids,
        'clicked': predictions
    })

    submission.to_csv(output_path, index=False)

    print(f"Submission file saved: {output_path}")
    print(f"Total predictions: {len(submission):,}")
    print(f"Prediction stats:")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print()


# ============================================================================
# Main 실행
# ============================================================================

def main():
    """메인 추론 파이프라인"""
    print("\n" + "=" * 70)
    print("Toss CTR Challenge - Final Inference")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random Seed: {RANDOM_SEED}")
    print()

    # 1. XGBoost 예측
    xgb_preds, test_ids = predict_xgboost(TEST_DATA_PATH, XGBOOST_CHECKPOINT_DIR)

    # 2. DeepFM 예측
    deepfm_preds = predict_deepfm(TEST_DATA_PATH, DEEPFM_CHECKPOINT_DIR, DEEPFM_PROCESSOR_PATH)

    # 3. 앙상블
    ensemble_preds = ensemble_predictions(xgb_preds, deepfm_preds, XGBOOST_WEIGHT, DEEPFM_WEIGHT)

    # 4. 제출 파일 생성
    output_filename = f"ensemble_xgb_deepfm_{int(XGBOOST_WEIGHT*100)}_{int(DEEPFM_WEIGHT*100)}_final.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    create_submission(test_ids, ensemble_preds, output_path)

    print("=" * 70)
    print("Inference Complete!")
    print("=" * 70)
    print(f"Output: {output_path}")
    print()


if __name__ == "__main__":
    main()
