"""
Toss CTR Challenge - 본선 제출용 추론(Inference) 스크립트
XGBoost + DeepFM 2-Model Ensemble Inference

사용법:
    python inference.py

요구사항:
    - data/test.parquet: 테스트 데이터
    - checkpoints/xgboost/: XGBoost K-fold 모델 가중치
    - checkpoints/deepfm_kfold/: DeepFM K-fold 모델 가중치
    - data/processed/xgboost/: XGBoost 전처리 객체 (processor.pkl, feature_names.pkl)
    - data/processed/deepfm_new/: DeepFM 전처리 객체 (metadata.pkl)

출력:
    - submissions/ensemble/inference_output_MMDD_HHMM.csv
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle

# GPU 감지
try:
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    DEVICE = None

try:
    import xgboost as xgb
except ImportError:
    pass

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# =============================================================================
# 경로 설정
# =============================================================================
TEST_DATA_PATH = PROJECT_ROOT / 'data' / 'test.parquet'

# XGBoost 관련 경로
XGBOOST_CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints' / 'xgboost'
XGBOOST_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'xgboost'

# DeepFM 관련 경로
DEEPFM_CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints' / 'deepfm_kfold'
DEEPFM_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'deepfm_new'

# 출력 경로
SUBMISSION_DIR = PROJECT_ROOT / 'submissions' / 'ensemble'
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# XGBoost 추론 함수
# =============================================================================
def load_xgboost_models(checkpoint_dir):
    """XGBoost K-fold 모델들 로드"""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"XGBoost checkpoint directory not found: {checkpoint_dir}")

    # K-fold 디렉토리 찾기
    kfold_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and 'kfold' in d.name]

    if not kfold_dirs:
        raise FileNotFoundError(f"No XGBoost K-fold models found in {checkpoint_dir}")

    # 최신 K-fold 디렉토리
    latest_kfold_dir = max(kfold_dirs, key=lambda x: x.stat().st_mtime)

    # 모델 파일들 로드
    model_files = sorted([f for f in latest_kfold_dir.glob('xgboost_fold_*.json')])

    if not model_files:
        raise FileNotFoundError(f"No model files found in {latest_kfold_dir}")

    models = []
    for model_file in model_files:
        model = xgb.Booster()
        model.load_model(str(model_file))
        models.append(model)

    return models


def predict_xgboost(test_data_path, checkpoint_dir, processed_dir):
    """XGBoost K-fold 앙상블 예측"""
    # 모델 로드
    models = load_xgboost_models(checkpoint_dir)

    # 테스트 데이터 로드
    test = pd.read_parquet(test_data_path)
    test_ids = test['ID'].copy() if 'ID' in test.columns else None

    if 'ID' in test.columns:
        test = test.drop(columns=['ID'])

    # 전처리 객체 로드
    with open(processed_dir / 'processor.pkl', 'rb') as f:
        processor = pickle.load(f)

    with open(processed_dir / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    # 데이터 전처리
    test_processed = processor.transform(test)
    test_processed = test_processed[feature_names]

    # 예측 생성
    predictions = np.zeros(len(test_processed))
    dtest = xgb.DMatrix(test_processed.values, feature_names=feature_names)

    for i, model in enumerate(models, 1):
        fold_pred = model.predict(dtest)
        predictions += fold_pred / len(models)

    # 후처리
    predictions = np.clip(predictions, 0, 1)
    predictions = np.nan_to_num(predictions, nan=0.5)

    return predictions, test_ids


# =============================================================================
# DeepFM 추론 함수
# =============================================================================
def load_deepfm_models(checkpoint_dir):
    """DeepFM K-fold 모델들 로드"""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"DeepFM checkpoint directory not found: {checkpoint_dir}")

    # 모델 파일들 찾기 (하위 디렉토리 포함)
    model_files = sorted(checkpoint_dir.glob('**/*fold_*.pth'))

    if not model_files:
        raise FileNotFoundError(f"No DeepFM model files found in {checkpoint_dir}")

    # 모델 아키텍처 import
    from models.deepfm_optimized import OptimizedDeepFM
    from models.deepfm_dynamic_embedding import DynamicEmbeddingDeepFM

    models = []
    model_type = None

    for model_file in model_files:
        checkpoint = torch.load(model_file, map_location=DEVICE)

        # 모델 타입 감지
        if model_type is None:
            if 'use_dynamic_embedding' in checkpoint['config'] and checkpoint['config']['use_dynamic_embedding']:
                model_type = 'dynamic'
            else:
                model_type = 'optimized'

        # 모델 생성
        if model_type == 'dynamic':
            model = DynamicEmbeddingDeepFM(
                categorical_features=checkpoint['categorical_features'],
                numerical_features=checkpoint['numerical_features'],
                config=checkpoint['config']
            )
        else:
            model = OptimizedDeepFM(
                categorical_features=checkpoint['categorical_features'],
                numerical_features=checkpoint['numerical_features'],
                config=checkpoint['config']
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        models.append(model)

    return models, checkpoint['categorical_features'], checkpoint['numerical_features']


def predict_deepfm(test_data_path, checkpoint_dir, processed_dir):
    """DeepFM K-fold 앙상블 예측"""
    # 모델 로드
    models, cat_features, num_features = load_deepfm_models(checkpoint_dir)

    # 테스트 데이터 로드
    test = pd.read_parquet(test_data_path)
    test_ids = test['ID'].copy() if 'ID' in test.columns else None

    if 'ID' in test.columns:
        test = test.drop(columns=['ID'])

    # 전처리 객체 로드
    with open(processed_dir / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    from deepfm.deepfm_processor import DeepFMProcessor
    processor = DeepFMProcessor(categorical_features=cat_features)
    processor.encoders = metadata['encoders']
    processor.feature_dims = metadata['feature_dims']

    # 데이터 전처리
    test_processed = processor.transform(test)

    # PyTorch 데이터셋 생성
    X_cat_tensor = torch.LongTensor(test_processed[cat_features].values).to(DEVICE)
    X_num_tensor = torch.FloatTensor(test_processed[num_features].values).to(DEVICE)

    # 예측 생성
    predictions = np.zeros(len(test_processed))

    batch_size = 4096
    num_batches = (len(test_processed) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i, model in enumerate(models, 1):
            fold_preds = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(test_processed))

                batch_cat = X_cat_tensor[start_idx:end_idx]
                batch_num = X_num_tensor[start_idx:end_idx]

                outputs = model(batch_cat, batch_num)
                fold_preds.append(outputs.cpu().numpy())

            fold_preds = np.concatenate(fold_preds)
            predictions += fold_preds.flatten() / len(models)

    # 후처리
    predictions = np.clip(predictions, 0, 1)
    predictions = np.nan_to_num(predictions, nan=0.5)

    return predictions, test_ids


# =============================================================================
# 앙상블 함수
# =============================================================================
def ensemble_predictions(xgb_preds, deepfm_preds, xgb_weight=0.7, deepfm_weight=0.3):
    """XGBoost와 DeepFM 예측을 가중치 앙상블"""
    ensemble_preds = xgb_weight * xgb_preds + deepfm_weight * deepfm_preds
    ensemble_preds = np.clip(ensemble_preds, 0, 1)

    return ensemble_preds


# =============================================================================
# 메인 추론 함수
# =============================================================================
def main():
    """본선 제출용 메인 추론 함수"""
    try:
        # XGBoost 예측
        xgb_preds, test_ids = predict_xgboost(
            test_data_path=TEST_DATA_PATH,
            checkpoint_dir=XGBOOST_CHECKPOINT_DIR,
            processed_dir=XGBOOST_PROCESSED_DIR
        )

        # DeepFM 예측
        deepfm_preds, _ = predict_deepfm(
            test_data_path=TEST_DATA_PATH,
            checkpoint_dir=DEEPFM_CHECKPOINT_DIR,
            processed_dir=DEEPFM_PROCESSED_DIR
        )

        # 앙상블
        # XGBoost Public: 0.348568, DeepFM v2.2 Public: 0.346596
        # 최적 가중치: XGBoost 70%, DeepFM 30%
        ensemble_preds = ensemble_predictions(
            xgb_preds=xgb_preds,
            deepfm_preds=deepfm_preds,
            xgb_weight=0.7,
            deepfm_weight=0.3
        )

        # 제출 파일 생성
        if test_ids is None:
            test_ids = range(len(ensemble_preds))

        submission = pd.DataFrame({
            'ID': test_ids,
            'clicked': ensemble_preds
        })

        timestamp = datetime.now().strftime('%m%d_%H%M')
        output_path = SUBMISSION_DIR / f'inference_output_{timestamp}.csv'
        submission.to_csv(output_path, index=False)

        print(f"Inference completed: {output_path}")

        return str(output_path)

    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
