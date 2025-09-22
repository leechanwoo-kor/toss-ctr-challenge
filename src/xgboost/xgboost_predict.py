"""
XGBoost 예측 스크립트
"""

import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import xgboost as xgb

from ..models.feature_processor import InferenceFeatureProcessor


def find_latest_model():
    """최신 XGBoost 모델 자동 탐색"""
    model_dir = "checkpoints/xgboost"
    if not os.path.exists(model_dir):
        return None

    experiments = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not experiments:
        return None

    # 가장 최근 수정된 실험 디렉토리 찾기
    latest_exp = max(experiments, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    model_path = f"{model_dir}/{latest_exp}/xgboost_model.json"

    if os.path.exists(model_path):
        return model_path, f"{model_dir}/{latest_exp}/model_metadata.pkl"
    else:
        return None


def predict_with_xgboost(model_path=None, metadata_path=None):
    """XGBoost 모델로 예측"""

    print(f"🚀 Starting XGBoost prediction...")

    # 모델 경로 자동 탐색
    if model_path is None:
        result = find_latest_model()
        if result is None:
            raise FileNotFoundError("No XGBoost model found. Run training first.")
        model_path, metadata_path = result
        print(f"🔍 Auto-detected model: {model_path}")
    else:
        print(f"📁 Using specified model: {model_path}")

    # 모델 로드
    print("📂 Loading saved model...")
    model = xgb.Booster()
    model.load_model(model_path)

    # 메타데이터 로드
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    processor = metadata['processor']
    feature_names = metadata['feature_names']
    val_metrics = metadata['val_metrics']

    print(f"✅ Model loaded: {model.best_iteration} iterations")
    print(f"📊 Model performance:")
    print(f"  Competition Score: {val_metrics['comp_score']:.6f}")
    print(f"  AUC: {val_metrics['auc']:.6f}")
    print(f"  Best iteration: {metadata['best_iteration']}")

    # 테스트 데이터 로드
    print("📊 Loading test data...")
    test = pd.read_parquet("data/test.parquet", engine="pyarrow")
    test_ids = test['ID'].copy() if 'ID' in test.columns else None

    if 'ID' in test.columns:
        test = test.drop(columns=['ID'])
    print(f"Test data shape: {test.shape}")

    # 테스트 데이터 전처리
    print("🔧 Processing test data...")

    # Processor 생성/복사
    if isinstance(processor, InferenceFeatureProcessor):
        inference_processor = processor
        print("   Using existing InferenceFeatureProcessor")
    else:
        inference_processor = InferenceFeatureProcessor()

        # 핵심 속성들만 복사
        essential_attrs = [
            'categorical_encoders', 'categorical_features', 'numerical_features',
            'feature_dims', 'scaler', 'numerical_stats'
        ]

        for attr in essential_attrs:
            if hasattr(processor, attr):
                setattr(inference_processor, attr, getattr(processor, attr))

        print("   Converted TrainingFeatureProcessor to InferenceFeatureProcessor")

    # 전처리 적용
    test_processed = inference_processor.transform(test.copy())
    print(f"✅ Test data processed: {test_processed.shape}")

    # 피처 순서 맞추기
    test_processed = test_processed[feature_names]

    # 메모리 최적화
    for col in test_processed.columns:
        if test_processed[col].dtype == 'int64':
            test_processed[col] = test_processed[col].astype('int32')
        elif test_processed[col].dtype == 'float64':
            test_processed[col] = test_processed[col].astype('float32')

    # XGBoost DMatrix 생성 (GPU 최적화)
    print("🔄 Creating DMatrix for prediction...")
    dtest = xgb.DMatrix(
        test_processed.values,
        feature_names=feature_names
    )

    # GPU 사용 여부 확인 및 설정
    config = metadata.get('config', {})
    use_gpu = ('device' in config and 'cuda' in config.get('device', '')) or \
              ('tree_method' in config and 'gpu' in config.get('tree_method', ''))

    if use_gpu:
        print("🚀 GPU acceleration enabled for prediction")
        # GPU 예측을 위한 설정
        try:
            # GPU predictor 사용 시도
            predictions = model.predict(
                dtest,
                iteration_range=(0, metadata['best_iteration']),
                pred_leaf=False,
                pred_contribs=False,
                approx_contribs=False,
                pred_interactions=False
            )
        except Exception as e:
            print(f"⚠️  GPU prediction failed, falling back to CPU: {e}")
            predictions = model.predict(
                dtest,
                iteration_range=(0, metadata['best_iteration'])
            )
    else:
        print("💻 Using CPU for prediction")
        predictions = model.predict(
            dtest,
            iteration_range=(0, metadata['best_iteration'])
        )

    print(f"✅ Predictions generated: {len(predictions):,}")

    # 예측 결과 검증
    print("🔍 Validating predictions...")

    # 1. 범위 검증 (0-1 사이)
    if np.any((predictions < 0) | (predictions > 1)):
        invalid_count = np.sum((predictions < 0) | (predictions > 1))
        print(f"⚠️  Warning: {invalid_count} predictions outside [0,1] range")
        predictions = np.clip(predictions, 0, 1)
        print("   Clipped to valid range")

    # 2. NaN/Inf 검증
    if np.any(~np.isfinite(predictions)):
        nan_count = np.sum(~np.isfinite(predictions))
        print(f"⚠️  Warning: {nan_count} NaN/Inf predictions found")
        predictions = np.nan_to_num(predictions, nan=0.5, posinf=1.0, neginf=0.0)
        print("   Replaced with safe values")

    # 3. 예측 분포 검증
    mean_pred = predictions.mean()
    if mean_pred < 0.001 or mean_pred > 0.999:
        print(f"⚠️  Warning: Unusual prediction mean {mean_pred:.6f}")

    print("✅ Validation complete")

    # Submission 생성
    print("📝 Creating submission...")

    # 실험별 고유 파일명 생성
    timestamp = datetime.now().strftime('%m%d_%H%M')
    if 'models/xgboost' in model_path:
        exp_name = os.path.basename(os.path.dirname(model_path))
        file_suffix = f"{exp_name}_{timestamp}"
    else:
        file_suffix = f"manual_{timestamp}"

    if os.path.exists('submissions/sample_submission.csv'):
        submit_path = 'submissions/sample_submission.csv'
        base_dir = 'submissions/xgboost'
    else:
        submit_path = '../submissions/sample_submission.csv'
        base_dir = '../submissions/xgboost'

    # xgboost 하위 폴더 생성
    os.makedirs(base_dir, exist_ok=True)
    output_path = f'{base_dir}/xgboost_{file_suffix}.csv'

    submit = pd.read_csv(submit_path)
    submit['clicked'] = predictions
    submit.to_csv(output_path, index=False)

    print(f"\n🎯 Submission saved: {output_path}")
    print(f"📊 Prediction stats:")
    print(f"  Count: {len(predictions):,}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")

    return predictions


if __name__ == "__main__":
    import sys

    # 명령행 인자로 모델 경로 지정 가능
    model_path = None
    metadata_path = None

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        # 메타데이터 경로 추정
        model_dir = os.path.dirname(model_path)
        metadata_path = os.path.join(model_dir, "model_metadata.pkl")

        if not os.path.exists(metadata_path):
            # 다른 가능한 메타데이터 파일명들 시도
            possible_names = ["metadata.pkl", "model_info.pkl"]
            for name in possible_names:
                test_path = os.path.join(model_dir, name)
                if os.path.exists(test_path):
                    metadata_path = test_path
                    break

    try:
        predictions = predict_with_xgboost(model_path, metadata_path)
        print("\\n🏆 Prediction completed successfully!")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()