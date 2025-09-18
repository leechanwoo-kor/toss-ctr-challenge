"""
XGBoost K-Fold 앙상블 예측 스크립트
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_kfold_models(kfold_dir):
    """K-fold 모델들 로드"""
    models = []
    model_files = [f for f in os.listdir(kfold_dir) if f.startswith('xgboost_fold_') and f.endswith('.json')]
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
    print("📊 Loading test data...")

    # 원본 테스트 데이터 로드
    test = pd.read_parquet("data/test.parquet")
    test_ids = test['ID'].copy() if 'ID' in test.columns else None

    if 'ID' in test.columns:
        test = test.drop(columns=['ID'])

    print(f"Test data shape: {test.shape}")

    # 전처리된 데이터 로드 (XGBoost 학습에 사용된 것과 동일)
    data_dir = 'data/processed/xgboost_enhanced'

    # 프로세서 로드
    with open(f'{data_dir}/processor.pkl', 'rb') as f:
        processor = pickle.load(f)

    # 피처명 로드
    with open(f'{data_dir}/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    print("🔧 Processing test data...")

    # TreeBasedFeatureProcessor의 transform_test 메서드 사용
    if hasattr(processor, 'transform_test'):
        test_processed = processor.transform_test(test)
    else:
        # 일반 transform 사용
        test_processed = processor.transform(test)

    # 피처 순서 맞추기
    test_processed = test_processed[feature_names]

    print(f"✅ Test data processed: {test_processed.shape}")

    return test_processed, test_ids, feature_names


def predict_kfold_ensemble():
    """K-fold 앙상블 예측"""
    print("🚀 Starting K-Fold ensemble prediction...")

    # 최신 K-fold 모델 디렉토리 찾기
    kfold_base_dir = "checkpoints/xgboost"
    kfold_dirs = [d for d in os.listdir(kfold_base_dir) if 'kfold' in d]

    if not kfold_dirs:
        raise FileNotFoundError("No K-fold models found. Run K-fold training first.")

    # 최신 K-fold 디렉토리
    latest_kfold = max(kfold_dirs, key=lambda x: os.path.getctime(os.path.join(kfold_base_dir, x)))
    kfold_dir = os.path.join(kfold_base_dir, latest_kfold)

    print(f"🔍 Using K-fold models from: {kfold_dir}")

    # 모델들 로드
    models = load_kfold_models(kfold_dir)

    # 테스트 데이터 로드
    test_processed, test_ids, feature_names = load_test_data()

    # 예측
    print("🔮 Generating predictions...")
    predictions = np.zeros(len(test_processed))

    for i, model in enumerate(models):
        print(f"  Predicting with fold {i+1}/{len(models)}...")

        # DMatrix 생성
        dtest = xgb.DMatrix(test_processed.values, feature_names=feature_names)

        # 예측
        fold_pred = model.predict(dtest)
        predictions += fold_pred / len(models)

    print(f"✅ Ensemble predictions generated: {len(predictions):,}")

    # 예측 결과 검증
    print("🔍 Validating predictions...")
    predictions = np.clip(predictions, 0, 1)
    predictions = np.nan_to_num(predictions, nan=0.5)

    # Submission 생성
    print("📝 Creating submission...")

    # submission 디렉토리 확인
    if os.path.exists('submissions'):
        submit_dir = 'submissions'
    else:
        os.makedirs('submissions', exist_ok=True)
        submit_dir = 'submissions'

    # 샘플 submission 로드
    sample_paths = [
        'submissions/sample_submission.csv',
        '../submissions/sample_submission.csv',
        'data/sample_submission.csv'
    ]

    sample_path = None
    for path in sample_paths:
        if os.path.exists(path):
            sample_path = path
            break

    if sample_path:
        submit = pd.read_csv(sample_path)
        submit['clicked'] = predictions
    else:
        # 샘플이 없으면 직접 생성
        if test_ids is not None:
            submit = pd.DataFrame({
                'ID': test_ids,
                'clicked': predictions
            })
        else:
            submit = pd.DataFrame({
                'ID': range(len(predictions)),
                'clicked': predictions
            })

    # 파일명 생성
    timestamp = datetime.now().strftime('%m%d_%H%M')
    output_path = f'{submit_dir}/xgboost_kfold_{timestamp}.csv'

    submit.to_csv(output_path, index=False)

    print(f"\n🎯 Submission saved: {output_path}")
    print(f"📊 Prediction stats:")
    print(f"  Count: {len(predictions):,}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")

    return predictions, output_path


if __name__ == "__main__":
    try:
        predictions, output_path = predict_kfold_ensemble()
        print("\n🏆 K-Fold ensemble prediction completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()