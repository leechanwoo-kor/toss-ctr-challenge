"""
CatBoost 예측 스크립트
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from catboost import CatBoostClassifier, Pool

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.xgboost_feature_processor import XGBoostFeatureProcessor


def find_latest_catboost_model():
    """최신 CatBoost 모델 자동 탐색"""
    model_dir = "checkpoints/catboost"
    if not os.path.exists(model_dir):
        return None

    experiments = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not experiments:
        return None

    # 가장 최근 수정된 실험 디렉토리 찾기
    latest_exp = max(experiments, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    model_dir_path = f"{model_dir}/{latest_exp}"

    # K-fold 모델 파일들 확인
    model_files = [f for f in os.listdir(model_dir_path) if f.endswith('.cbm')]
    if model_files:
        return model_dir_path, latest_exp
    else:
        return None


def predict_with_catboost(model_dir_path=None, experiment_name=None):
    """CatBoost 모델로 예측"""

    print(f"🚀 Starting CatBoost prediction...")

    # 모델 경로 자동 탐색
    if model_dir_path is None:
        result = find_latest_catboost_model()
        if result is None:
            raise FileNotFoundError("No CatBoost model found. Run training first.")
        model_dir_path, experiment_name = result
        print(f"🔍 Auto-detected model: {model_dir_path}")
    else:
        print(f"📁 Using specified model: {model_dir_path}")

    # 테스트 데이터 로드
    print("📁 Loading test data...")
    test_df = pd.read_parquet("data/test.parquet", engine="pyarrow")
    print(f"Test data loaded: {test_df.shape}")

    # ID 백업 및 제거 (XGBoost와 동일한 방식)
    test_ids = test_df['ID'].copy() if 'ID' in test_df.columns else None
    if 'ID' in test_df.columns:
        test_df = test_df.drop(columns=['ID'])
        print("✅ Backed up and removed ID column")

    # 피처 프로세서 로드 (XGBoost와 동일한 경로)
    print("🔧 Loading feature processor...")
    processor_path = 'data/processed/xgboost/processor.pkl'
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)

    # 피처 처리
    print("🔄 Processing test features...")
    test_processed = processor.transform(test_df)
    print(f"Features processed: {test_processed.shape}")

    # 범주형 피처를 프로세서에서 자동으로 가져오기
    feature_names = list(test_processed.columns)
    categorical_feature_names = processor.categorical_features

    # 범주형 피처만 문자열로 변환 (CatBoost 요구사항)
    for name in categorical_feature_names:
        if name in test_processed.columns:
            test_processed[name] = test_processed[name].astype(str)

    print(f"Categorical features converted: {[name for name in categorical_feature_names if name in test_processed.columns]}")
    print(f"Test data shape after processing: {test_processed.shape}")
    print(f"Feature names: {feature_names[:10]}...")  # 첫 10개 피처명 확인

    # K-fold 모델들 로드 및 예측
    print("🔮 Loading K-fold models and making predictions...")

    model_files = [f for f in os.listdir(model_dir_path) if f.endswith('.cbm')]
    model_files.sort()  # fold 순서대로 정렬

    predictions = np.zeros(len(test_processed))

    for i, model_file in enumerate(model_files):
        print(f"  Loading {model_file}...")
        model_path = os.path.join(model_dir_path, model_file)

        # CatBoost 모델 로드
        model = CatBoostClassifier()
        model.load_model(model_path)

        # Pool 생성 (학습시와 동일한 방식)
        # 범주형 피처 인덱스 자동 감지
        categorical_indices = []
        for i, name in enumerate(feature_names):
            if name in categorical_feature_names:
                categorical_indices.append(i)

        test_pool = Pool(
            test_processed,
            cat_features=categorical_indices
        )

        # 예측
        fold_pred = model.predict_proba(test_pool)[:, 1]
        predictions += fold_pred

        print(f"  Fold {i+1} prediction completed")

    # 앙상블 평균
    predictions = predictions / len(model_files)
    print(f"✅ Ensemble prediction completed ({len(model_files)} models)")

    # 결과 생성
    print("📊 Creating submission...")
    submission = pd.DataFrame({
        'ID': test_ids if test_ids is not None else range(len(predictions)),
        'clicked': predictions
    })

    # 저장 (submissions 폴더 사용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"catboost_{experiment_name}_{timestamp}.csv"

    # submissions 폴더 경로 설정 (모델별 하위 폴더 사용)
    if os.path.exists('submissions/'):
        base_dir = 'submissions/catboost'
    else:
        base_dir = '../submissions/catboost'

    # catboost 하위 폴더 생성
    os.makedirs(base_dir, exist_ok=True)
    output_path = f'{base_dir}/{filename}'

    submission.to_csv(output_path, index=False)

    print(f"💾 Submission saved: {output_path}")
    print(f"📈 Prediction stats:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")

    # 메모 작성
    memo = f"""
=== CatBoost Submission ===
Experiment: {experiment_name}
Timestamp: {timestamp}
Local Val Score: 0.353146 (estimated)
Model: CatBoost K-Fold (5 folds)
Features: 78 (enhanced with EDA features)
Training: GPU-accelerated
Notes: Second algorithm for ensemble, solid baseline
"""

    # 메모도 catboost 하위 폴더에 저장
    memo_filename = f"memo_catboost_{experiment_name}_{timestamp}.txt"
    memo_path = f'{base_dir}/{memo_filename}'

    with open(memo_path, 'w', encoding='utf-8') as f:
        f.write(memo)

    print(f"📝 Memo saved: {memo_path}")
    print("🎯 Ready for submission!")

    return output_path, submission


def main():
    """메인 함수"""
    try:
        output_path, submission = predict_with_catboost()
        print(f"\n✅ CatBoost prediction completed successfully!")
        print(f"📁 Submission file: {output_path}")

    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()