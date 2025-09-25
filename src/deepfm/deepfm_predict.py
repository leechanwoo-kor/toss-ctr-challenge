"""
DeepFM 예측 스크립트 (모듈화 버전)
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.deepfm_optimized import OptimizedDeepFM


def find_latest_model():
    """최신 실험 모델 자동 탐색"""
    model_dir = "checkpoints/deepfm"
    if not os.path.exists(model_dir):
        return "checkpoints/deepfm_modular_best_model.pth"  # fallback

    experiments = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not experiments:
        return "checkpoints/deepfm_modular_best_model.pth"

    # 가장 최근 수정된 실험 디렉토리 찾기
    latest_exp = max(experiments, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    model_path = f"{model_dir}/{latest_exp}/best_model.pth"

    if os.path.exists(model_path):
        return model_path
    else:
        return "checkpoints/deepfm_modular_best_model.pth"  # fallback


def predict_with_modular_model(model_path=None):
    """모듈화된 모델로 예측"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔮 Starting prediction with device: {device}")

    # 모델 경로 자동 탐색
    if model_path is None:
        model_path = find_latest_model()
        print(f"🔍 Auto-detected model: {model_path}")
    else:
        print(f"📁 Using specified model: {model_path}")

    # 저장된 체크포인트 로드
    print("📂 Loading saved model...")
    checkpoint = torch.load(model_path, weights_only=False)

    # 프로세서 복원
    processor = checkpoint['processor']
    config = checkpoint.get('config', {})
    metadata = checkpoint.get('metadata', {})

    # 버전 호환성 처리 (점수 표시)
    if 'best_comp_score' in checkpoint:
        score = checkpoint['best_comp_score']
        score_type = "Competition Score"
    elif 'best_auc' in checkpoint:
        score = checkpoint['best_auc']
        score_type = "AUC"
    elif 'score' in checkpoint:
        score = checkpoint['score']
        score_type = "Score"
    else:
        score = 0.0
        score_type = "Unknown"

    print(f"✅ Processor loaded: {len(processor.categorical_features)} categorical, {len(processor.numerical_features)} numerical")
    print(f"📊 Model config: {config}")
    print(f"🎯 Best {score_type}: {score:.6f}")
    print(f"📋 Feature dims: {len(processor.categorical_feature_dims)} categorical features")

    # 테스트 데이터 로드
    print("📊 Loading test data...")
    test = pd.read_parquet("data/test.parquet", engine="pyarrow")
    test_ids = test['ID'].copy() if 'ID' in test.columns else None

    if 'ID' in test.columns:
        test = test.drop(columns=['ID'])
    print(f"Test data shape: {test.shape}")

    # 테스트 데이터 전처리
    print("🔧 Processing test data...")

    # ExactDeepFMProcessor 직접 사용
    test_processed = processor.transform(test.copy())

    print(f"✅ Test data processed: {test_processed.shape}")

    # 모델 초기화 및 가중치 로드
    print("🧠 Initializing model...")
    model = OptimizedDeepFM(
        feature_dims=processor.categorical_feature_dims,
        default_embedding_dim=config.get('EMBEDDING_DIM', 16),
        hidden_units=config.get('HIDDEN_UNITS', [256, 128]),
        dropout=config.get('DROPOUT', 0.2),
        num_numerical=len(processor.numerical_features)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded successfully")

    # 배치 예측
    print("🔮 Generating predictions...")
    batch_size = 1024
    predictions = []

    with torch.no_grad():
        for start_idx in tqdm(range(0, len(test_processed), batch_size), desc="Predicting"):
            end_idx = min(start_idx + batch_size, len(test_processed))
            batch = test_processed.iloc[start_idx:end_idx]

            # 범주형 입력
            categorical_inputs = {}
            for col in processor.categorical_features:
                categorical_inputs[col] = torch.LongTensor(batch[col].values).to(device)

            # 수치형 입력
            numerical_inputs = torch.FloatTensor(batch[processor.numerical_features].values).to(device)

            # 예측
            outputs = model(categorical_inputs, numerical_inputs)
            preds = torch.sigmoid(outputs)
            predictions.extend(preds.cpu().numpy())

    predictions = np.array(predictions)
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

    # Submission 생성 (실험별 파일명)
    print("📝 Creating submission...")

    # 실험별 고유 파일명 생성
    timestamp = datetime.now().strftime('%m%d_%H%M')
    if 'checkpoints/deepfm' in model_path:
        exp_name = os.path.basename(os.path.dirname(model_path))
        file_suffix = f"{exp_name}_{timestamp}"
    else:
        file_suffix = f"manual_{timestamp}"

    if os.path.exists('submissions/sample_submission.csv'):
        submit_path = 'submissions/sample_submission.csv'
        base_dir = 'submissions/deepfm'
    else:
        submit_path = '../submissions/sample_submission.csv'
        base_dir = '../submissions/deepfm'

    # deepfm 하위 폴더 생성
    os.makedirs(base_dir, exist_ok=True)
    output_path = f'{base_dir}/deepfm_{file_suffix}.csv'

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
    try:
        predictions = predict_with_modular_model()
        print("\n🏆 Prediction completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()