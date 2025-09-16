"""
DeepFM 예측 스크립트 (모듈화 버전)
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.deepfm_core import DeepFM
from src.models.feature_processor import InferenceFeatureProcessor


def predict_with_modular_model(model_path='deepfm_modular_best_model.pth'):
    """모듈화된 모델로 예측"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔮 Starting prediction with device: {device}")

    # 저장된 체크포인트 로드
    print("📂 Loading saved model...")
    checkpoint = torch.load(model_path, weights_only=False)

    # 프로세서 복원
    processor = checkpoint['processor']
    config = checkpoint.get('config', {})

    print(f"✅ Processor loaded: {len(processor.categorical_features)} categorical, {len(processor.numerical_features)} numerical")
    print(f"📊 Model config: {config}")

    # 테스트 데이터 로드
    print("📊 Loading test data...")
    test = pd.read_parquet("data/test.parquet", engine="pyarrow")
    test_ids = test['ID'].copy() if 'ID' in test.columns else None

    if 'ID' in test.columns:
        test = test.drop(columns=['ID'])
    print(f"Test data shape: {test.shape}")

    # 테스트 데이터 전처리
    print("🔧 Processing test data...")

    # InferenceFeatureProcessor 생성 및 설정 복사
    inference_processor = InferenceFeatureProcessor()
    inference_processor.categorical_encoders = processor.categorical_encoders
    inference_processor.categorical_features = processor.categorical_features
    inference_processor.numerical_features = processor.numerical_features
    inference_processor.feature_dims = processor.feature_dims
    inference_processor.scaler = processor.scaler
    inference_processor.numerical_stats = processor.numerical_stats

    # 전처리 적용
    test_processed = inference_processor.transform(test.copy())

    print(f"✅ Test data processed: {test_processed.shape}")

    # 모델 초기화 및 가중치 로드
    print("🧠 Initializing model...")
    model = DeepFM(
        feature_dims=processor.feature_dims,
        embedding_dim=config.get('EMBEDDING_DIM', 32),
        hidden_units=config.get('HIDDEN_UNITS', [768, 512, 256]),
        dropout=config.get('DROPOUT', 0.3),
        num_numerical=len(processor.numerical_features)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded with best score: {checkpoint['score']:.6f}")

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

    # Submission 생성
    print("📝 Creating submission...")
    if os.path.exists('submissions/sample_submission.csv'):
        submit_path = 'submissions/sample_submission.csv'
        output_path = 'submissions/deepfm_modular_submission.csv'
    else:
        submit_path = '../submissions/sample_submission.csv'
        output_path = '../submissions/deepfm_modular_submission.csv'

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