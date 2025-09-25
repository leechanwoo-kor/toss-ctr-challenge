"""
새로운 DeepFMProcessor를 사용한 DeepFM 예측
완전히 독립적이고 깔끔한 아키텍처
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.deepfm_optimized import OptimizedDeepFM
from src.deepfm.deepfm_processor import DeepFMProcessor


def find_latest_model():
    """최신 실험 모델 자동 탐색"""
    base_dir = "checkpoints/deepfm_new"
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {base_dir}")

    # 가장 최신 실험 폴더 찾기
    experiments = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not experiments:
        raise FileNotFoundError(f"No experiment folders found in {base_dir}")

    # 폴더명으로 정렬하여 최신 선택
    latest_exp = sorted(experiments)[-1]
    model_path = os.path.join(base_dir, latest_exp, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"🎯 Found latest model: {latest_exp}")
    print(f"📁 Model path: {model_path}")

    return model_path


def create_prediction_batches(df, processor, batch_size=10000):
    """예측용 배치 데이터 생성"""
    print(f"🔄 Creating prediction batches (batch size: {batch_size:,})...")

    # 프로세서로 데이터 변환
    print("   Applying processor transformation...")
    df_processed = processor.transform(df.copy())

    print(f"   Processed shape: {df_processed.shape}")

    # 배치별로 텐서 생성
    num_batches = (len(df_processed) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_processed))
        batch_df = df_processed.iloc[start_idx:end_idx]

        # 범주형 데이터
        categorical_data = {}
        for col in processor.categorical_features:
            if col in batch_df.columns:
                categorical_data[col] = torch.tensor(batch_df[col].values, dtype=torch.long)

        # 수치형 데이터
        numerical_data = None
        if processor.numerical_features:
            numerical_cols = [col for col in processor.numerical_features if col in batch_df.columns]
            if numerical_cols:
                numerical_data = torch.tensor(batch_df[numerical_cols].values, dtype=torch.float32)

        batches.append((categorical_data, numerical_data))

    print(f"   Created {len(batches)} batches")
    return batches


def predict_with_model(model, batches, device):
    """모델로 배치 예측"""
    print("🧠 Making predictions...")

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for i, (categorical_data, numerical_data) in enumerate(tqdm(batches, desc="Predicting")):
            # GPU로 이동
            for key in categorical_data:
                categorical_data[key] = categorical_data[key].to(device)

            if numerical_data is not None:
                numerical_data = numerical_data.to(device)

            # 예측
            outputs = model(categorical_data, numerical_data)
            predictions = torch.sigmoid(outputs.squeeze()).cpu().numpy()

            all_predictions.append(predictions)

    # 결합
    final_predictions = np.concatenate(all_predictions)
    print(f"✅ Predictions complete: {len(final_predictions):,} samples")

    return final_predictions


def main():
    print("🚀 DeepFM Prediction (New Clean Architecture)")
    print("Using independent DeepFMProcessor - No dependencies!")

    # 1. 최신 모델 탐색
    print("\n🔍 Finding latest trained model...")
    model_path = find_latest_model()

    # 2. 체크포인트 로드
    print("\n📂 Loading model checkpoint...")
    checkpoint = torch.load(model_path, weights_only=False)

    processor = checkpoint['processor']
    metadata = checkpoint['metadata']
    config = checkpoint.get('config', {})
    best_score = checkpoint.get('best_comp_score', 0.0)
    run_name = checkpoint.get('run_name', 'unknown')

    print(f"✅ Checkpoint loaded:")
    print(f"   Run: {run_name}")
    print(f"   Best score: {best_score:.6f}")
    print(f"   Architecture: {checkpoint.get('architecture', 'unknown')}")
    print(f"   Categorical features: {len(processor.categorical_features)}")
    print(f"   Numerical features: {len(processor.numerical_features)}")

    # 3. 모델 초기화
    print("\n🧠 Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = OptimizedDeepFM(
        feature_dims=processor.categorical_feature_dims,
        default_embedding_dim=config.get('EMBEDDING_DIM', 16),
        hidden_units=config.get('HIDDEN_UNITS', [256, 128]),
        dropout=config.get('DROPOUT', 0.2),
        num_numerical=len(processor.numerical_features)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model loaded on {device}")
    print(f"   Total parameters: {total_params:,}")

    # 4. 테스트 데이터 로드
    print("\n📊 Loading test data...")
    test = pd.read_parquet("data/test.parquet", engine="pyarrow")
    original_ids = None

    # ID 컬럼 처리
    if 'ID' in test.columns:
        original_ids = test['ID'].copy()
        test = test.drop(columns=['ID'])

    print(f"   Test data shape: {test.shape}")

    # 5. 예측 배치 생성
    batch_size = config.get('PREDICTION_BATCH_SIZE', 10000)
    batches = create_prediction_batches(test, processor, batch_size)

    # 6. 예측 실행
    predictions = predict_with_model(model, batches, device)

    # 7. 결과 저장
    print("\n💾 Saving predictions...")

    # 결과 DataFrame 생성
    if original_ids is not None:
        result_df = pd.DataFrame({
            'ID': original_ids,
            'clicked': predictions
        })
    else:
        result_df = pd.DataFrame({
            'clicked': predictions
        })

    # 저장 경로
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"deepfm_new_predictions_{timestamp}.csv"

    # 결과 통계
    print(f"📊 Prediction Statistics:")
    print(f"   Samples: {len(predictions):,}")
    print(f"   Mean: {predictions.mean():.6f}")
    print(f"   Std: {predictions.std():.6f}")
    print(f"   Min: {predictions.min():.6f}")
    print(f"   Max: {predictions.max():.6f}")
    print(f"   Predicted CTR: {predictions.mean():.4f}%")

    # 저장
    result_df.to_csv(filename, index=False)
    print(f"✅ Predictions saved to: {filename}")

    # 8. 최종 요약
    print("\n" + "="*60)
    print("🎉 DeepFM Prediction Complete!")
    print("="*60)
    print(f"🎯 Model: {run_name}")
    print(f"📈 Training Score: {best_score:.6f}")
    print(f"🔮 Predictions: {len(predictions):,} samples")
    print(f"📁 Output: {filename}")
    print("✅ Clean Architecture - No Dependencies!")

    return predictions, filename


if __name__ == "__main__":
    try:
        predictions, filename = main()
        print(f"\n🎯 Prediction completed successfully!")
        print(f"📁 Results saved to: {filename}")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()