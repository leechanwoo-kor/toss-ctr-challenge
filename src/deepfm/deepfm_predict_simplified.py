"""
간소화된 DeepFM 예측 스크립트
훈련된 모델로 테스트 데이터에 대한 예측 수행
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

from ..models.deepfm_core import DeepFM, DeepFMDataset
from ..utils.config import DEEPFM_CONFIG


def load_trained_model(checkpoint_path):
    """훈련된 모델 로드"""
    print(f"📁 Loading trained model from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint['metadata']

    # 모델 초기화
    model = DeepFM(
        feature_dims=metadata['categorical_feature_dims'],
        embedding_dim=DEEPFM_CONFIG['EMBEDDING_DIM'],
        hidden_units=DEEPFM_CONFIG['HIDDEN_DIMS'],
        dropout=DEEPFM_CONFIG['DROPOUT'],
        num_numerical=len(metadata['numerical_features'])
    ).to(device)

    # 모델 상태 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Best score: {checkpoint['best_score']:.6f}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Device: {device}")

    return model, metadata, device


def load_processor():
    """전처리 프로세서 로드"""
    processor_path = 'data/processed/deepfm_simplified/processor.pkl'

    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"Processor not found: {processor_path}. Run data preparation first.")

    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)

    print("✅ Processor loaded successfully")
    return processor


def prepare_test_data(test_df, processor):
    """테스트 데이터 전처리"""
    print(f"🔧 Processing test data: {test_df.shape}")

    # 동일한 전처리 적용
    processed_test = processor.transform(test_df.copy())

    # DeepFM Dataset 생성 (target 없음)
    test_dataset = DeepFMDataset(
        processed_test, processor, target_col=None, has_target=False
    )

    print(f"✅ Test dataset prepared: {len(test_dataset)} samples")
    return test_dataset


def predict_batch(model, test_loader, device):
    """배치 단위 예측"""
    print("🔮 Generating predictions...")

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # 예측 시에는 target이 없으므로 2개 요소만 반환
            if len(batch) == 2:
                categorical_features, numerical_features = batch
            else:
                # 혹시 3개가 반환된다면 (target 포함)
                categorical_features, numerical_features, _ = batch

            categorical_features = {k: v.to(device) for k, v in categorical_features.items()}
            numerical_features = numerical_features.to(device)

            # 예측 (로짓)
            outputs = model(categorical_features, numerical_features).squeeze()

            # 확률로 변환
            probabilities = torch.sigmoid(outputs)
            all_predictions.append(probabilities.cpu().numpy())

    # 모든 배치 결합
    predictions = np.concatenate(all_predictions)

    print(f"✅ Predictions generated: {len(predictions)} samples")
    print(f"   Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"   Mean prediction: {predictions.mean():.6f}")

    return predictions


def main(test_data_path=None, checkpoint_path=None, output_path=None):
    """메인 예측 함수"""
    print("🚀 Starting Simplified DeepFM prediction...")
    start_time = time.time()

    # 1. 기본 경로 설정
    if test_data_path is None:
        test_data_path = "data/test.parquet"

    if checkpoint_path is None:
        # 최신 체크포인트 찾기
        checkpoint_base = "checkpoints/deepfm_simplified"
        if os.path.exists(checkpoint_base):
            # 가장 최근 모델 폴더 찾기
            model_dirs = [d for d in os.listdir(checkpoint_base)
                         if os.path.isdir(os.path.join(checkpoint_base, d))]
            if model_dirs:
                latest_dir = sorted(model_dirs)[-1]
                checkpoint_path = os.path.join(checkpoint_base, latest_dir, "best_model.pth")
            else:
                raise FileNotFoundError("No trained model found. Run training first.")
        else:
            raise FileNotFoundError("No checkpoint directory found. Run training first.")

    if output_path is None:
        timestamp = datetime.now().strftime('%m%d_%H%M')
        output_path = f"submissions/deepfm/deepfm_simplified_predictions_{timestamp}.csv"

    print(f"📊 Configuration:")
    print(f"   Test data: {test_data_path}")
    print(f"   Model: {checkpoint_path}")
    print(f"   Output: {output_path}")

    # 2. 테스트 데이터 로드
    print("\n📁 Loading test data...")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    test_df = pd.read_parquet(test_data_path, engine="pyarrow")
    print(f"Test data loaded: {test_df.shape}")

    # 3. 훈련된 모델 로드
    print("\n🧠 Loading trained model...")
    model, metadata, device = load_trained_model(checkpoint_path)

    # 4. 전처리 프로세서 로드
    print("\n⚙️ Loading preprocessor...")
    processor = load_processor()

    # 5. 테스트 데이터 전처리
    print("\n🔧 Preprocessing test data...")
    test_dataset = prepare_test_data(test_df, processor)

    # ID 정보 저장 (결과 생성용)
    test_ids = test_df['ID'].values if 'ID' in test_df.columns else None
    test_samples_count = len(test_df)

    # 메모리 정리
    del test_df
    gc.collect()

    # 6. 데이터 로더 생성 (예측용 최적화)
    # 예측 시에는 더 큰 배치 사이즈 사용 가능 (gradient 계산 없음)
    predict_batch_size = min(DEEPFM_CONFIG['BATCH_SIZE'] * 2, 8192)
    test_loader = DataLoader(
        test_dataset,
        batch_size=predict_batch_size,
        shuffle=False,  # 예측 시에는 순서 유지 필요
        num_workers=2,
        pin_memory=True
    )
    print(f"Using prediction batch size: {predict_batch_size}")

    # 7. 예측 수행
    print("\n🔮 Making predictions...")
    predictions = predict_batch(model, test_loader, device)

    # 8. 결과 저장
    print("\n💾 Saving predictions...")

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 예측 결과 DataFrame 생성
    if test_ids is not None:
        result_df = pd.DataFrame({
            'ID': test_ids,
            'clicked': predictions
        })
    else:
        result_df = pd.DataFrame({
            'clicked': predictions
        })

    # CSV 저장
    result_df.to_csv(output_path, index=False)

    total_time = time.time() - start_time

    print("✅ Prediction completed!")
    print(f"📊 Summary:")
    print(f"   - Total time: {total_time:.2f} seconds")
    print(f"   - Test samples: {test_samples_count:,}")
    print(f"   - Predictions saved: {output_path}")
    print(f"   - Prediction stats:")
    print(f"     • Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"     • Mean: {predictions.mean():.6f}")
    print(f"     • Std: {predictions.std():.6f}")

    # 예측 분포 출력
    print(f"\n📈 Prediction distribution:")
    for threshold in [0.01, 0.05, 0.1, 0.2, 0.5]:
        count = (predictions > threshold).sum()
        percentage = count / len(predictions) * 100
        print(f"   > {threshold:4.2f}: {count:6,} ({percentage:5.1f}%)")

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DeepFM Simplified Prediction')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data (default: data/test.parquet)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: latest in checkpoints/)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for predictions (default: submissions/deepfm/deepfm_simplified_predictions_{timestamp}.csv)')

    args = parser.parse_args()

    result_df = main(
        test_data_path=args.test_data,
        checkpoint_path=args.checkpoint,
        output_path=args.output
    )