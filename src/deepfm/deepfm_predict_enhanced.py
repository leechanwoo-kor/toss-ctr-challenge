"""
Enhanced DeepFM 예측 스크립트
훈련된 DeepFM 모델을 사용하여 테스트 데이터에 대한 예측 수행
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.deepfm_core import DeepFM, DeepFMDataset
from ..models.enhanced_deepfm_processor import EnhancedDeepFMFeatureProcessor
from ..utils.config import DEEPFM_CONFIG


def load_best_model_and_metadata():
    """최고 성능 모델과 메타데이터 로드"""
    print("📁 Loading best model and metadata...")

    # 최신 실험 디렉토리 찾기
    checkpoints_dir = "checkpoints/deepfm_enhanced"
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    # 가장 최신 실험 폴더 찾기
    experiment_dirs = [d for d in os.listdir(checkpoints_dir) if d.startswith("deepfm-enhanced-")]
    if not experiment_dirs:
        raise FileNotFoundError("No experiment directories found")

    latest_dir = sorted(experiment_dirs)[-1]
    model_dir = os.path.join(checkpoints_dir, latest_dir)

    print(f"   Using experiment: {latest_dir}")

    # 메타데이터 로드
    metadata_path = "data/processed/deepfm_enhanced/metadata.pkl"
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # 프로세서 로드
    processor_path = "data/processed/deepfm_enhanced/processor.pkl"
    with open(processor_path, "rb") as f:
        processor = pickle.load(f)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 최고 모델 가중치 로드
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found: {model_path}")

    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 체크포인트 구조 확인 및 모델 상태 추출
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        checkpoint_config = checkpoint.get('config', {})
        print(f"   Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        print(f"   Best score: {checkpoint.get('best_score', 'N/A')}")
        print(f"   Optimal temperature: {checkpoint.get('optimal_temperature', 'N/A')}")
    else:
        model_state_dict = checkpoint
        checkpoint_config = {}

    # 체크포인트에서 실제 사용된 설정으로 모델 생성
    actual_embedding_dim = checkpoint_config.get('EMBED_DIM', 9)  # 기본값 9 (훈련시 최적화됨)
    actual_hidden_dims = checkpoint_config.get('HIDDEN_DIMS', DEEPFM_CONFIG['HIDDEN_DIMS'])
    actual_dropout = checkpoint_config.get('DROPOUT', DEEPFM_CONFIG['DROPOUT'])

    print(f"   Using actual embedding dim: {actual_embedding_dim} (not {DEEPFM_CONFIG['EMBEDDING_DIM']})")

    # 올바른 설정으로 모델 재생성
    model = DeepFM(
        feature_dims=metadata['categorical_feature_dims'],
        embedding_dim=actual_embedding_dim,
        hidden_units=actual_hidden_dims,
        dropout=actual_dropout,
        num_numerical=len(metadata['numerical_features'])
    ).to(device)

    model.load_state_dict(model_state_dict)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Competition Score: {metadata.get('best_score', 'N/A')}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, processor, metadata, device


def prepare_test_data(processor):
    """테스트 데이터 준비"""
    print("📊 Preparing test data...")

    # 테스트 데이터 로드
    test_df = pd.read_parquet("data/test.parquet", engine="pyarrow")
    original_ids = test_df['ID'].copy()  # ID 저장

    print(f"Test data loaded: {test_df.shape}")

    # 동일한 전처리 적용 (fit 없이 transform만)
    print("🔧 Applying same preprocessing...")
    processed_test = processor.transform(test_df.copy())

    print(f"Test data processed: {processed_test.shape}")
    print(f"Features: {len(processed_test.columns)}")

    return processed_test, original_ids


def create_test_dataset(processed_test, processor, metadata):
    """테스트용 데이터셋 생성"""
    print("🧠 Creating test dataset...")

    # DeepFM Dataset 생성 (타겟 없음)
    test_dataset = DeepFMDataset(
        processed_test,
        processor,
        target_col=None,
        has_target=False
    )

    print(f"Test dataset created: {len(test_dataset)} samples")

    return test_dataset


def predict_with_model(model, test_dataset, device):
    """모델을 사용한 예측 수행 (배치 처리 없이 한 번에)"""
    print("🚀 Running predictions...")

    model.eval()

    with torch.no_grad():
        print("   📊 Loading all test data to GPU...")

        # 모든 데이터를 한 번에 처리
        categorical_data = {}
        for col, data in test_dataset.categorical_data.items():
            categorical_data[col] = data.to(device)

        numerical_data = test_dataset.numerical_data.to(device)

        print(f"   🔥 Processing {len(test_dataset):,} samples in one batch...")

        # 한 번에 모든 예측 수행
        outputs = model(categorical_data, numerical_data)
        probabilities = torch.sigmoid(outputs).cpu().numpy()

    print(f"✅ Predictions completed: {len(probabilities)} samples")
    print(f"   Prediction range: [{probabilities.min():.6f}, {probabilities.max():.6f}]")
    print(f"   Mean prediction: {probabilities.mean():.6f}")

    return probabilities


def save_predictions(predictions, original_ids, metadata):
    """예측 결과 저장"""
    print("💾 Saving predictions...")

    # 제출 파일 생성
    submission_df = pd.DataFrame({
        'ID': original_ids,
        'clicked': predictions
    })

    # 타임스탬프 생성
    timestamp = time.strftime("%m%d_%H%M")

    # 파일명 생성
    competition_score = metadata.get('best_score', 'unknown')
    if isinstance(competition_score, (int, float)):
        score_str = f"{competition_score:.6f}".replace('.', '')
    else:
        score_str = "unknown"

    filename = f"deepfm_enhanced_{timestamp}_{score_str}.csv"
    filepath = os.path.join("submissions", filename)

    # submissions 디렉토리 생성
    os.makedirs("submissions", exist_ok=True)

    # 저장
    submission_df.to_csv(filepath, index=False)

    print(f"✅ Predictions saved: {filepath}")
    print(f"   Submission shape: {submission_df.shape}")
    print(f"   Competition Score used: {competition_score}")

    # 통계 정보 출력
    print(f"\n📊 Prediction Statistics:")
    print(f"   Min: {predictions.min():.6f}")
    print(f"   Max: {predictions.max():.6f}")
    print(f"   Mean: {predictions.mean():.6f}")
    print(f"   Std: {predictions.std():.6f}")
    print(f"   Median: {np.median(predictions):.6f}")

    return filepath


def main():
    print("🚀 Starting Enhanced DeepFM prediction...")
    print("   🔧 Using best trained model")

    start_time = time.time()

    try:
        # 1. 모델과 메타데이터 로드
        model, processor, metadata, device = load_best_model_and_metadata()

        # 2. 테스트 데이터 준비
        processed_test, original_ids = prepare_test_data(processor)

        # 3. 테스트 데이터셋 생성
        test_dataset = create_test_dataset(processed_test, processor, metadata)

        # 4. 예측 수행 (배치 처리 없이 한 번에)
        predictions = predict_with_model(model, test_dataset, device)

        # 6. 예측 결과 저장
        filepath = save_predictions(predictions, original_ids, metadata)

        total_time = time.time() - start_time

        print(f"\n✅ Enhanced DeepFM prediction completed!")
        print(f"📊 Summary:")
        print(f"   - Total time: {total_time:.2f} seconds")
        print(f"   - Samples predicted: {len(predictions):,}")
        print(f"   - Output file: {filepath}")
        print(f"   - Model used: Best competition score {metadata.get('best_score', 'N/A')}")

        return filepath

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise


if __name__ == "__main__":
    filepath = main()