"""
DeepFM K-Fold Ensemble Prediction
5개 폴드 모델의 평균을 사용한 앙상블 예측
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.deepfm_optimized import OptimizedDeepFM
from src.models.deepfm_dynamic_embedding import DynamicEmbeddingDeepFM
from src.deepfm.deepfm_processor import DeepFMProcessor


def find_latest_kfold_experiment():
    """최신 K-Fold 실험 폴더 자동 탐색 (수정 시간 기준)"""
    base_dir = "checkpoints/deepfm_kfold"
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"K-Fold checkpoint directory not found: {base_dir}")

    # 가장 최신 실험 폴더 찾기
    experiments = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not experiments:
        raise FileNotFoundError(f"No K-Fold experiment folders found in {base_dir}")

    # 수정 시간 기준으로 정렬하여 최신 선택
    exp_paths = [(d, os.path.getmtime(os.path.join(base_dir, d))) for d in experiments]
    exp_paths.sort(key=lambda x: x[1], reverse=True)  # 최신 것이 먼저
    latest_exp = exp_paths[0][0]
    exp_path = os.path.join(base_dir, latest_exp)

    print(f"🎯 Found latest K-Fold experiment: {latest_exp}")
    print(f"📁 Experiment path: {exp_path}")

    return exp_path, latest_exp


def load_kfold_models(exp_path):
    """K-Fold 모델들 로드"""
    print("🔄 Loading K-Fold models...")

    models = []
    processors = []
    configs = []

    # 5개 폴드 모델 로드
    for fold in range(1, 6):
        model_path = os.path.join(exp_path, f"fold_{fold}_model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fold {fold} model not found: {model_path}")

        print(f"   Loading fold {fold} model...")
        checkpoint = torch.load(model_path, weights_only=False)

        processor = checkpoint['processor']
        metadata = checkpoint['metadata']
        config = checkpoint.get('config', {})

        # Dynamic Embedding 모델인지 확인
        use_dynamic = config.get('USE_DYNAMIC_EMBEDDING', False)

        if use_dynamic:
            # Dynamic Embedding 모델 초기화
            model = DynamicEmbeddingDeepFM(
                feature_dims=processor.categorical_feature_dims,
                hidden_units=config.get('HIDDEN_DIMS', [256, 128]),
                dropout=config.get('DROPOUT', 0.25),
                num_numerical=len(processor.numerical_features),
                min_embed_dim=config.get('MIN_EMBED_DIM', 4),
                max_embed_dim=config.get('MAX_EMBED_DIM', 50),
                use_dynamic_embedding=True
            )
            if fold == 1:
                print(f"   📊 Using DynamicEmbeddingDeepFM")
        else:
            # OptimizedDeepFM 모델 초기화
            model = OptimizedDeepFM(
                feature_dims=processor.categorical_feature_dims,
                default_embedding_dim=config.get('EMBEDDING_DIM', 16),
                hidden_units=config.get('HIDDEN_DIMS', [256, 128]),
                dropout=config.get('DROPOUT', 0.2),
                num_numerical=len(processor.numerical_features)
            )
            if fold == 1:
                print(f"   📊 Using OptimizedDeepFM")

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        models.append(model)
        processors.append(processor)
        configs.append(config)

    print(f"✅ Loaded {len(models)} K-Fold models")

    # 첫 번째 프로세서를 대표로 사용 (모두 동일해야 함)
    return models, processors[0], configs[0]


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


def predict_with_kfold_ensemble(models, batches, device):
    """K-Fold 앙상블로 배치 예측"""
    print(f"🧠 Making ensemble predictions with {len(models)} models...")

    # 모든 모델을 GPU로 이동
    for model in models:
        model.to(device)
        model.eval()

    all_predictions = []

    with torch.no_grad():
        for i, (categorical_data, numerical_data) in enumerate(tqdm(batches, desc="Ensemble Predicting")):
            batch_predictions = []

            # 각 폴드 모델로 예측
            for fold_idx, model in enumerate(models):
                # GPU로 이동
                batch_categorical = {}
                for key, data in categorical_data.items():
                    batch_categorical[key] = data.to(device)

                batch_numerical = None
                if numerical_data is not None:
                    batch_numerical = numerical_data.to(device)

                # 예측
                outputs = model(batch_categorical, batch_numerical)
                predictions = torch.sigmoid(outputs.squeeze()).cpu().numpy()

                batch_predictions.append(predictions)

            # 5개 폴드 평균
            ensemble_predictions = np.mean(batch_predictions, axis=0)
            all_predictions.append(ensemble_predictions)

    # 결합
    final_predictions = np.concatenate(all_predictions)
    print(f"✅ Ensemble predictions complete: {len(final_predictions):,} samples")

    return final_predictions


def main():
    print("🚀 DeepFM K-Fold Ensemble Prediction")
    print("Using 5-Fold Cross Validation Models")

    # 1. 최신 K-Fold 실험 탐색
    print("\n🔍 Finding latest K-Fold experiment...")
    exp_path, exp_name = find_latest_kfold_experiment()

    # 2. K-Fold 모델들 로드
    print("\n📂 Loading K-Fold models...")
    models, processor, config = load_kfold_models(exp_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    print(f"   Models loaded: {len(models)}")
    print(f"   Categorical features: {len(processor.categorical_features)}")
    print(f"   Numerical features: {len(processor.numerical_features)}")

    # 3. 테스트 데이터 로드
    print("\n📊 Loading test data...")
    test = pd.read_parquet("data/test.parquet", engine="pyarrow")
    original_ids = None

    # ID 컬럼 처리
    if 'ID' in test.columns:
        original_ids = test['ID'].copy()
        test = test.drop(columns=['ID'])

    print(f"   Test data shape: {test.shape}")

    # 4. 예측 배치 생성
    batch_size = config.get('PREDICTION_BATCH_SIZE', 10000)
    batches = create_prediction_batches(test, processor, batch_size)

    # 5. K-Fold 앙상블 예측 실행
    predictions = predict_with_kfold_ensemble(models, batches, device)

    # 6. 결과 저장
    print("\n💾 Saving ensemble predictions...")

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
    filename = f"submissions/deepfm/deepfm_kfold_ensemble_{timestamp}.csv"

    # 디렉토리 생성
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 결과 통계
    print(f"📊 Ensemble Prediction Statistics:")
    print(f"   Samples: {len(predictions):,}")
    print(f"   Mean: {predictions.mean():.6f}")
    print(f"   Std: {predictions.std():.6f}")
    print(f"   Min: {predictions.min():.6f}")
    print(f"   Max: {predictions.max():.6f}")
    print(f"   Predicted CTR: {predictions.mean():.4f}%")

    # 저장
    result_df.to_csv(filename, index=False)
    print(f"✅ Ensemble predictions saved to: {filename}")

    # 7. 최종 요약
    print("\n" + "="*70)
    print("🎉 DeepFM K-Fold Ensemble Prediction Complete!")
    print("="*70)
    print(f"🎯 Experiment: {exp_name}")
    print(f"📈 Models Used: {len(models)} K-Fold models")
    print(f"🔮 Predictions: {len(predictions):,} samples")
    print(f"📁 Output: {filename}")
    print("✅ K-Fold Ensemble - Superior Generalization!")

    # 8. 개별 폴드 예측도 저장 (디버깅용)
    print(f"\n💡 Saving individual fold predictions for analysis...")
    fold_predictions = {}

    with torch.no_grad():
        for fold_idx, model in enumerate(models):
            model.to(device).eval()
            fold_preds = []

            for categorical_data, numerical_data in tqdm(batches, desc=f"Fold {fold_idx+1}"):
                batch_categorical = {}
                for key, data in categorical_data.items():
                    batch_categorical[key] = data.to(device)

                batch_numerical = None
                if numerical_data is not None:
                    batch_numerical = numerical_data.to(device)

                outputs = model(batch_categorical, batch_numerical)
                predictions_fold = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                fold_preds.append(predictions_fold)

            fold_predictions[f'fold_{fold_idx+1}'] = np.concatenate(fold_preds)

    # 개별 폴드 예측 분석
    print(f"\n📊 Individual Fold Prediction Analysis:")
    fold_means = []
    for fold_name, fold_pred in fold_predictions.items():
        fold_mean = fold_pred.mean()
        fold_std = fold_pred.std()
        fold_means.append(fold_mean)
        print(f"   {fold_name}: mean={fold_mean:.6f}, std={fold_std:.6f}")

    print(f"\n🎯 Fold Consistency:")
    print(f"   Fold means std: {np.std(fold_means):.6f} (lower is better)")
    print(f"   Ensemble mean: {predictions.mean():.6f}")

    return predictions, filename


if __name__ == "__main__":
    try:
        predictions, filename = main()
        print(f"\n🎯 K-Fold ensemble prediction completed successfully!")
        print(f"📁 Results saved to: {filename}")
    except Exception as e:
        print(f"\n❌ K-Fold ensemble prediction failed: {e}")
        import traceback
        traceback.print_exc()