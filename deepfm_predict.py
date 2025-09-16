"""
DeepFM 예측 전용 스크립트
이미 훈련된 모델로 테스트 데이터 예측
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle

# 기존 클래스들 import (동일한 정의 필요)
class FeatureProcessor:
    """DeepFM용 피처 전처리기"""
    def __init__(self):
        self.categorical_encoders = {}
        self.categorical_features = []
        self.numerical_features = []
        self.feature_dims = {}

    def process_sequence_features(self, seq_series):
        """시퀀스 컬럼을 통계적 피처로 변환"""
        print("📊 Processing sequence features...")

        seq_features = pd.DataFrame(index=seq_series.index)

        seq_lengths = []
        seq_means = []
        seq_stds = []
        seq_mins = []
        seq_maxs = []
        seq_sums = []

        for seq_str in tqdm(seq_series.astype(str), desc="Processing sequences"):
            if seq_str and seq_str != 'nan' and seq_str != '':
                try:
                    seq_array = np.fromstring(seq_str, sep=",", dtype=np.float32)
                    if len(seq_array) > 0:
                        seq_lengths.append(len(seq_array))
                        seq_means.append(np.mean(seq_array))
                        seq_stds.append(np.std(seq_array))
                        seq_mins.append(np.min(seq_array))
                        seq_maxs.append(np.max(seq_array))
                        seq_sums.append(np.sum(seq_array))
                    else:
                        seq_lengths.append(0)
                        seq_means.append(0.0)
                        seq_stds.append(0.0)
                        seq_mins.append(0.0)
                        seq_maxs.append(0.0)
                        seq_sums.append(0.0)
                except:
                    seq_lengths.append(0)
                    seq_means.append(0.0)
                    seq_stds.append(0.0)
                    seq_mins.append(0.0)
                    seq_maxs.append(0.0)
                    seq_sums.append(0.0)
            else:
                seq_lengths.append(0)
                seq_means.append(0.0)
                seq_stds.append(0.0)
                seq_mins.append(0.0)
                seq_maxs.append(0.0)
                seq_sums.append(0.0)

        seq_features['seq_length'] = seq_lengths
        seq_features['seq_mean'] = seq_means
        seq_features['seq_std'] = seq_stds
        seq_features['seq_min'] = seq_mins
        seq_features['seq_max'] = seq_maxs
        seq_features['seq_sum'] = seq_sums

        # 파생 피처
        seq_features['seq_range'] = seq_features['seq_max'] - seq_features['seq_min']
        seq_features['seq_mean_per_len'] = seq_features['seq_mean'] / (seq_features['seq_length'] + 1)

        return seq_features.astype(np.float32)

class DeepFM(nn.Module):
    """DeepFM 모델 구현"""
    def __init__(self, feature_dims, embedding_dim=16, hidden_units=[512, 256, 128], dropout=0.2, num_numerical=100):
        super(DeepFM, self).__init__()

        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        self.num_numerical = num_numerical

        # === FM Part ===
        self.fm_embeddings = nn.ModuleDict()
        self.fm_bias = nn.ModuleDict()

        for feature, dim in feature_dims.items():
            self.fm_bias[feature] = nn.Embedding(dim, 1)
            self.fm_embeddings[feature] = nn.Embedding(dim, embedding_dim)

        self.numerical_linear = nn.Linear(num_numerical, 1)
        self.numerical_embedding = nn.Linear(num_numerical, embedding_dim)

        # === Deep Part ===
        total_embedding_dim = len(feature_dims) * embedding_dim + embedding_dim

        deep_layers = []
        input_dim = total_embedding_dim

        for hidden_dim in hidden_units:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        deep_layers.append(nn.Linear(input_dim, 1))
        self.deep_layers = nn.Sequential(*deep_layers)

        self.output_layer = nn.Linear(2, 1)

    def forward(self, categorical_inputs, numerical_inputs):
        batch_size = numerical_inputs.size(0)

        # === FM Part ===
        fm_first_order = self.numerical_linear(numerical_inputs)

        for feature, values in categorical_inputs.items():
            fm_first_order += self.fm_bias[feature](values)

        embeddings = []

        for feature, values in categorical_inputs.items():
            emb = self.fm_embeddings[feature](values)
            embeddings.append(emb)

        numerical_emb = self.numerical_embedding(numerical_inputs)
        embeddings.append(numerical_emb)

        embeddings = torch.stack(embeddings, dim=1)

        sum_square = torch.sum(embeddings, dim=1) ** 2
        square_sum = torch.sum(embeddings ** 2, dim=1)
        fm_second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

        fm_output = fm_first_order + fm_second_order

        # === Deep Part ===
        deep_input = embeddings.view(batch_size, -1)
        deep_output = self.deep_layers(deep_input)

        # === 최종 결합 ===
        combined = torch.cat([fm_output, deep_output], dim=1)
        output = self.output_layer(combined)

        return output.squeeze(1)

def predict_with_saved_model():
    """저장된 모델로 예측"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔮 Starting prediction with device: {device}")

    # 저장된 체크포인트 로드
    print("📂 Loading saved model...")
    checkpoint = torch.load('deepfm_best_model.pth', weights_only=False)

    # 프로세서 복원
    processor = checkpoint['processor']
    print(f"✅ Processor loaded: {len(processor.categorical_features)} categorical, {len(processor.numerical_features)} numerical")

    # 테스트 데이터 로드
    print("📊 Loading test data...")
    test = pd.read_parquet("data/test.parquet", engine="pyarrow")
    if 'ID' in test.columns:
        test = test.drop(columns=['ID'])
    print(f"Test data shape: {test.shape}")

    # 테스트 데이터 전처리
    print("🔧 Processing test data...")

    # 범주형 피처 변환
    for col in processor.categorical_features:
        test[col] = test[col].fillna('unknown').astype(str)
        unknown_mask = ~test[col].isin(processor.categorical_encoders[col].classes_)
        test.loc[unknown_mask, col] = 'unknown'
        test[col] = processor.categorical_encoders[col].transform(test[col])

    # 시퀀스 피처 처리
    if 'seq' in test.columns:
        seq_features = processor.process_sequence_features(test['seq'])
        test = pd.concat([test, seq_features], axis=1)

    # 수치형 피처 처리
    existing_numerical = [col for col in processor.numerical_features if col in test.columns]
    if existing_numerical:
        test[existing_numerical] = test[existing_numerical].fillna(0).astype(np.float32)

    print(f"✅ Test data processed: {test.shape}")

    # 모델 초기화 및 가중치 로드
    print("🧠 Initializing model...")
    model = DeepFM(
        feature_dims=processor.feature_dims,
        embedding_dim=16,
        hidden_units=[512, 256, 128],
        dropout=0.2,
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
        for start_idx in tqdm(range(0, len(test), batch_size), desc="Predicting"):
            end_idx = min(start_idx + batch_size, len(test))
            batch = test.iloc[start_idx:end_idx]

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
        output_path = 'submissions/deepfm_submission.csv'
    else:
        submit_path = '../submissions/sample_submission.csv'
        output_path = '../submissions/deepfm_submission.csv'

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
    import os
    predictions = predict_with_saved_model()
    print("\n🏆 Prediction completed successfully!")