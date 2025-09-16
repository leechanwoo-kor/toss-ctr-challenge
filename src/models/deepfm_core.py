"""
DeepFM 핵심 모델 및 데이터셋 클래스
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


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


class DeepFMDataset(Dataset):
    """DeepFM용 데이터셋"""

    def __init__(self, df, processor, target_col=None, has_target=True):
        self.processor = processor
        self.has_target = has_target
        self.target_col = target_col

        # 범주형 피처 (임베딩용)
        self.categorical_data = {}
        for col in processor.categorical_features:
            self.categorical_data[col] = torch.LongTensor(df[col].values)

        # 수치형 피처
        self.numerical_data = torch.FloatTensor(df[processor.numerical_features].values)

        if has_target:
            self.targets = torch.FloatTensor(df[target_col].values)

        print(f"Dataset created: {len(df)} samples")
        print(f"Categorical features shape: {[(k, v.shape) for k, v in self.categorical_data.items()]}")
        print(f"Numerical features shape: {self.numerical_data.shape}")

    def __len__(self):
        return len(self.numerical_data)

    def __getitem__(self, idx):
        categorical_inputs = {col: data[idx] for col, data in self.categorical_data.items()}
        numerical_inputs = self.numerical_data[idx]

        if self.has_target:
            return categorical_inputs, numerical_inputs, self.targets[idx]
        else:
            return categorical_inputs, numerical_inputs