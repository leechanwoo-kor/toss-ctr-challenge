"""
inventory_id embedding 최적화된 DeepFM
inventory_id: 10차원, 나머지: 16차원
"""

import torch
import torch.nn as nn

class OptimizedDeepFM(nn.Module):
    """inventory_id embedding을 10차원으로 최적화한 DeepFM"""

    def __init__(self, feature_dims, default_embedding_dim=16, hidden_units=[256, 128], dropout=0.2, num_numerical=None):
        super(OptimizedDeepFM, self).__init__()

        self.feature_dims = feature_dims
        self.default_embedding_dim = default_embedding_dim
        # num_numerical이 None이면 기본값 사용 (하위 호환성)
        self.num_numerical = num_numerical if num_numerical is not None else 83

        # inventory_id는 10차원, 나머지는 16차원
        self.inventory_embedding_dim = 10

        # === FM Part ===
        self.fm_embeddings = nn.ModuleDict()
        self.fm_bias = nn.ModuleDict()

        for feature, dim in feature_dims.items():
            self.fm_bias[feature] = nn.Embedding(dim, 1)

            # inventory_id만 10차원, 나머지는 16차원
            if feature == 'inventory_id':
                self.fm_embeddings[feature] = nn.Embedding(dim, self.inventory_embedding_dim)
            else:
                self.fm_embeddings[feature] = nn.Embedding(dim, default_embedding_dim)

        self.numerical_linear = nn.Linear(num_numerical, 1)
        self.numerical_embedding = nn.Linear(num_numerical, default_embedding_dim)

        # === Deep Part ===
        # Total embedding 차원 계산 (inventory_id: 10차원, 나머지: 16차원씩)
        total_embedding_dim = 0
        for feature in feature_dims.keys():
            if feature == 'inventory_id':
                total_embedding_dim += self.inventory_embedding_dim
            else:
                total_embedding_dim += default_embedding_dim
        total_embedding_dim += default_embedding_dim  # numerical embedding

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

        # Output layer
        self.output_layer = nn.Linear(2, 1)

        print(f"OptimizedDeepFM created:")
        print(f"   inventory_id embedding: {self.inventory_embedding_dim}D")
        print(f"   Other embeddings: {default_embedding_dim}D")
        print(f"   Numerical features: {self.num_numerical}")
        print(f"   Total embedding dim: {total_embedding_dim}")

    def forward(self, categorical_features, numerical_features):
        # === FM Part ===
        fm_linear_part = 0
        fm_embeddings_list = []

        for feature, values in categorical_features.items():
            # Linear part (bias)
            fm_linear_part += self.fm_bias[feature](values).squeeze(-1)

            # Embedding part
            fm_embeddings_list.append(self.fm_embeddings[feature](values))

        if numerical_features is not None:
            fm_linear_part += self.numerical_linear(numerical_features).squeeze(-1)
            fm_embeddings_list.append(self.numerical_embedding(numerical_features).unsqueeze(1))

        # FM interaction - 간단한 방법: 각 embedding별로 interaction 계산 후 합산
        fm_interaction = 0

        # 모든 embedding 쌍에 대해 interaction 계산
        for i in range(len(fm_embeddings_list)):
            for j in range(i + 1, len(fm_embeddings_list)):
                emb_i = fm_embeddings_list[i]  # [batch, emb_dim_i] 또는 [batch, 1, emb_dim_i]
                emb_j = fm_embeddings_list[j]  # [batch, emb_dim_j] 또는 [batch, 1, emb_dim_j]

                # 차원 맞추기
                if len(emb_i.shape) == 3:
                    emb_i = emb_i.squeeze(1)  # [batch, emb_dim_i]
                if len(emb_j.shape) == 3:
                    emb_j = emb_j.squeeze(1)  # [batch, emb_dim_j]

                # 공통 차원만큼만 interaction 계산
                min_dim = min(emb_i.size(1), emb_j.size(1))
                interaction = torch.sum(emb_i[:, :min_dim] * emb_j[:, :min_dim], dim=1)
                fm_interaction += interaction

        fm_output = fm_linear_part + fm_interaction

        # === Deep Part ===
        # 원본 embedding (차원 다름) 사용
        deep_input = torch.cat([emb.view(emb.size(0), -1) for emb in fm_embeddings_list], dim=1)
        deep_output = self.deep_layers(deep_input).squeeze(-1)

        # Combine
        combined = torch.stack([fm_output, deep_output], dim=1)
        final_output = self.output_layer(combined)

        return final_output