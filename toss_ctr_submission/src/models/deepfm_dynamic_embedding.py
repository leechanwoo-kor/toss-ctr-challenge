"""
DeepFM v2.2 with Dynamic Embedding Dimensions (Based on OptimizedDeepFM)
카디널리티 기반 동적 embedding 크기 할당: 6 × cardinality^0.25

v2.2 검증된 아키텍처 기반:
- FM + Deep (Attention 없음, 단순하고 안정적)
- [256, 128] hidden layers
- Dropout 0.2~0.25
- 동적 embedding으로 파라미터 효율성 개선
"""

import torch
import torch.nn as nn
import math


def calculate_embedding_dim(cardinality, min_dim=4, max_dim=50):
    """
    카디널리티 기반 embedding 차원 계산

    공식: 6 × cardinality^(1/4)
    - Google의 경험적 규칙
    - 희귀 카테고리는 작은 차원, 빈번한 카테고리는 큰 차원

    Args:
        cardinality: 카테고리 개수
        min_dim: 최소 embedding 차원 (기본 4)
        max_dim: 최대 embedding 차원 (기본 50)

    Returns:
        int: 계산된 embedding 차원

    예시 (Toss CTR 데이터):
        gender (2 categories) → 7D (현재: 16D, 절약: 56%)
        age_group (8 categories) → 10D (현재: 16D, 절약: 38%)
        inventory_id (18 categories) → 12D (현재: 10D, 증가: 20%)
        hour (24 categories) → 13D (현재: 16D, 절약: 19%)
        day_of_week (7 categories) → 9D (현재: 16D, 절약: 44%)
    """
    dim = int(6 * (cardinality ** 0.25))
    return max(min_dim, min(max_dim, dim))


class DynamicEmbeddingDeepFM(nn.Module):
    """
    DeepFM v2.2 with Dynamic Embedding Dimensions

    OptimizedDeepFM (v2.2) 기반으로 동적 embedding 적용:
    - 검증된 FM + Deep 구조 유지
    - Attention 제거 (단순성과 안정성 우선)
    - 카디널리티 기반 embedding 크기 자동 조정
    - 파라미터 효율성 향상으로 과적합 방지
    """

    def __init__(
        self,
        feature_dims,
        hidden_units=[256, 128],
        dropout=0.25,
        num_numerical=None,
        min_embed_dim=4,
        max_embed_dim=50,
        use_dynamic_embedding=True
    ):
        super(DynamicEmbeddingDeepFM, self).__init__()

        self.feature_dims = feature_dims
        self.num_numerical = num_numerical if num_numerical is not None else 119
        self.use_dynamic_embedding = use_dynamic_embedding

        # 각 피처별 embedding 차원 계산
        self.embedding_dims = {}
        total_saved_params = 0

        if use_dynamic_embedding:
            # 동적 embedding: 6 × cardinality^0.25
            for feature, cardinality in feature_dims.items():
                dim = calculate_embedding_dim(cardinality, min_embed_dim, max_embed_dim)
                self.embedding_dims[feature] = dim

                # 기존 고정 크기와 비교
                old_dim = 10 if feature == 'inventory_id' else 16
                saved = (old_dim - dim) * cardinality
                total_saved_params += saved
        else:
            # 기존 v2.2 방식
            for feature, cardinality in feature_dims.items():
                self.embedding_dims[feature] = 10 if feature == 'inventory_id' else 16

        print(f"DynamicEmbeddingDeepFM (v2.2 based) created:")
        print(f"   Dynamic Embedding: {'Enabled' if use_dynamic_embedding else 'Disabled (v2.2 original)'}")
        print(f"   Embedding dimensions:")

        for feature in sorted(feature_dims.keys()):
            cardinality = feature_dims[feature]
            dim = self.embedding_dims[feature]
            if use_dynamic_embedding:
                old_dim = 10 if feature == 'inventory_id' else 16
                change = dim - old_dim
                change_str = f"({'+' if change > 0 else ''}{change})" if change != 0 else ""
                print(f"      {feature:15s} ({cardinality:3d} cats): {dim:2d}D {change_str}")
            else:
                print(f"      {feature:15s} ({cardinality:3d} cats): {dim:2d}D")

        if use_dynamic_embedding and total_saved_params != 0:
            print(f"   Parameter efficiency: {total_saved_params:+,} params vs v2.2")

        print(f"   Numerical features: {self.num_numerical}")
        print(f"   Hidden layers: {hidden_units}")
        print(f"   Dropout: {dropout}")

        # === FM Part ===
        self.fm_embeddings = nn.ModuleDict()
        self.fm_bias = nn.ModuleDict()

        for feature, cardinality in feature_dims.items():
            self.fm_bias[feature] = nn.Embedding(cardinality, 1)
            embed_dim = self.embedding_dims[feature]
            self.fm_embeddings[feature] = nn.Embedding(cardinality, embed_dim)

        # Numerical features
        self.numerical_embedding_dim = 16
        self.numerical_linear = nn.Linear(self.num_numerical, 1)
        self.numerical_embedding = nn.Linear(self.num_numerical, self.numerical_embedding_dim)

        # === Deep Part ===
        total_embedding_dim = sum(self.embedding_dims.values()) + self.numerical_embedding_dim

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

        # === Output Layer ===
        self.output_layer = nn.Linear(2, 1)

        print(f"   Total embedding dim: {total_embedding_dim}")

        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total parameters: {total_params:,}")

    def forward(self, categorical_features, numerical_features):
        # === FM Part ===
        fm_linear_part = 0
        fm_embeddings_list = []

        for feature, values in categorical_features.items():
            fm_linear_part += self.fm_bias[feature](values).squeeze(-1)
            fm_embeddings_list.append(self.fm_embeddings[feature](values))

        if numerical_features is not None:
            fm_linear_part += self.numerical_linear(numerical_features).squeeze(-1)
            fm_embeddings_list.append(
                self.numerical_embedding(numerical_features).unsqueeze(1)
            )

        # FM interaction
        fm_interaction = 0

        for i in range(len(fm_embeddings_list)):
            for j in range(i + 1, len(fm_embeddings_list)):
                emb_i = fm_embeddings_list[i]
                emb_j = fm_embeddings_list[j]

                if len(emb_i.shape) == 3:
                    emb_i = emb_i.squeeze(1)
                if len(emb_j.shape) == 3:
                    emb_j = emb_j.squeeze(1)

                min_dim = min(emb_i.size(1), emb_j.size(1))
                interaction = torch.sum(emb_i[:, :min_dim] * emb_j[:, :min_dim], dim=1)
                fm_interaction += interaction

        fm_output = fm_linear_part + fm_interaction

        # === Deep Part ===
        deep_input = torch.cat([
            emb.view(emb.size(0), -1) for emb in fm_embeddings_list
        ], dim=1)
        deep_output = self.deep_layers(deep_input).squeeze(-1)

        # === Fusion ===
        combined = torch.stack([fm_output, deep_output], dim=1)
        final_output = self.output_layer(combined)

        return final_output
