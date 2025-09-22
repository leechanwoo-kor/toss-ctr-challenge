"""
DeepFM에 특화된 향상된 피처 프로세서
임베딩과 딥러닝에 최적화된 피처 엔지니어링을 포함
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict

from ..config import CATEGORICAL_FEATURES
from ..utils.config import EXCLUDE_COLUMNS
from .feature_processor import TrainingFeatureProcessor


class EnhancedDeepFMFeatureProcessor(TrainingFeatureProcessor):
    """DeepFM에 특화된 향상된 피처 프로세서"""

    def __init__(self):
        super().__init__()

        # DeepFM 특화 통계량 저장소
        self.interaction_stats = {}
        self.sequence_stats = {}
        self.embedding_friendly_features = []

        # 사용자-아이템 상호작용 통계
        self.user_item_stats = {}
        self.temporal_stats = {}

        # 임베딩 최적화를 위한 범주형 피처 조합
        self.categorical_combinations = [
            ('gender', 'age_group'),
            ('age_group', 'inventory_id'),
            ('gender', 'inventory_id'),
            ('hour', 'gender'),
            ('hour', 'age_group')
        ]

        # 딥러닝에 불리한 노이즈 피처들 (XGB 분석 기반)
        self.noise_features_to_remove = [
            'l_feat_20', 'l_feat_23',  # 상수 피처
            'l_feat_16', 'l_feat_17',  # 중복 피처
            'l_feat_24'                # 통계적 비유의
        ]

        # 저중요도 피처 (DeepFM에서 노이즈가 될 수 있음)
        self.low_value_features = [
            'feat_e_4', 'l_feat_22', 'feat_a_2', 'l_feat_18', 'l_feat_26',
            'history_b_6', 'feat_a_12', 'feat_a_8', 'l_feat_19', 'feat_a_4'
        ]

    def fit_transform(self, df):
        """훈련 데이터에 대한 피처 변환"""
        print("🚀 Processing features with DeepFM-optimized enhancements...")

        # 1. 기본 피처 처리
        df = super().fit_transform(df.copy())

        # 2. DeepFM 특화 피처 생성
        df = self._create_deepfm_features(df, is_training=True)

        # 3. 임베딩 친화적 피처 조합 생성
        df = self._create_embedding_combinations(df, is_training=True)

        # 4. 노이즈 피처 제거
        df = self._remove_noise_features(df)

        # 5. 딥러닝 최적화 (스케일링 등)
        df = self._optimize_for_deep_learning(df, is_training=True)

        print(f"✅ DeepFM features created. Final shape: {df.shape}")
        return df

    def transform(self, df):
        """테스트 데이터에 대한 피처 변환"""
        print("🔄 Transforming test data with DeepFM features...")

        df = super().transform(df)
        df = self._create_deepfm_features(df, is_training=False)
        df = self._create_embedding_combinations(df, is_training=False)
        df = self._remove_noise_features(df)
        df = self._optimize_for_deep_learning(df, is_training=False)

        return df

    def _create_deepfm_features(self, df, is_training=True):
        """DeepFM에 특화된 피처들 생성"""
        print("📊 Creating DeepFM-specific features...")

        new_features = {}

        # 1. 사용자 행동 강도 피처 (CTR 예측에 중요)
        if 'clicked' in df.columns and is_training:
            # 인벤토리별 CTR (count 기반, 리키지 방지)
            inventory_counts = df.groupby('inventory_id').size()
            inventory_popularity = inventory_counts.rank(ascending=False, method='dense')
            self.interaction_stats['inventory_popularity'] = inventory_popularity.to_dict()

            # 시간대별 활동성
            if 'hour' in df.columns:
                hour_activity = df.groupby('hour').size()
                hour_popularity = hour_activity.rank(ascending=False, method='dense')
                self.interaction_stats['hour_activity'] = hour_popularity.to_dict()

        # 인벤토리 인기도 적용
        if 'inventory_id' in df.columns:
            new_features['inventory_popularity'] = df['inventory_id'].map(
                self.interaction_stats.get('inventory_popularity', {})
            ).fillna(0).astype(np.float32)

        # 시간대 활동성 적용
        if 'hour' in df.columns:
            new_features['hour_activity'] = df['hour'].map(
                self.interaction_stats.get('hour_activity', {})
            ).fillna(0).astype(np.float32)

        # 2. 임베딩 차원 최적화를 위한 피처
        # 범주형 피처의 카디널리티 기반 가중치
        for col in self.categorical_features:
            if col in df.columns:
                cardinality = df[col].nunique()
                # 카디널리티를 정규화하여 임베딩 힌트로 사용
                new_features[f'{col}_cardinality_weight'] = (
                    np.log1p(cardinality) / 10.0
                )

        # 3. 시퀀스/패턴 기반 피처 (딥러닝이 잘 학습)
        if 'hour' in df.columns:
            # 시간 사이클 피처 (sin/cos 변환)
            new_features['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            new_features['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # 4. 상호작용 강도 피처
        if 'gender' in df.columns and 'age_group' in df.columns:
            if is_training:
                # 성별-연령 조합별 빈도
                gender_age_counts = df.groupby(['gender', 'age_group']).size()
                self.interaction_stats['gender_age_strength'] = gender_age_counts.to_dict()

            # 조합 키 생성 및 매핑
            gender_age_keys = list(zip(df['gender'], df['age_group']))
            new_features['gender_age_interaction_strength'] = pd.Series(gender_age_keys).map(
                self.interaction_stats.get('gender_age_strength', {})
            ).fillna(0).astype(np.float32)

        # 새 피처들을 DataFrame에 추가
        if new_features:
            new_feature_df = pd.DataFrame(new_features, index=df.index)
            df = pd.concat([df, new_feature_df], axis=1)

            # 수치형 피처 목록에 추가
            for feat_name in new_features.keys():
                if feat_name not in self.numerical_features:
                    self.numerical_features.append(feat_name)

        print(f"   ✅ Created {len(new_features)} DeepFM-specific features")
        return df

    def _create_embedding_combinations(self, df, is_training=True):
        """임베딩에 친화적인 범주형 피처 조합 생성"""
        print("🔗 Creating embedding-friendly combinations...")

        new_combinations = {}

        for combo in self.categorical_combinations:
            col1, col2 = combo
            if col1 in df.columns and col2 in df.columns:
                combo_name = f'{col1}_{col2}_combo'

                # 조합 피처 생성
                combo_values = df[col1].astype(str) + '_' + df[col2].astype(str)

                if is_training:
                    # 훈련 시 LabelEncoder 학습
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(combo_values)
                    self.categorical_encoders[combo_name] = le
                else:
                    # 테스트 시 기존 인코더 사용
                    le = self.categorical_encoders.get(combo_name)
                    if le is not None:
                        # 새로운 값은 0으로 처리
                        encoded_values = []
                        for val in combo_values:
                            try:
                                encoded_values.append(le.transform([val])[0])
                            except ValueError:
                                encoded_values.append(0)
                        encoded_values = np.array(encoded_values)
                    else:
                        encoded_values = np.zeros(len(combo_values))

                new_combinations[combo_name] = encoded_values.astype(np.float32)

                # 범주형 피처 목록에 추가
                if combo_name not in self.categorical_features:
                    self.categorical_features.append(combo_name)
                    self.embedding_friendly_features.append(combo_name)

        if new_combinations:
            combo_df = pd.DataFrame(new_combinations, index=df.index)
            df = pd.concat([df, combo_df], axis=1)

        print(f"   ✅ Created {len(new_combinations)} embedding combinations")
        return df

    def _remove_noise_features(self, df):
        """딥러닝에 해로운 노이즈 피처들 제거"""
        print("🗑️ Removing noise features for DeepFM...")

        features_to_remove = self.noise_features_to_remove + self.low_value_features
        removed_features = []

        for feat in features_to_remove:
            if feat in df.columns:
                df = df.drop(columns=[feat])
                removed_features.append(feat)

                # 피처 목록에서도 제거
                if feat in self.categorical_features:
                    self.categorical_features.remove(feat)
                if feat in self.numerical_features:
                    self.numerical_features.remove(feat)

        if removed_features:
            print(f"   🗑️ Removed {len(removed_features)} noise features: {removed_features[:5]}...")

        return df

    def _optimize_for_deep_learning(self, df, is_training=True):
        """딥러닝 최적화 (스케일링, 정규화 등)"""
        print("⚡ Optimizing features for deep learning...")

        # 수치형 피처들의 분포 개선
        numerical_cols = [col for col in self.numerical_features if col in df.columns]

        if numerical_cols and is_training:
            # StandardScaler로 정규화 (메모리 효율적인 배치 처리)
            self.deep_scaler = StandardScaler()

            # 대용량 데이터의 경우 배치 처리로 메모리 절약
            if len(df) > 1000000:  # 100만개 이상이면 배치 처리
                print(f"   🔄 Processing {len(df):,} samples in batches to save memory...")
                batch_size = 100000

                # 첫 번째 배치로 scaler fit
                first_batch = df[numerical_cols].iloc[:batch_size]
                self.deep_scaler.fit(first_batch)

                # 모든 배치를 변환
                for i in range(0, len(df), batch_size):
                    end_idx = min(i + batch_size, len(df))
                    batch_data = df[numerical_cols].iloc[i:end_idx]
                    df.loc[df.index[i:end_idx], numerical_cols] = self.deep_scaler.transform(batch_data)

                    if i % (batch_size * 10) == 0:  # 매 10 배치마다 진행 상황 출력
                        progress = (end_idx / len(df)) * 100
                        print(f"     📊 Batch processing: {progress:.1f}% ({end_idx:,}/{len(df):,})")
            else:
                df[numerical_cols] = self.deep_scaler.fit_transform(df[numerical_cols])

        elif numerical_cols and hasattr(self, 'deep_scaler'):
            # 테스트/검증 데이터도 배치 처리
            if len(df) > 1000000:
                batch_size = 100000
                for i in range(0, len(df), batch_size):
                    end_idx = min(i + batch_size, len(df))
                    batch_data = df[numerical_cols].iloc[i:end_idx]
                    df.loc[df.index[i:end_idx], numerical_cols] = self.deep_scaler.transform(batch_data)
            else:
                df[numerical_cols] = self.deep_scaler.transform(df[numerical_cols])

        # 범주형 피처들의 임베딩 차원 힌트 계산
        if is_training:
            self.feature_dims = {}
            for col in self.categorical_features:
                if col in df.columns:
                    n_unique = df[col].nunique()
                    # 임베딩 차원 최적화 공식 (log scale)
                    embed_dim = min(50, max(4, int(np.log2(n_unique + 1) * 2)))
                    self.feature_dims[col] = {
                        'vocab_size': n_unique + 1,  # +1 for unknown
                        'embed_dim': embed_dim
                    }

        print(f"   ⚡ Optimized {len(numerical_cols)} numerical and {len(self.categorical_features)} categorical features")
        return df

    def get_feature_importance_hints(self):
        """DeepFM 모델에 제공할 피처 중요도 힌트"""
        return {
            'embedding_friendly': self.embedding_friendly_features,
            'interaction_features': [f for f in self.numerical_features if 'interaction' in f],
            'temporal_features': [f for f in self.numerical_features if any(x in f for x in ['hour', 'sin', 'cos'])],
            'popularity_features': [f for f in self.numerical_features if 'popularity' in f or 'activity' in f]
        }