"""
DeepFM 전용 독립적인 피처 프로세서
XGBoost 의존성 없이 깔끔한 아키텍처로 설계
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.sequence_utils import process_sequence_features


class DeepFMProcessor:
    """독립적인 DeepFM 피처 프로세서"""

    def __init__(self):
        self.categorical_features = ['gender', 'age_group', 'inventory_id', 'hour']
        self.categorical_encoders = {}
        self.categorical_feature_dims = {}
        self.numerical_features = []
        self.scaler = QuantileTransformer(
            output_distribution="uniform",
            n_quantiles=1000,
            subsample=100000,
            random_state=42
        )

        # EDA 피처 매핑 정보 (훈련 시 생성)
        self.inventory_popularity = {}
        self.age_inventory_affinity = {}
        self.gender_inventory_affinity = {}

    def fit_transform(self, df):
        """훈련 데이터 처리 및 프로세서 학습"""
        print("🚀 DeepFM Processor: Training data processing...")

        df = df.copy()

        # 1. 범주형 피처 인코딩
        print("📊 Encoding categorical features...")
        for col in self.categorical_features:
            if col in df.columns:
                # 결측치 처리
                df[col] = df[col].fillna('unknown').astype(str)

                # LabelEncoder 학습 및 적용
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

                # 저장
                self.categorical_encoders[col] = le
                self.categorical_feature_dims[col] = len(le.classes_)

                print(f"   {col}: {self.categorical_feature_dims[col]} categories")

        # 2. EDA 기반 피처 생성
        print("🏪 Creating EDA features...")
        df = self._create_eda_features(df, is_training=True)

        # 3. 시퀀스 피처 처리 (우리가 만든 utils 사용)
        if 'seq' in df.columns:
            print("📈 Processing sequence features...")
            seq_features = process_sequence_features(df['seq'])

            # 기존 seq 컬럼 제거 후 새 피처 추가
            df = df.drop('seq', axis=1)

            # 인덱스 맞춰서 결합
            seq_features.index = df.index
            df = pd.concat([df, seq_features], axis=1)

            print(f"   Added {seq_features.shape[1]} sequence statistical features")

        # 4. 수치형 피처 정의
        excluded_cols = {'clicked'}.union(set(self.categorical_features))
        self.numerical_features = [
            col for col in df.columns
            if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        # 5. 불필요한 피처 제거 (SimplifiedDeepFMProcessor에서 가져온 로직)
        df = self._remove_problematic_features(df)

        # numerical_features 업데이트 (제거된 피처 반영)
        excluded_cols = {'clicked'}.union(set(self.categorical_features))
        self.numerical_features = [
            col for col in df.columns
            if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        # 6. 수치형 피처 스케일링
        print("⚖️ Scaling numerical features...")
        if self.numerical_features:
            # 결측치 처리
            for col in self.numerical_features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median()).astype(np.float32)

            # 스케일러 학습 및 적용
            df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])
            df[self.numerical_features] = df[self.numerical_features].astype(np.float32)

            print(f"   Scaled {len(self.numerical_features)} numerical features")

        print(f"✅ Processing complete!")
        print(f"   Categorical: {len(self.categorical_features)} features")
        print(f"   Numerical: {len(self.numerical_features)} features")
        print(f"   Total: {len(self.categorical_features) + len(self.numerical_features)} features")

        return df

    def transform(self, df):
        """추론 데이터 처리 (학습된 파라미터 사용)"""
        print("🔧 DeepFM Processor: Inference data processing...")

        df = df.copy()

        # 1. EDA 기반 피처 생성 (추론 모드)
        print("   Creating EDA features...")
        df = self._create_eda_features(df, is_training=False)

        # 2. 시퀀스 피처 처리
        if 'seq' in df.columns:
            print("   Processing sequence features...")
            seq_features = process_sequence_features(df['seq'])

            df = df.drop('seq', axis=1)
            seq_features.index = df.index
            df = pd.concat([df, seq_features], axis=1)

        # 3. 범주형 피처 인코딩 (학습된 인코더 사용)
        print("   Encoding categorical features...")
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)

                # 새로운 카테고리는 'unknown'으로 처리
                unknown_mask = ~df[col].isin(self.categorical_encoders[col].classes_)
                if unknown_mask.sum() > 0:
                    df.loc[unknown_mask, col] = 'unknown'

                df[col] = self.categorical_encoders[col].transform(df[col])

        # 4. 불필요한 피처 제거 (훈련 시와 동일)
        df = self._remove_problematic_features(df)

        # 5. 수치형 피처 스케일링 (학습된 스케일러 사용)
        print("   Scaling numerical features...")
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols:
            # 결측치 처리
            for col in numerical_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0).astype(np.float32)  # 추론 시에는 0으로 채움

            # 스케일러 적용
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            df[numerical_cols] = df[numerical_cols].astype(np.float32)

            print(f"   Applied scaler to {len(numerical_cols)} features")

        print(f"✅ Transform complete: {df.shape}")
        return df

    def _create_eda_features(self, df, is_training=True):
        """EDA 기반 피처 생성 (SimplifiedDeepFMProcessor에서 가져온 로직)"""

        has_gender = 'gender' in df.columns
        has_age_group = 'age_group' in df.columns
        has_inventory = 'inventory_id' in df.columns
        has_clicked = 'clicked' in df.columns

        # 1. Inventory popularity (인기도 순위)
        if has_inventory:
            if is_training and has_clicked:
                inventory_counts = df.groupby('inventory_id').size()
                self.inventory_popularity = inventory_counts.rank(ascending=False).to_dict()

            df['inventory_popularity_rank'] = df['inventory_id'].map(
                self.inventory_popularity
            ).fillna(len(self.inventory_popularity) // 2).astype(np.float32)

        # 2. Gender-Age combination
        if has_gender and has_age_group:
            gender_age_combo = df['gender'].astype(str) + '_' + df['age_group'].astype(str)
            le_combo = LabelEncoder()
            df['gender_age_combo'] = le_combo.fit_transform(gender_age_combo).astype(np.float32)

        # 3. Age-Inventory affinity
        if has_age_group and has_inventory:
            if is_training:
                age_inventory_counts = df.groupby(['age_group', 'inventory_id']).size()
                self.age_inventory_affinity = age_inventory_counts.groupby('age_group').rank(ascending=False).to_dict()

            age_inventory_key = list(zip(df['age_group'], df['inventory_id']))
            df['age_inventory_affinity'] = pd.Series(age_inventory_key, index=df.index).map(
                self.age_inventory_affinity
            ).fillna(len(df['inventory_id'].unique()) // 2).astype(np.float32)

        # 4. Gender-Inventory affinity
        if has_gender and has_inventory:
            if is_training:
                gender_inventory_counts = df.groupby(['gender', 'inventory_id']).size()
                self.gender_inventory_affinity = gender_inventory_counts.groupby('gender').rank(ascending=False).to_dict()

            gender_inventory_key = list(zip(df['gender'], df['inventory_id']))
            df['gender_inventory_affinity'] = pd.Series(gender_inventory_key, index=df.index).map(
                self.gender_inventory_affinity
            ).fillna(len(df['inventory_id'].unique()) // 2).astype(np.float32)

        return df

    def _remove_problematic_features(self, df):
        """문제가 되는 피처들 제거 (SimplifiedDeepFMProcessor와 동일)"""
        features_to_remove = [
            'l_feat_20', 'l_feat_23',  # 상수 피처
            'l_feat_16', 'l_feat_17',  # 중복 피처
            'l_feat_24',               # 통계적 무의미
            'day_of_week'              # 낮은 중요도
        ]

        existing_features = [f for f in features_to_remove if f in df.columns]
        if existing_features:
            df = df.drop(columns=existing_features)
            print(f"   Removed {len(existing_features)} problematic features: {existing_features}")

        return df

    def save(self, filepath):
        """프로세서 저장"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Processor saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """프로세서 로드"""
        with open(filepath, 'rb') as f:
            processor = pickle.load(f)
        print(f"✅ Processor loaded from {filepath}")
        return processor