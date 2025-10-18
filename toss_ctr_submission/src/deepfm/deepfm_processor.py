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


class DeepFMProcessor:
    """독립적인 DeepFM 피처 프로세서"""

    def __init__(self):
        self.categorical_features = ['gender', 'age_group', 'inventory_id', 'hour', 'day_of_week']
        self.categorical_encoders = {}
        self.categorical_feature_dims = {}
        self.numerical_features = []
        self.numerical_medians = {}  # 수치형 피처의 median 값 저장 (추론 시 사용)
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
        self.inventory_frequency_map = {}

    def fit_transform(self, df):
        """훈련 데이터 처리 및 프로세서 학습"""
        print("DeepFM Processor: Training data processing...")

        df = df.copy()

        # 1. EDA 기반 피처 생성 먼저 (범주형 인코딩 전에)
        print("Creating EDA features...")
        df = self._create_eda_features(df, is_training=True)

        # 2. 범주형 피처 인코딩 (EDA 피처 생성 후에)
        print("Encoding categorical features...")
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

        # 3. 시퀀스 피처 처리 - DISABLED (순수 DeepFM 테스트)
        if 'seq' in df.columns:
            print("Removing sequence column (pure DeepFM test)...")
            df = df.drop('seq', axis=1)
            print("   Sequence column removed - using only categorical/numerical features")

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
        print("Scaling numerical features...")
        if self.numerical_features:
            # 결측치 처리 및 median 저장 (추론 시 사용)
            for col in self.numerical_features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    median_value = df[col].median()
                    self.numerical_medians[col] = median_value  # median 저장!
                    df[col] = df[col].fillna(median_value).astype(np.float32)

            # 스케일러 학습 및 적용
            df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])
            df[self.numerical_features] = df[self.numerical_features].astype(np.float32)

            print(f"   Scaled {len(self.numerical_features)} numerical features")
            print(f"   Saved {len(self.numerical_medians)} median values for inference")

        print(f"Processing complete!")
        print(f"   Categorical: {len(self.categorical_features)} features")
        print(f"   Numerical: {len(self.numerical_features)} features")
        print(f"   Total: {len(self.categorical_features) + len(self.numerical_features)} features")

        return df

    def transform(self, df):
        """추론 데이터 처리 (학습된 파라미터 사용)"""
        print("DeepFM Processor: Inference data processing...")

        df = df.copy()

        # 1. EDA 기반 피처 생성 (추론 모드)
        print("   Creating EDA features...")
        df = self._create_eda_features(df, is_training=False)

        # 2. 범주형 피처 인코딩 (학습된 인코더 사용)
        print("   Encoding categorical features...")
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)

                # 새로운 카테고리는 가장 빈도 높은 클래스로 처리 (unknown이 classes_에 없을 수 있음)
                unknown_mask = ~df[col].isin(self.categorical_encoders[col].classes_)
                if unknown_mask.sum() > 0:
                    # 첫 번째 클래스 (가장 빈도 높은 값)로 대체
                    most_frequent_class = self.categorical_encoders[col].classes_[0]
                    df.loc[unknown_mask, col] = most_frequent_class
                    print(f"   {col}: {unknown_mask.sum()} unknown values mapped to '{most_frequent_class}'")

                df[col] = self.categorical_encoders[col].transform(df[col])

        # 3. 시퀀스 피처 처리 - DISABLED (순수 DeepFM 테스트)
        if 'seq' in df.columns:
            print("   Removing sequence column (pure DeepFM test)...")
            df = df.drop('seq', axis=1)

        # 4. 불필요한 피처 제거 (훈련 시와 동일)
        df = self._remove_problematic_features(df)

        # 5. 수치형 피처 스케일링 (학습된 스케일러 사용)
        print("   Scaling numerical features...")
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols:
            # 결측치 처리 - 훈련 시 학습한 median 사용 (CRITICAL FIX!)
            for col in numerical_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # 학습된 median 사용 (없으면 0으로 fallback)
                median_value = self.numerical_medians.get(col, 0)
                df[col] = df[col].fillna(median_value).astype(np.float32)

            # 스케일러 적용
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            df[numerical_cols] = df[numerical_cols].astype(np.float32)

            print(f"   Applied scaler to {len(numerical_cols)} features (using learned medians)")

        print(f"Transform complete: {df.shape}")
        return df

    def _create_eda_features(self, df, is_training=True):
        """EDA 기반 피처 생성 (SimplifiedDeepFMProcessor에서 가져온 로직)"""

        has_gender = 'gender' in df.columns
        has_age_group = 'age_group' in df.columns
        has_inventory = 'inventory_id' in df.columns
        has_clicked = 'clicked' in df.columns

        # 0. 희귀 카테고리 통합 (과적합 방지)
        if has_inventory:
            if is_training:
                # 훈련 시: 희귀 카테고리 식별 및 저장
                rare_threshold = len(df) * 0.001  # 0.1% 미만
                value_counts = df['inventory_id'].value_counts()
                self.rare_inventory_values = value_counts[value_counts < rare_threshold].index.tolist()

                if len(self.rare_inventory_values) > 0:
                    print(f"   Found {len(self.rare_inventory_values)} rare inventory categories (< 0.1%)")
                    df.loc[df['inventory_id'].isin(self.rare_inventory_values), 'inventory_id'] = 'RARE'
            else:
                # 추론 시: 훈련에서 식별된 희귀 카테고리 통합
                if hasattr(self, 'rare_inventory_values') and len(self.rare_inventory_values) > 0:
                    df.loc[df['inventory_id'].isin(self.rare_inventory_values), 'inventory_id'] = 'RARE'

        # 1. Inventory popularity & frequency features
        if has_inventory:
            if is_training:
                inventory_counts = df.groupby('inventory_id').size()
                self.inventory_popularity = inventory_counts.rank(ascending=False).to_dict()
                self.inventory_frequency_map = inventory_counts.to_dict()

            df['inventory_popularity_rank'] = df['inventory_id'].map(
                self.inventory_popularity
            ).fillna(len(self.inventory_popularity) // 2).astype(np.float32)

            # NEW: Inventory frequency features
            df['inventory_frequency'] = df['inventory_id'].map(
                self.inventory_frequency_map
            ).fillna(1).astype(np.float32)

            df['inventory_log_frequency'] = np.log1p(df['inventory_frequency']).astype(np.float32)

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

        # 5. Time cyclical encoding features (hour)
        if 'hour' in df.columns:
            # Ensure hour is numeric and handle any data type issues
            hour_values = pd.to_numeric(df['hour'], errors='coerce').fillna(12)
            hour_values = hour_values.clip(0, 23)  # Ensure valid hour range

            # Create cyclical features using numeric values
            df['hour_sin'] = np.sin(2 * np.pi * hour_values / 24).astype(np.float32)
            df['hour_cos'] = np.cos(2 * np.pi * hour_values / 24).astype(np.float32)

            # Keep hour as string for categorical encoding (will be converted to int later)
            df['hour'] = hour_values.astype(int).astype(str).str.zfill(2)  # 0 -> '00', 1 -> '01'

        # 6. Day of week cyclical encoding
        if 'day_of_week' in df.columns:
            # Ensure day_of_week is numeric (0-6)
            day_values = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(3)  # Default to Wednesday
            day_values = day_values.clip(0, 6)  # Ensure valid day range

            # Create cyclical features
            df['day_sin'] = np.sin(2 * np.pi * day_values / 7).astype(np.float32)
            df['day_cos'] = np.cos(2 * np.pi * day_values / 7).astype(np.float32)

            # Keep day_of_week as string for categorical encoding
            df['day_of_week'] = day_values.astype(int).astype(str)  # 0 -> '0', 1 -> '1', etc.

        # 7. 결측치 인디케이터 피처 (High Impact - CTR 45.5% 상승!)
        # feat_a 계열: 18개 피처가 항상 함께 결측됨 → 하나의 통합 인디케이터
        if 'feat_a_1' in df.columns:
            df['feat_a_is_missing'] = df['feat_a_1'].isna().astype(np.float32)
            # 결측치 시 CTR: 0.0277 vs 비결측치 CTR: 0.0191 (45% 차이)

        # gender + age_group: 항상 함께 결측됨 (17,208개) → 하나의 통합 인디케이터
        if 'gender' in df.columns:
            df['user_info_missing'] = df['gender'].isna().astype(np.float32)
            # 결측치 시 CTR: 0.0236 vs 비결측치 CTR: 0.0191 (24% 차이)

        return df

    def _remove_problematic_features(self, df):
        """문제가 되는 피처들 제거"""
        features_to_remove = [
            'l_feat_20', 'l_feat_23',  # 상수 피처
            'l_feat_16', 'l_feat_17',  # 중복 피처
            'l_feat_24',               # 통계적 무의미
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
        print(f"[OK] Processor saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """프로세서 로드"""
        with open(filepath, 'rb') as f:
            processor = pickle.load(f)
        print(f"[OK] Processor loaded from {filepath}")
        return processor