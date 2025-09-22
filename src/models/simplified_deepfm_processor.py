"""
Tree 모델과 동일한 피처를 사용하는 간소화된 DeepFM 프로세서
공정한 비교를 위해 트리 모델의 EDA 기반 피처만 사용
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from ..config import CATEGORICAL_FEATURES
from ..utils.config import EXCLUDE_COLUMNS
from .feature_processor import BaseFeatureProcessor


class SimplifiedDeepFMProcessor(BaseFeatureProcessor):
    """트리 모델과 동일한 피처를 사용하는 간소화된 DeepFM 프로세서"""

    def __init__(self):
        super().__init__()
        self.inventory_popularity = {}
        self.age_inventory_affinity = {}
        self.gender_inventory_affinity = {}
        self.new_features = []

        # 트리 모델과 동일한 피처 제거 설정
        self.constant_features_to_remove = ['l_feat_20', 'l_feat_23']
        self.duplicate_features_to_remove = ['l_feat_16', 'l_feat_17']
        self.statistical_insignificant_features = ['l_feat_24']
        self.low_importance_features = ['day_of_week']

        # 트리 모델과 동일: day_of_week 제거 (4개 범주형 피처만 사용)
        self.categorical_features = [col for col in CATEGORICAL_FEATURES if col != 'day_of_week']

        print(f"🎯 Simplified DeepFM: Using {len(self.categorical_features)} categorical features (same as Tree)")
        print(f"   Categorical features: {self.categorical_features}")

    def fit_transform(self, df):
        """트리 모델과 동일한 피처 전처리"""
        print("🚀 Processing features with Tree-model compatible approach...")

        # 1. 범주형 피처 먼저 인코딩
        for col in self.categorical_features:
            df[col] = df[col].fillna('unknown').astype(str)

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.categorical_encoders[col] = le

            self.feature_dims[col] = len(le.classes_)
            print(f"  {col}: {self.feature_dims[col]} categories")

        # 2. EDA 기반 피처 생성 (트리 모델과 동일)
        df = self._create_eda_features(df.copy(), is_training=True)

        # 3. 시퀀스 피처 처리 (트리 모델과 동일: 완전 제거)
        if 'seq' in df.columns:
            print("🚀 Excluding sequence column (following Tree model approach)")
            df = df.drop('seq', axis=1)

        # 4. 수치형 피처 정의 (EDA 피처 포함, 범주형 제외)
        excluded_cols = EXCLUDE_COLUMNS.union(set(self.categorical_features))
        self.numerical_features = [col for col in df.columns
                                 if col not in excluded_cols
                                 and pd.api.types.is_numeric_dtype(df[col])]

        print(f"📊 Feature summary:")
        print(f"   - Categorical: {len(self.categorical_features)} (same as Tree)")
        print(f"   - Numerical: {len(self.numerical_features)} (including EDA + sequence)")

        # 5. 수치형 피처 전처리 (DeepFM용 스케일링 적용)
        df = self.preprocess_numerical_features(df)

        # 6. 불필요한 피처 제거 (트리 모델과 동일)
        df = self._remove_constant_features(df)

        # 7. numerical_features 업데이트 (제거된 피처 반영)
        excluded_cols = EXCLUDE_COLUMNS.union(set(self.categorical_features))
        self.numerical_features = [col for col in df.columns
                                 if col not in excluded_cols
                                 and pd.api.types.is_numeric_dtype(df[col])]

        print(f"📊 Final feature summary:")
        print(f"   - Categorical: {len(self.categorical_features)}")
        print(f"   - Numerical: {len(self.numerical_features)} (after cleanup)")

        return df

    def transform(self, df):
        """추론용 변환 (트리 모델과 동일한 피처)"""
        print("🔧 Transforming test data with Tree-compatible features...")

        # 1. EDA 기반 피처 생성 (훈련 시와 동일)
        df = self._create_eda_features(df, is_training=False)

        # 2. 시퀀스 피처 처리 (트리 모델과 동일: 완전 제거)
        if 'seq' in df.columns:
            df = df.drop('seq', axis=1)

        # 3. 범주형 피처 변환
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)

                # 새로운 카테고리는 'unknown'으로 처리
                unknown_mask = ~df[col].isin(self.categorical_encoders[col].classes_)
                if unknown_mask.sum() > 0:
                    df.loc[unknown_mask, col] = 'unknown'

                df[col] = self.categorical_encoders[col].transform(df[col])

        # 4. 수치형 피처 전처리 (DeepFM에 필요한 스케일링 포함)
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols:
            for col in numerical_cols:
                df = self._apply_numerical_preprocessing_step(df, col, is_training=False)

            # 훈련된 스케일러 적용
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            df[numerical_cols] = df[numerical_cols].astype(np.float32)
            print(f"✅ Applied trained QuantileTransformer to {len(numerical_cols)} numerical features")

        # 5. 불필요한 피처 제거
        df = self._remove_constant_features(df)

        return df

    def _create_eda_features(self, df, is_training=True):
        """트리 모델과 동일한 EDA 기반 피처 생성 (메모리 최적화)"""
        print(f"🏪 Creating EDA-based features (training={is_training})...")

        has_gender = 'gender' in df.columns
        has_age_group = 'age_group' in df.columns
        has_inventory = 'inventory_id' in df.columns
        has_clicked = 'clicked' in df.columns

        new_feature_count = 0

        # 1. Inventory popularity (트리 모델과 동일)
        if has_inventory:
            if is_training and has_clicked:
                print("  📊 Computing inventory popularity (count-based, no leakage)...")
                inventory_counts = df.groupby('inventory_id').size()
                self.inventory_popularity = inventory_counts.rank(ascending=False).to_dict()
                print(f"     Created {len(self.inventory_popularity)} inventory popularity ranks")

            df['inventory_popularity_rank'] = df['inventory_id'].map(
                self.inventory_popularity).fillna(len(self.inventory_popularity)/2).astype(np.float32)
            new_feature_count += 1
            print(f"     ✅ inventory_popularity_rank: mean={df['inventory_popularity_rank'].mean():.1f}")

        # 2. Gender-Age combination (트리 모델과 동일)
        if has_gender and has_age_group:
            gender_age_combo = df['gender'].astype(str) + '_' + df['age_group'].astype(str)
            le_combo = LabelEncoder()
            df['gender_age_combo'] = le_combo.fit_transform(gender_age_combo).astype(np.float32)
            new_feature_count += 1

        # 3. Age-Inventory affinity (트리 모델과 동일)
        if has_age_group and has_inventory:
            if is_training:
                age_inventory_counts = df.groupby(['age_group', 'inventory_id']).size()
                self.age_inventory_affinity = age_inventory_counts.groupby('age_group').rank(ascending=False).to_dict()

            age_inventory_key = list(zip(df['age_group'], df['inventory_id']))
            df['age_inventory_affinity'] = pd.Series(age_inventory_key, index=df.index).map(
                self.age_inventory_affinity).fillna(len(df['inventory_id'].unique())/2).astype(np.float32)
            new_feature_count += 1

        # 4. Gender-Inventory affinity (트리 모델과 동일)
        if has_gender and has_inventory:
            if is_training:
                gender_inventory_counts = df.groupby(['gender', 'inventory_id']).size()
                self.gender_inventory_affinity = gender_inventory_counts.groupby('gender').rank(ascending=False).to_dict()

            gender_inventory_key = list(zip(df['gender'], df['inventory_id']))
            df['gender_inventory_affinity'] = pd.Series(gender_inventory_key, index=df.index).map(
                self.gender_inventory_affinity).fillna(len(df['inventory_id'].unique())/2).astype(np.float32)
            new_feature_count += 1

        new_features = [
            'inventory_popularity_rank',
            'gender_age_combo',
            'age_inventory_affinity',
            'gender_inventory_affinity'
        ]

        for feat in new_features:
            if feat in df.columns and feat not in self.new_features:
                self.new_features.append(feat)

        print(f"✅ Created {new_feature_count} EDA-based features (same as Tree model)")
        print(f"   New features: {[f for f in new_features if f in df.columns]}")

        return df

    def _remove_constant_features(self, df):
        """트리 모델과 동일한 피처 제거"""
        removed_features = []

        all_features_to_remove = (
            self.constant_features_to_remove +
            self.duplicate_features_to_remove +
            self.statistical_insignificant_features +
            self.low_importance_features
        )

        for feat in all_features_to_remove:
            if feat in df.columns:
                df = df.drop(columns=[feat])
                removed_features.append(feat)

        if removed_features:
            print(f"🗑️ Removed features (same as Tree): {removed_features}")

        return df

    def preprocess_numerical_features(self, df):
        """DeepFM에 필요한 수치형 피처 전처리 (스케일링 추가)"""
        print("🔧 Applying DeepFM-optimized numerical preprocessing (WITH scaling)...")

        # 1. 각 피처별 전처리 (결측치, 이상치)
        for col in self.numerical_features:
            if col in df.columns:
                df = self._apply_numerical_preprocessing_step(df, col, is_training=True)

        # 2. DeepFM에 필요한 QuantileTransformer 적용 (메모리 최적화)
        print(f"   Applying QuantileTransformer to {len(self.numerical_features)} features...")
        batch_size = 100000  # 메모리 절약을 위한 배치 처리

        if len(df) > batch_size:
            print(f"   Processing in batches of {batch_size:,} samples...")
            # 먼저 샘플로 fit
            sample_df = df[self.numerical_features].sample(n=min(batch_size, len(df)), random_state=42)
            self.scaler.fit(sample_df)

            # 배치로 transform
            processed_features = []
            for i in range(0, len(df), batch_size):
                batch_end = min(i + batch_size, len(df))
                batch_df = df[self.numerical_features].iloc[i:batch_end]
                processed_batch = self.scaler.transform(batch_df)
                processed_features.append(processed_batch)

                if i % (batch_size * 5) == 0:  # 진행률 출력
                    print(f"   Processed {batch_end:,}/{len(df):,} samples...")

            # 결합
            df[self.numerical_features] = np.vstack(processed_features)
        else:
            df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])

        print(f"✅ Applied QuantileTransformer to {len(self.numerical_features)} numerical features (DeepFM-optimized)")

        # 3. float32로 변환 (메모리 최적화)
        df[self.numerical_features] = df[self.numerical_features].astype(np.float32)

        return df

    def get_feature_importance_groups(self):
        """피처 중요도 그룹 (트리 모델과 동일)"""
        return {
            'original_categorical': [f for f in self.categorical_features],
            'original_numerical': [f for f in self.numerical_features if f not in self.new_features and not f.startswith('seq_')],
            'eda_inventory_features': ['inventory_popularity_rank', 'age_inventory_affinity', 'gender_inventory_affinity'],
            'eda_user_features': ['gender_age_combo'],
            'sequence_features': [f for f in self.numerical_features if f.startswith('seq_')],
            'all_new_features': self.new_features
        }