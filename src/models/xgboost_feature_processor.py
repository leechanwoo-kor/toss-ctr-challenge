
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ..config import CATEGORICAL_FEATURES
from ..utils.config import EXCLUDE_COLUMNS


class XGBoostFeatureProcessor:

    def __init__(self):
        # 범주형/수치형 피처 저장
        self.categorical_features = []
        self.numerical_features = []
        self.categorical_encoders = {}

        # 결측치 처리용 통계 저장 (훈련/테스트 일관성 보장)
        self.numerical_medians = {}

        # EDA 피처 관련
        self.inventory_popularity = {}
        self.age_inventory_affinity = {}
        self.gender_inventory_affinity = {}
        self.new_features = []

        # EDA 피처 인코더 및 통계 (훈련/테스트 일관성 보장)
        self.combo_encoders = {}
        self.default_ranks = {}
        self.train_inventory_count = 0

        # 제거할 피처들 정의
        self.features_to_remove = [
            'l_feat_20', 'l_feat_23',  # constant features
            'l_feat_16', 'l_feat_17',  # duplicate features
            'l_feat_24',               # statistically insignificant
            'day_of_week'              # low importance
        ]

    def fit_transform(self, df):
        """XGBoost 데이터 전처리 및 EDA 피처 생성"""
        print("Processing features for XGBoost...")

        # 1. 기본 피처 처리
        df = self._process_basic_features(df, is_training=True)

        # 2. EDA 피처 생성
        df = self._create_eda_features(df, is_training=True)

        # 3. 불필요한 피처 제거
        df = self._remove_features(df)

        return df

    def transform(self, df):
        """테스트 데이터 변환"""
        print("Transforming test data...")

        # 1. 기본 피처 처리
        df = self._process_basic_features(df, is_training=False)

        # 2. EDA 피처 생성
        df = self._create_eda_features(df, is_training=False)

        # 3. 불필요한 피처 제거
        df = self._remove_features(df)

        return df

    def _process_basic_features(self, df, is_training=True):
        """기본 피처 전처리 (범주형/수치형)"""

        if is_training:
            # 범주형 피처 정의 (day_of_week 제외)
            self.categorical_features = [
                col for col in CATEGORICAL_FEATURES
                if col in df.columns and col != 'day_of_week'
            ]

            # 수치형 피처 정의 - 데이터 타입도 확인
            exclude_cols = set(EXCLUDE_COLUMNS) | set(self.categorical_features)
            self.numerical_features = []

            for col in df.columns:
                if col not in exclude_cols:
                    # 수치형 변환 가능한지 확인
                    try:
                        pd.to_numeric(df[col].dropna().iloc[:100], errors='raise')
                        self.numerical_features.append(col)
                    except (ValueError, TypeError):
                        # 수치형 변환 불가능하면 범주형으로 처리
                        print(f"Warning: {col} contains non-numeric data, treating as categorical")
                        self.categorical_features.append(col)

        # 시퀀스 피처 제거 (먼저 처리)
        if 'seq' in df.columns:
            df = df.drop('seq', axis=1)

        # 범주형 피처 인코딩 (벡터화)
        categorical_cols = [col for col in self.categorical_features if col in df.columns]
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown').astype(str)

            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.categorical_encoders[col] = le
            else:
                # 벡터화된 unknown 처리
                le = self.categorical_encoders[col]
                unknown_mask = ~df[col].isin(le.classes_)
                if unknown_mask.any():
                    # 알려진 클래스들로만 먼저 transform
                    known_values = df[col].copy()
                    known_values.loc[unknown_mask] = le.classes_[0]  # 임시로 첫 번째 클래스 사용
                    df[col] = le.transform(known_values).astype(float)
                    # unknown은 -1로 처리 (새로운 카테고리임을 표시)
                    df.loc[unknown_mask, col] = -1
                    df[col] = df[col].astype(int)
                else:
                    df[col] = le.transform(df[col])

        # 수치형 피처 처리 (훈련/테스트 일관성 보장)
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols:
            if is_training:
                # 수치형 변환 시도
                for col in numerical_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # 훈련 시: median 계산하고 저장
                self.numerical_medians = df[numerical_cols].median().to_dict()
                df[numerical_cols] = df[numerical_cols].fillna(self.numerical_medians).astype(np.float32)
            else:
                # 테스트 시: 수치형 변환 후 저장된 훈련 median 사용
                for col in numerical_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df[numerical_cols] = df[numerical_cols].fillna(self.numerical_medians).astype(np.float32)

        return df

    def _remove_features(self, df):
        """불필요한 피처 제거"""
        # 존재하는 피처만 필터링
        features_to_drop = [feat for feat in self.features_to_remove if feat in df.columns]

        if features_to_drop:
            df = df.drop(columns=features_to_drop)
            print(f"Removed {len(features_to_drop)} features: {features_to_drop}")

        return df

    def _create_eda_features(self, df, is_training=True):
        print(f"Creating EDA features (training={is_training})...")

        new_columns = {}

        # 1. Inventory popularity ranking
        if 'inventory_id' in df.columns:
            new_columns.update(self._create_inventory_features(df, is_training))

        # 2. User demographic combinations
        if 'gender' in df.columns and 'age_group' in df.columns:
            new_columns.update(self._create_demographic_features(df, is_training))

        # 새로운 피처들을 한 번에 추가 (메모리 효율적)
        if new_columns:
            # pandas assign으로 한 번에 추가
            df = df.assign(**new_columns)

            # 새로운 피처명들 추가 (set 사용으로 효율적 중복 체크)
            new_feature_names = list(new_columns.keys())
            existing_new_features = set(self.new_features)
            existing_numerical_features = set(self.numerical_features)

            self.new_features.extend([name for name in new_feature_names if name not in existing_new_features])
            self.numerical_features.extend([name for name in new_feature_names if name not in existing_numerical_features])

            print(f"Created {len(new_columns)} EDA features")

        return df

    def _create_inventory_features(self, df, is_training=True):
        """Create inventory-related features"""
        new_features = {}

        if is_training and 'clicked' in df.columns:
            inventory_counts = df.groupby('inventory_id').size()
            self.inventory_popularity = inventory_counts.rank(ascending=False).to_dict()
            # 훈련 시 inventory 수 저장
            self.train_inventory_count = len(inventory_counts)
            self.default_ranks['inventory_popularity'] = self.train_inventory_count / 2

        if self.inventory_popularity:
            # 훈련 시 계산된 default_rank 사용
            default_rank = self.default_ranks.get('inventory_popularity', self.train_inventory_count / 2)
            new_features['inventory_popularity_rank'] = (
                df['inventory_id'].map(self.inventory_popularity)
                .fillna(default_rank).astype(np.float32)
            )

        return new_features

    def _create_demographic_features(self, df, is_training=True):
        """Create demographic combination features"""
        new_features = {}

        # Gender + Age combination (메모리 효율적)
        new_features['gender_age_combo'] = self._create_combo_feature(
            df, ['gender', 'age_group'], 'gender_age_combo', is_training
        ).astype(np.float32)

        # Inventory 관련 affinity 피처들
        if 'inventory_id' in df.columns:
            # 훈련/테스트 일관성 보장
            if is_training:
                self.default_ranks['affinity'] = len(df['inventory_id'].unique()) / 2
            default_rank = self.default_ranks.get('affinity', 0)

            # Age + Inventory affinity
            if is_training:
                self.age_inventory_affinity = self._compute_affinity_mapping(
                    df, ['age_group', 'inventory_id'], 'age_group'
                )

            if self.age_inventory_affinity:
                new_features['age_inventory_affinity'] = self._apply_affinity_mapping(
                    df, ['age_group', 'inventory_id'], self.age_inventory_affinity, default_rank
                )

            # Gender + Inventory affinity
            if is_training:
                self.gender_inventory_affinity = self._compute_affinity_mapping(
                    df, ['gender', 'inventory_id'], 'gender'
                )

            if self.gender_inventory_affinity:
                new_features['gender_inventory_affinity'] = self._apply_affinity_mapping(
                    df, ['gender', 'inventory_id'], self.gender_inventory_affinity, default_rank
                )

        return new_features

    def _create_combo_feature(self, df, columns, feature_name, is_training=True):
        """메모리 효율적인 조합 피처 생성 (훈련/테스트 일관성 보장)"""
        # 조합 값 생성
        combo_values = df[columns[0]].astype(str)
        for col in columns[1:]:
            combo_values = combo_values + '_' + df[col].astype(str)

        if is_training:
            # 훈련 시: LabelEncoder 생성하고 저장
            le = LabelEncoder()
            result = le.fit_transform(combo_values)
            self.combo_encoders[feature_name] = le
            return result
        else:
            # 테스트 시: 저장된 LabelEncoder 사용
            le = self.combo_encoders.get(feature_name)
            if le is None:
                raise ValueError(f"No encoder found for {feature_name}. Make sure fit_transform was called first.")

            # 새로운 조합은 가장 큰 값 + 1로 처리
            unknown_mask = ~combo_values.isin(le.classes_)
            if unknown_mask.any():
                # 임시로 known 값을 처리
                known_values = combo_values.copy()
                known_values.loc[unknown_mask] = le.classes_[0]  # 첫 번째 클래스로 임시 처리
                result = le.transform(known_values).astype(float)
                # unknown은 max + 1로 설정
                result[unknown_mask] = len(le.classes_)
                return result.astype(int)
            else:
                return le.transform(combo_values)

    def _compute_affinity_mapping(self, df, group_cols, rank_by_col):
        """Affinity 매핑 계산 (중복 로직 통합)"""
        counts = df.groupby(group_cols).size()
        return counts.groupby(rank_by_col).rank(ascending=False).to_dict()

    def _apply_affinity_mapping(self, df, columns, mapping, default_rank):
        """Affinity 매핑 적용 (메모리 효율적)"""
        # list(zip()) 대신 pandas MultiIndex 사용
        multi_index = pd.MultiIndex.from_arrays([df[col] for col in columns])
        result = (
            multi_index.to_series().map(mapping)
            .fillna(default_rank)
            .astype(np.float32)
            .values
        )
        # 메모리 정리
        del multi_index
        return result

    def get_feature_importance_groups(self):
        """Return feature groups for analysis"""
        return {
            'original_categorical': [f for f in self.categorical_features],
            'original_numerical': [f for f in self.numerical_features if f not in self.new_features],
            'eda_inventory_features': ['inventory_popularity_rank', 'age_inventory_affinity', 'gender_inventory_affinity'],
            'eda_user_features': ['gender_age_combo'],
            'all_new_features': self.new_features
        }