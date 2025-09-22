"""
DeepFM용 피처 전처리 클래스들
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from ..config import CATEGORICAL_FEATURES
from ..utils.config import EXCLUDE_COLUMNS


class BaseFeatureProcessor:
    """기본 피처 전처리기"""

    def __init__(self):
        self.categorical_encoders = {}
        self.categorical_features = []
        self.numerical_features = []
        self.feature_dims = {}
        self.scaler = QuantileTransformer(
            output_distribution='uniform',  # 균등분포로 변환 (CTR 예측에 최적)
            n_quantiles=1000,              # 충분한 분위수
            subsample=100000,              # 대용량 데이터 샘플링
            random_state=42                # 재현성 보장
        )
        self.numerical_stats = {}

    def _apply_numerical_preprocessing_step(self, df, col, is_training=True):
        """통합된 수치형 피처 전처리 단계"""
        if is_training:
            # 훈련 시: 통계 계산 및 저장
            missing_pct = df[col].isnull().sum() / len(df) * 100
            median_val = df[col].median()
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)

            self.numerical_stats[col] = {
                'median': median_val,
                'missing_pct': missing_pct,
                'q01': q01,
                'q99': q99
            }

            # 결측치 대체
            if missing_pct > 0:
                df[col] = df[col].fillna(median_val)
                if missing_pct > 5:
                    print(f"  {col}: {missing_pct:.2f}% missing → filled with median {median_val:.4f}")

            # 이상치 클리핑
            outlier_mask = (df[col] > q99) | (df[col] < q01)
            outlier_pct = outlier_mask.sum() / len(df) * 100
            if outlier_pct > 1:
                df[col] = np.clip(df[col], q01, q99)
                print(f"  {col}: {outlier_pct:.2f}% outliers clipped [{q01:.4f}, {q99:.4f}]")

        else:
            # 추론 시: 저장된 통계 사용
            if col in self.numerical_stats:
                stats = self.numerical_stats[col]
                median_val, q01, q99 = stats['median'], stats['q01'], stats['q99']

                # 결측치 대체
                missing_before = df[col].isnull().sum()
                df[col] = df[col].fillna(median_val)
                if missing_before > 0:
                    print(f"  {col}: {missing_before} missing → filled with training median {median_val:.4f}")

                # 클리핑
                outlier_mask = (df[col] > q99) | (df[col] < q01)
                if outlier_mask.sum() > 0:
                    df[col] = np.clip(df[col], q01, q99)
                    print(f"  {col}: {outlier_mask.sum()} values clipped using training bounds [{q01:.4f}, {q99:.4f}]")
            else:
                # Fallback
                df[col] = df[col].fillna(df[col].median())
                print(f"  {col}: No training stats found, using test median for missing values")

        return df

    def process_sequence_features(self, seq_series):
        """시퀀스 컬럼을 통계적 피처로 변환 (벡터화 최적화)"""
        print("📊 Processing sequence features (vectorized)...")

        def extract_sequence_stats(seq_str):
            """단일 시퀀스에서 통계 추출"""
            if pd.isna(seq_str) or seq_str == '' or seq_str == 'nan':
                return [0, 0.0, 0.0, 0.0, 0.0, 0.0]

            try:
                seq_array = np.fromstring(str(seq_str), sep=",", dtype=np.float32)
                if len(seq_array) == 0:
                    return [0, 0.0, 0.0, 0.0, 0.0, 0.0]

                return [
                    len(seq_array),           # length
                    float(np.mean(seq_array)), # mean
                    float(np.std(seq_array)),  # std
                    float(np.min(seq_array)),  # min
                    float(np.max(seq_array)),  # max
                    float(np.sum(seq_array))   # sum
                ]
            except Exception:
                return [0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 벡터화된 처리 (리스트 컴프리헨션 사용)
        print(f"  Processing {len(seq_series):,} sequences...")
        stats_list = [extract_sequence_stats(seq) for seq in tqdm(seq_series, desc="Processing sequences")]
        stats_array = np.array(stats_list, dtype=np.float32)

        # DataFrame 생성
        seq_features = pd.DataFrame({
            'seq_length': stats_array[:, 0],
            'seq_mean': stats_array[:, 1],
            'seq_std': stats_array[:, 2],
            'seq_min': stats_array[:, 3],
            'seq_max': stats_array[:, 4],
            'seq_sum': stats_array[:, 5]
        }, index=seq_series.index)

        # 개선된 파생 피처 (카테고리 시퀀스에 최적화)
        print("  Generating improved derived features...")

        # 1. 다양성 측정: 표준편차 대비 범위 (정규화된 변동성)
        seq_features['seq_normalized_range'] = (seq_features['seq_max'] - seq_features['seq_min']) / (seq_features['seq_std'] + 1e-8)

        # 2. 밀도 측정: 합계 대비 길이 (행동 강도)
        seq_features['seq_density'] = seq_features['seq_sum'] / (seq_features['seq_length'] + 1)

        # 3. 변동성 측정: 변동계수 (일관성 지표)
        seq_features['seq_cv'] = seq_features['seq_std'] / (seq_features['seq_mean'] + 1e-8)

        print(f"  Generated {len(seq_features.columns)} sequence features")
        return seq_features.astype(np.float32)


class TrainingFeatureProcessor(BaseFeatureProcessor):
    """훈련용 피처 전처리기"""

    def fit_transform(self, df):
        """피처 전처리 및 변환"""
        print("🔧 Processing features for DeepFM...")

        # 범주형 피처 정의
        self.categorical_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]

        # 수치형 피처 정의
        self.numerical_features = [col for col in df.columns
                                 if col not in EXCLUDE_COLUMNS
                                 and col not in self.categorical_features]

        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        print(f"Numerical features ({len(self.numerical_features)}): {len(self.numerical_features)}")

        # 범주형 피처 인코딩
        for col in self.categorical_features:
            df[col] = df[col].fillna('unknown').astype(str)

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.categorical_encoders[col] = le

            self.feature_dims[col] = len(le.classes_)
            print(f"  {col}: {self.feature_dims[col]} categories")

        # 수치형 피처 전처리 (개선된 방식)
        df = self.preprocess_numerical_features(df)

        # 시퀀스 피처 처리 (메모리 최적화)
        if 'seq' in df.columns:
            seq_features = self.process_sequence_features(df['seq'])

            # 새로운 시퀀스 피처들을 수치형에 추가
            new_seq_cols = list(seq_features.columns)
            self.numerical_features.extend(new_seq_cols)
            print(f"Added sequence features: {new_seq_cols}")

            # 원본 seq 컬럼 제거하고 새 피처 추가 (메모리 절약)
            df = df.drop('seq', axis=1)
            df = pd.concat([df, seq_features], axis=1)

            # seq_features 메모리 해제
            del seq_features

        return df

    def preprocess_numerical_features(self, df):
        """개선된 수치형 피처 전처리 (훈련용)"""
        print("🔧 Applying improved numerical preprocessing...")

        # 1. 각 피처별 전처리 (통합된 메서드 사용)
        for col in self.numerical_features:
            if col in df.columns:
                df = self._apply_numerical_preprocessing_step(df, col, is_training=True)

        # 2. QuantileTransformer 적용
        df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])
        print(f"  Applied QuantileTransformer to {len(self.numerical_features)} numerical features")

        # 3. float32로 변환
        df[self.numerical_features] = df[self.numerical_features].astype(np.float32)

        return df


class TreeBasedFeatureProcessor(BaseFeatureProcessor):
    """트리 기반 모델(LightGBM/XGBoost) 전용 피처 전처리기"""

    def __init__(self):
        super().__init__()
        # 트리 모델은 정규화가 불필요하므로 scaler 제거
        self.scaler = None

    def fit_transform(self, df):
        """트리 모델에 최적화된 피처 전처리"""
        print("🌳 Processing features for Tree-based models (LightGBM/XGBoost)...")

        # 범주형 피처 정의
        self.categorical_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]

        # 수치형 피처 정의
        self.numerical_features = [col for col in df.columns
                                 if col not in EXCLUDE_COLUMNS
                                 and col not in self.categorical_features]

        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        print(f"Numerical features ({len(self.numerical_features)}): {len(self.numerical_features)}")

        # 범주형 피처 인코딩
        for col in self.categorical_features:
            df[col] = df[col].fillna('unknown').astype(str)

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.categorical_encoders[col] = le

            self.feature_dims[col] = len(le.classes_)
            print(f"  {col}: {self.feature_dims[col]} categories")

        # 수치형 피처 전처리 (트리 최적화)
        df = self.preprocess_numerical_features_for_trees(df)

        # 시퀀스 피처 처리 (NVIDIA Merlin 방식: 완전 제거)
        if 'seq' in df.columns:
            print("🚀 Excluding sequence column (following NVIDIA Merlin approach)")
            df = df.drop('seq', axis=1)

        return df

    def preprocess_numerical_features_for_trees(self, df):
        """트리 모델에 최적화된 수치형 피처 전처리 (정규화 없음)"""
        print("🌳 Applying tree-optimized numerical preprocessing (NO scaling)...")

        # 1. 결측치 처리 + 이상치 클리핑만 수행 (정규화 생략)
        for col in self.numerical_features:
            if col in df.columns:
                df = self._apply_numerical_preprocessing_step(df, col, is_training=True)

        # 2. 정규화 생략 - 트리 모델은 원본 값의 분포가 중요
        print(f"  Skipped scaling for {len(self.numerical_features)} numerical features (tree-optimized)")

        # 3. float32로 변환 (메모리 최적화)
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        return df

    def transform(self, df):
        """추론용 변환 (트리 모델 최적화)"""
        # 범주형 피처 변환
        for col in self.categorical_features:
            df[col] = df[col].fillna('unknown').astype(str)

            # 새로운 카테고리는 'unknown'으로 처리
            unknown_mask = ~df[col].isin(self.categorical_encoders[col].classes_)
            df.loc[unknown_mask, col] = 'unknown'

            df[col] = self.categorical_encoders[col].transform(df[col])

        # 수치형 피처 처리 (정규화 없음)
        original_numerical = [col for col in self.numerical_features
                             if col in df.columns and not col.startswith('seq_')]
        if original_numerical:
            for col in original_numerical:
                df = self._apply_numerical_preprocessing_step(df, col, is_training=False)

        # 시퀀스 피처 처리 (NVIDIA Merlin 방식: 완전 제거)
        if 'seq' in df.columns:
            df = df.drop('seq', axis=1)

        return df


class InferenceFeatureProcessor(BaseFeatureProcessor):
    """추론용 피처 전처리기"""

    def transform(self, df):
        """기존 fit된 정보로 변환 (추론용)"""
        # 범주형 피처 변환
        for col in self.categorical_features:
            df[col] = df[col].fillna('unknown').astype(str)

            # 새로운 카테고리는 'unknown'으로 처리
            unknown_mask = ~df[col].isin(self.categorical_encoders[col].classes_)
            df.loc[unknown_mask, col] = 'unknown'

            df[col] = self.categorical_encoders[col].transform(df[col])

        # 기존 수치형 피처만 먼저 처리 (스케일러 적용 전)
        original_numerical = [col for col in self.numerical_features
                             if col in df.columns and not col.startswith('seq_')]
        if original_numerical:
            df = self.preprocess_original_numerical_features_inference(df, original_numerical)

        # 시퀀스 피처 처리 (스케일러 적용 후, 메모리 최적화)
        if 'seq' in df.columns:
            seq_features = self.process_sequence_features(df['seq'])

            # 원본 seq 컬럼 제거 (메모리 절약)
            df = df.drop('seq', axis=1)

            # 새로운 시퀀스 피처들을 후처리
            seq_cols = list(seq_features.columns)
            for col in seq_cols:
                if col in self.numerical_stats:
                    median_val = self.numerical_stats[col]['median']
                    seq_features[col] = seq_features[col].fillna(median_val)

            # 처리된 시퀀스 피처 추가
            df = pd.concat([df, seq_features], axis=1)

            # seq_features 메모리 해제
            del seq_features

        return df

    def preprocess_original_numerical_features_inference(self, df, original_numerical):
        """추론용 기존 수치형 피처 전처리 (시퀀스 피처 제외)"""
        print("🔧 Applying improved numerical preprocessing for inference...")

        # 1. 각 피처별 전처리 (통합된 메서드 사용)
        for col in original_numerical:
            df = self._apply_numerical_preprocessing_step(df, col, is_training=False)

        # 2. 훈련된 스케일러 적용 (기존 피처만)
        df[original_numerical] = self.scaler.transform(df[original_numerical])
        print(f"  Applied trained QuantileTransformer to {len(original_numerical)} original numerical features")

        # 3. float32로 변환
        df[original_numerical] = df[original_numerical].astype(np.float32)

        return df