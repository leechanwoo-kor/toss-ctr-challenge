"""
DeepFM용 피처 전처리 클래스들
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, RobustScaler

from ..utils.config import CATEGORICAL_FEATURES, EXCLUDE_COLUMNS


class BaseFeatureProcessor:
    """기본 피처 전처리기"""

    def __init__(self):
        self.categorical_encoders = {}
        self.categorical_features = []
        self.numerical_features = []
        self.feature_dims = {}
        self.scaler = RobustScaler()
        self.numerical_stats = {}

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

        # 시퀀스 피처 처리
        if 'seq' in df.columns:
            seq_features = self.process_sequence_features(df['seq'])
            df = pd.concat([df, seq_features], axis=1)

            # 새로운 시퀀스 피처들을 수치형에 추가
            new_seq_cols = [col for col in seq_features.columns]
            self.numerical_features.extend(new_seq_cols)
            print(f"Added sequence features: {new_seq_cols}")

        return df

    def preprocess_numerical_features(self, df):
        """개선된 수치형 피처 전처리 (훈련용)"""
        print("🔧 Applying improved numerical preprocessing...")

        # 1. 통계적 결측치 대체
        for col in self.numerical_features:
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df) * 100

                if missing_pct > 0:
                    median_val = df[col].median()
                    self.numerical_stats[col] = {'median': median_val, 'missing_pct': missing_pct}
                    df[col] = df[col].fillna(median_val)

                    if missing_pct > 5:
                        print(f"  {col}: {missing_pct:.2f}% missing → filled with median {median_val:.4f}")

                # 2. 극단 이상치 클리핑
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)

                outlier_mask = (df[col] > q99) | (df[col] < q01)
                outlier_pct = outlier_mask.sum() / len(df) * 100

                if outlier_pct > 1:
                    df[col] = np.clip(df[col], q01, q99)
                    print(f"  {col}: {outlier_pct:.2f}% outliers clipped [{q01:.4f}, {q99:.4f}]")

        # 3. RobustScaler 적용
        df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])
        print(f"  Applied RobustScaler to {len(self.numerical_features)} numerical features")

        # float32로 변환
        df[self.numerical_features] = df[self.numerical_features].astype(np.float32)

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

        # 시퀀스 피처 처리 (스케일러 적용 후)
        if 'seq' in df.columns:
            seq_features = self.process_sequence_features(df['seq'])
            df = pd.concat([df, seq_features], axis=1)

            # 새로운 시퀀스 피처들을 후처리
            seq_cols = [col for col in seq_features.columns]
            for col in seq_cols:
                if col in self.numerical_stats:
                    median_val = self.numerical_stats[col]['median']
                    df[col] = df[col].fillna(median_val)

        return df

    def preprocess_original_numerical_features_inference(self, df, original_numerical):
        """추론용 기존 수치형 피처 전처리 (시퀀스 피처 제외)"""
        print("🔧 Applying improved numerical preprocessing for inference...")

        # 1. 훈련 시 저장된 통계로 결측치 대체
        for col in original_numerical:
            if col in self.numerical_stats:
                median_val = self.numerical_stats[col]['median']
                missing_before = df[col].isnull().sum()
                df[col] = df[col].fillna(median_val)
                if missing_before > 0:
                    print(f"  {col}: {missing_before} missing → filled with training median {median_val:.4f}")
            else:
                df[col] = df[col].fillna(df[col].median())

            # 2. 극단값 클리핑
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            outlier_mask = (df[col] > q99) | (df[col] < q01)
            if outlier_mask.sum() > 0:
                df[col] = np.clip(df[col], q01, q99)

        # 3. 훈련된 스케일러 적용 (기존 피처만)
        df[original_numerical] = self.scaler.transform(df[original_numerical])
        print(f"  Applied trained RobustScaler to {len(original_numerical)} original numerical features")

        # float32로 변환
        df[original_numerical] = df[original_numerical].astype(np.float32)

        return df