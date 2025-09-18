"""
EDA 기반 피처 엔지니어링이 추가된 트리 모델용 프로세서
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from ..utils.config import CATEGORICAL_FEATURES, EXCLUDE_COLUMNS
from .feature_processor import TreeBasedFeatureProcessor


class EnhancedTreeFeatureProcessor(TreeBasedFeatureProcessor):
    """EDA 인사이트 기반 피처 엔지니어링이 추가된 트리 모델용 프로세서"""

    def __init__(self):
        super().__init__()
        # 새로운 피처들을 위한 매핑 테이블들
        self.user_segment_ctr = {}
        self.inventory_ctr = {}
        self.hour_ctr = {}
        self.day_ctr = {}
        self.hour_day_ctr = {}
        self.inventory_popularity = {}
        self.new_features = []

    def fit_transform(self, df):
        """트리 모델에 최적화된 피처 전처리 + EDA 기반 새 피처들"""
        print("🚀 Processing features with EDA-based enhancements...")

        # EDA 피처 생성을 기본 전처리 전에 수행 (clicked 컬럼이 있을 때)
        df = self._create_eda_features(df.copy(), is_training=True)

        # 기본 전처리 수행 (clicked 컬럼 제거)
        df = super().fit_transform(df)

        return df

    def transform(self, df):
        """테스트 데이터 변환 (기존 매핑 테이블 사용)"""
        print("🔄 Transforming test data with EDA features...")

        # 기본 변환 먼저 수행
        df = super().transform(df)

        # 새로운 피처들 생성 (학습된 매핑 사용)
        df = self._create_eda_features(df, is_training=False)

        return df

    def _create_eda_features(self, df, is_training=True):
        """EDA에서 발견한 인사이트 기반 새 피처들 생성"""
        print(f"📊 Creating EDA-based features (training={is_training})...")

        # 필요한 컬럼들 확인
        has_gender = 'gender' in df.columns
        has_age_group = 'age_group' in df.columns
        has_inventory = 'inventory_id' in df.columns
        has_hour = 'hour' in df.columns
        has_day = 'day_of_week' in df.columns
        has_clicked = 'clicked' in df.columns

        new_feature_count = 0
        new_columns = {}  # DataFrame fragmentation 방지를 위한 딕셔너리

        # 1. 사용자 세그먼트 CTR (최고 임팩트 0.0536)
        if has_gender and has_age_group:
            if is_training and has_clicked:
                print("  📈 Computing user segment CTR...")
                print(f"     Data shape: {df.shape}, clicked column exists: {has_clicked}")
                print(f"     Overall CTR: {df['clicked'].mean():.6f}")

                self.user_segment_ctr = df.groupby(['gender', 'age_group'])['clicked'].mean().to_dict()
                print(f"     Created {len(self.user_segment_ctr)} user segment mappings")

                # 샘플 출력
                sample_ctrs = list(self.user_segment_ctr.values())[:5]
                print(f"     Sample CTRs: {sample_ctrs}")
            else:
                print(f"  ⚠️ Skipping CTR computation: is_training={is_training}, has_clicked={has_clicked}")

            # 매핑 적용 (벡터화된 방법)
            if len(self.user_segment_ctr) > 0:
                # 임시 DataFrame을 만들어 merge 사용
                segment_df = pd.DataFrame(list(self.user_segment_ctr.keys()),
                                         columns=['gender', 'age_group'])
                segment_df['user_segment_ctr'] = list(self.user_segment_ctr.values())

                # 타입 일치시키기
                test_df = df[['gender', 'age_group']].copy()
                test_df['gender'] = test_df['gender'].astype(str)
                test_df['age_group'] = test_df['age_group'].astype(str)
                segment_df['gender'] = segment_df['gender'].astype(str)
                segment_df['age_group'] = segment_df['age_group'].astype(str)

                merged = test_df.merge(
                    segment_df, on=['gender', 'age_group'], how='left'
                )
                new_columns['user_segment_ctr'] = merged['user_segment_ctr'].fillna(0.019).astype(np.float32)
            else:
                # 매핑 테이블이 없으면 기본값 사용
                new_columns['user_segment_ctr'] = np.full(len(df), 0.019, dtype=np.float32)
            new_feature_count += 1
            print(f"     ✅ user_segment_ctr: mean={new_columns['user_segment_ctr'].mean():.6f}")

        # 2. 상품별 평균 CTR (0.0523 차이)
        if has_inventory:
            if is_training and has_clicked:
                print("  🏪 Computing inventory CTR...")
                self.inventory_ctr = df.groupby('inventory_id')['clicked'].mean().to_dict()
                print(f"     Created {len(self.inventory_ctr)} inventory CTR mappings")

                # 상품 인기도 (노출 횟수)
                inventory_counts = df.groupby('inventory_id').size()
                self.inventory_popularity = inventory_counts.rank(ascending=False).to_dict()
                print(f"     Created {len(self.inventory_popularity)} inventory popularity ranks")

            # 매핑 적용
            new_columns['inventory_ctr'] = df['inventory_id'].map(self.inventory_ctr).fillna(0.019).astype(np.float32)
            new_columns['inventory_popularity_rank'] = df['inventory_id'].map(self.inventory_popularity).fillna(len(self.inventory_popularity)/2).astype(np.float32)
            new_feature_count += 2
            print(f"     ✅ inventory_ctr: mean={new_columns['inventory_ctr'].mean():.6f}")
            print(f"     ✅ inventory_popularity_rank: mean={new_columns['inventory_popularity_rank'].mean():.1f}")

        # 3. 시간대별 CTR (0.0061 차이)
        if has_hour:
            if is_training and has_clicked:
                print("  ⏰ Computing hourly CTR...")
                self.hour_ctr = df.groupby('hour')['clicked'].mean().to_dict()
                print(f"     Created {len(self.hour_ctr)} hourly CTR mappings")

            # 매핑 적용
            new_columns['hour_ctr'] = df['hour'].map(self.hour_ctr).fillna(0.019).astype(np.float32)

            # 피크 시간대 여부 (새벽 2-4시) - 문자열 처리
            new_columns['is_peak_hour'] = df['hour'].isin(['02', '03', '04']).astype(np.int8)
            new_feature_count += 2
            print(f"     ✅ hour_ctr: mean={new_columns['hour_ctr'].mean():.6f}")
            print(f"     ✅ is_peak_hour: {new_columns['is_peak_hour'].sum()}/{len(df)} samples")

        # 4. 요일별 CTR (0.0042 차이)
        if has_day:
            if is_training and has_clicked:
                print("  📅 Computing daily CTR...")
                self.day_ctr = df.groupby('day_of_week')['clicked'].mean().to_dict()
                print(f"     Created {len(self.day_ctr)} daily CTR mappings")

            # 매핑 적용
            new_columns['day_ctr'] = df['day_of_week'].map(self.day_ctr).fillna(0.019).astype(np.float32)

            # 주말 여부 (토/일: 6,7) - 문자열 처리
            new_columns['is_weekend'] = df['day_of_week'].isin(['6', '7']).astype(np.int8)
            new_feature_count += 2
            print(f"     ✅ day_ctr: mean={new_columns['day_ctr'].mean():.6f}")
            print(f"     ✅ is_weekend: {new_columns['is_weekend'].sum()}/{len(df)} samples")

        # 5. 시간×요일 교차 CTR
        if has_hour and has_day:
            if is_training and has_clicked:
                print("  🕐📅 Computing hour×day CTR...")
                self.hour_day_ctr = df.groupby(['hour', 'day_of_week'])['clicked'].mean().to_dict()
                print(f"     Created {len(self.hour_day_ctr)} hour×day CTR mappings")

            # 매핑 적용 (벡터화된 방법)
            hour_day_df = pd.DataFrame(list(self.hour_day_ctr.keys()),
                                      columns=['hour', 'day_of_week'])
            hour_day_df['hour_day_ctr'] = list(self.hour_day_ctr.values())

            # 타입 일치시키기
            test_hd_df = df[['hour', 'day_of_week']].copy()
            test_hd_df['hour'] = test_hd_df['hour'].astype(str)
            test_hd_df['day_of_week'] = test_hd_df['day_of_week'].astype(str)
            hour_day_df['hour'] = hour_day_df['hour'].astype(str)
            hour_day_df['day_of_week'] = hour_day_df['day_of_week'].astype(str)

            merged = test_hd_df.merge(
                hour_day_df, on=['hour', 'day_of_week'], how='left'
            )
            new_columns['hour_day_ctr'] = merged['hour_day_ctr'].fillna(0.019).astype(np.float32)
            new_feature_count += 1
            print(f"     ✅ hour_day_ctr: mean={new_columns['hour_day_ctr'].mean():.6f}")

        # 6. 상품 CTR 분위수 (고급 피처)
        if has_inventory and 'inventory_ctr' in new_columns:
            try:
                # 유니크한 값들이 충분한지 확인
                unique_values = new_columns['inventory_ctr'].nunique()
                if unique_values >= 10:
                    new_columns['inventory_ctr_quantile'] = pd.qcut(
                        new_columns['inventory_ctr'],
                        q=10,
                        labels=range(10),
                        duplicates='drop'
                    ).astype('category').cat.codes.astype(np.int8)
                    new_feature_count += 1
                    print(f"     ✅ inventory_ctr_quantile: 10 quantiles from {unique_values} unique values")
                else:
                    # 유니크 값이 적으면 rank 기반으로 변환
                    new_columns['inventory_ctr_quantile'] = (new_columns['inventory_ctr'].rank(method='first') / len(df) * 10).astype(np.int8)
                    new_columns['inventory_ctr_quantile'] = np.clip(new_columns['inventory_ctr_quantile'], 0, 9)
                    new_feature_count += 1
                    print(f"     ✅ inventory_ctr_quantile: rank-based quantiles ({unique_values} unique values)")
            except (ValueError, TypeError) as e:
                # qcut 실패시 rank 기반으로 대체
                print(f"     ⚠️ qcut failed, using rank-based quantiles: {e}")
                new_columns['inventory_ctr_quantile'] = (new_columns['inventory_ctr'].rank(method='first') / len(df) * 10).astype(np.int8)
                new_columns['inventory_ctr_quantile'] = np.clip(new_columns['inventory_ctr_quantile'], 0, 9)
                new_feature_count += 1
                print(f"     ✅ inventory_ctr_quantile: rank-based quantiles")

        # 새 컬럼들을 한 번에 DataFrame에 추가 (fragmentation 방지)
        if new_columns:
            new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
            df = new_df

        # 새 피처들을 numerical_features에 추가
        new_features = [
            'user_segment_ctr', 'inventory_ctr', 'inventory_popularity_rank',
            'hour_ctr', 'is_peak_hour', 'day_ctr', 'is_weekend',
            'hour_day_ctr', 'inventory_ctr_quantile'
        ]

        # 실제로 생성된 피처들만 추가
        for feat in new_features:
            if feat in df.columns and feat not in self.numerical_features:
                self.numerical_features.append(feat)
                self.new_features.append(feat)

        print(f"🎉 Created {new_feature_count} new EDA-based features!")
        print(f"📊 New features: {[f for f in new_features if f in df.columns]}")

        return df

    def get_feature_importance_groups(self):
        """피처들을 그룹별로 분류하여 중요도 분석용"""
        return {
            'original_categorical': [f for f in self.categorical_features],
            'original_numerical': [f for f in self.numerical_features if f not in self.new_features],
            'eda_user_features': ['user_segment_ctr'],
            'eda_inventory_features': ['inventory_ctr', 'inventory_popularity_rank', 'inventory_ctr_quantile'],
            'eda_temporal_features': ['hour_ctr', 'is_peak_hour', 'day_ctr', 'is_weekend', 'hour_day_ctr'],
            'all_new_features': self.new_features
        }

    def transform_test(self, df):
        """테스트 데이터 전용 변환 메서드 (transform과 동일하지만 명시적)"""
        return self.transform(df)