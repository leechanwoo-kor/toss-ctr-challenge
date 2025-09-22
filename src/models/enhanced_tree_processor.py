
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from ..config import CATEGORICAL_FEATURES
from ..utils.config import EXCLUDE_COLUMNS
from .feature_processor import TreeBasedFeatureProcessor


class EnhancedTreeFeatureProcessor(TreeBasedFeatureProcessor):

    def __init__(self):
        super().__init__()
        self.inventory_popularity = {}
        self.age_inventory_affinity = {}
        self.gender_inventory_affinity = {}
        self.new_features = []

        self.constant_features_to_remove = ['l_feat_20', 'l_feat_23']
        self.duplicate_features_to_remove = ['l_feat_16', 'l_feat_17']
        self.statistical_insignificant_features = ['l_feat_24']
        self.low_importance_features = ['day_of_week']
        self.remove_feat_a_group = False

        if 'day_of_week' in self.categorical_features:
            self.categorical_features.remove('day_of_week')

    def fit_transform(self, df):
        print("🚀 Processing features with EDA-based enhancements...")
        df = self._create_eda_features(df.copy(), is_training=True)
        df = super().fit_transform(df)
        df = self._remove_constant_features(df)
        return df

    def transform(self, df):
        print("[INFO] Transforming test data with EDA features...")
        df = super().transform(df)
        df = self._create_eda_features(df, is_training=False)
        df = self._remove_constant_features(df)
        return df

    def _remove_constant_features(self, df):
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

        if self.remove_feat_a_group:
            feat_a_features = [col for col in df.columns if col.startswith('feat_a_')]
            df = df.drop(columns=feat_a_features)
            removed_features.extend(feat_a_features)
            if feat_a_features:
                print(f"[INFO] Removed feat_a group: {len(feat_a_features)} features")

        if removed_features:
            print(f"[INFO] Removed features: {removed_features}")
        return df

    def _create_eda_features(self, df, is_training=True):
        print(f"[INFO] Creating EDA-based features (training={is_training})...")

        has_gender = 'gender' in df.columns
        has_age_group = 'age_group' in df.columns
        has_inventory = 'inventory_id' in df.columns
        has_hour = 'hour' in df.columns
        has_clicked = 'clicked' in df.columns

        new_feature_count = 0
        new_columns = {}

        if has_inventory:
            if is_training and has_clicked:
                print("  🏪 Computing inventory popularity (count-based, no leakage)...")
                inventory_counts = df.groupby('inventory_id').size()
                self.inventory_popularity = inventory_counts.rank(ascending=False).to_dict()
                print(f"     Created {len(self.inventory_popularity)} inventory popularity ranks")

            new_columns['inventory_popularity_rank'] = df['inventory_id'].map(self.inventory_popularity).fillna(len(self.inventory_popularity)/2).astype(np.float32)
            new_feature_count += 1
            print(f"     [SUCCESS] inventory_popularity_rank: mean={new_columns['inventory_popularity_rank'].mean():.1f}")


        if has_gender and has_age_group:
            gender_age_combo = df['gender'].astype(str) + '_' + df['age_group'].astype(str)
            le_combo = LabelEncoder()
            new_columns['gender_age_combo'] = le_combo.fit_transform(gender_age_combo).astype(np.float32)
            new_feature_count += 1

        if has_age_group and has_inventory:
            if is_training:
                age_inventory_counts = df.groupby(['age_group', 'inventory_id']).size()
                self.age_inventory_affinity = age_inventory_counts.groupby('age_group').rank(ascending=False).to_dict()

            age_inventory_key = list(zip(df['age_group'], df['inventory_id']))
            new_columns['age_inventory_affinity'] = pd.Series(age_inventory_key).map(
                self.age_inventory_affinity).fillna(len(df['inventory_id'].unique())/2).astype(np.float32)
            new_feature_count += 1

        if has_gender and has_inventory:
            if is_training:
                gender_inventory_counts = df.groupby(['gender', 'inventory_id']).size()
                self.gender_inventory_affinity = gender_inventory_counts.groupby('gender').rank(ascending=False).to_dict()

            gender_inventory_key = list(zip(df['gender'], df['inventory_id']))
            new_columns['gender_inventory_affinity'] = pd.Series(gender_inventory_key).map(
                self.gender_inventory_affinity).fillna(len(df['inventory_id'].unique())/2).astype(np.float32)
            new_feature_count += 1


        if new_columns:
            df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

        new_features = [
            'inventory_popularity_rank',
            'gender_age_combo',
            'age_inventory_affinity',
            'gender_inventory_affinity'
        ]

        for feat in new_features:
            if feat in df.columns and feat not in self.numerical_features:
                self.numerical_features.append(feat)
                self.new_features.append(feat)

        print(f"[SUCCESS] Created {new_feature_count} new EDA-based features!")
        print(f"[INFO] New features: {[f for f in new_features if f in df.columns]}")
        return df

    def get_feature_importance_groups(self):
        return {
            'original_categorical': [f for f in self.categorical_features],
            'original_numerical': [f for f in self.numerical_features if f not in self.new_features],
            'eda_user_features': [],
            'eda_inventory_features': ['inventory_popularity_rank', 'age_inventory_affinity', 'gender_inventory_affinity'],
            'eda_user_features': ['gender_age_combo'],
            'all_new_features': self.new_features
        }

    def transform_test(self, df):
        return self.transform(df)