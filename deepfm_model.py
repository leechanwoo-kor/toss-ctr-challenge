"""
DeepFM Model for CTR Prediction
메모리 효율적이고 CTR 예측에 최적화된 DeepFM 구현
"""

import pandas as pd
import numpy as np
import os
import random
import gc
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from datetime import datetime

# DeepFM Configuration
DEEPFM_CFG = {
    'SAMPLE_SIZE': None,         # 전체 데이터 사용 (10.7M)
    'BATCH_SIZE': 1024,
    'EPOCHS': 10,
    'LEARNING_RATE': 1e-3,
    'EMBEDDING_DIM': 16,         # 임베딩 차원
    'HIDDEN_UNITS': [512, 256, 128],
    'DROPOUT': 0.2,
    'SEED': 42,
    'EARLY_STOPPING': 5,
}

# WandB Configuration
WANDB_CFG = {
    'PROJECT': 'toss-ctr-deepfm',
    'ENTITY': None,  # 개인 계정 사용
    'TAGS': ['deepfm', 'improved-preprocessing', 'full-dataset'],
    'NOTES': 'DeepFM with improved preprocessing: median imputation, outlier clipping, robust scaling'
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(DEEPFM_CFG['SEED'])

def calculate_competition_metrics(y_true, y_pred):
    """경쟁 지표 계산"""
    ap = average_precision_score(y_true, y_pred)

    eps = 1e-15
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)

    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)

    ll_0 = -np.mean(np.log(1 - y_pred_clip[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred_clip[mask_1])) if mask_1.sum() > 0 else 0
    wll = 0.5 * ll_0 + 0.5 * ll_1

    comp_score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    auc = roc_auc_score(y_true, y_pred)

    return comp_score, ap, wll, auc

class FeatureProcessor:
    """DeepFM용 피처 전처리기 (개선된 전처리 포함)"""
    def __init__(self):
        self.categorical_encoders = {}
        self.categorical_features = []
        self.numerical_features = []
        self.feature_dims = {}
        self.scaler = RobustScaler()  # 이상치에 강건한 스케일러
        self.numerical_stats = {}    # 통계적 대체값 저장

    def fit_transform(self, df):
        """피처 전처리 및 변환"""
        print("🔧 Processing features for DeepFM...")

        # 범주형 피처 정의 (카디널리티가 높은 것들)
        categorical_candidates = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']

        # 실제 존재하는 범주형 피처만 선택
        self.categorical_features = [col for col in categorical_candidates if col in df.columns]

        # 나머지는 수치형으로 처리
        exclude_cols = {'clicked', 'seq', 'ID'}
        self.numerical_features = [col for col in df.columns
                                 if col not in exclude_cols
                                 and col not in self.categorical_features]

        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        print(f"Numerical features ({len(self.numerical_features)}): {len(self.numerical_features)}")

        # 범주형 피처 인코딩
        for col in self.categorical_features:
            # 결측값을 'unknown'으로 처리
            df[col] = df[col].fillna('unknown').astype(str)

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.categorical_encoders[col] = le

            # 피처 차원 저장 (임베딩용)
            self.feature_dims[col] = len(le.classes_)
            print(f"  {col}: {self.feature_dims[col]} categories")

        # 수치형 피처 전처리 (개선된 방식)
        df = self.preprocess_numerical_features(df)

        # 시퀀스 피처 처리 (통계적 변환)
        if 'seq' in df.columns:
            seq_features = self.process_sequence_features(df['seq'])
            df = pd.concat([df, seq_features], axis=1)

            # 새로운 시퀀스 피처들을 수치형에 추가
            new_seq_cols = [col for col in seq_features.columns]
            self.numerical_features.extend(new_seq_cols)
            print(f"Added sequence features: {new_seq_cols}")

        return df

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

    def preprocess_numerical_features(self, df):
        """개선된 수치형 피처 전처리"""
        print("🔧 Applying improved numerical preprocessing...")

        # 1. 통계적 결측치 대체
        for col in self.numerical_features:
            if col in df.columns:
                # 결측치 비율 확인
                missing_pct = df[col].isnull().sum() / len(df) * 100

                if missing_pct > 0:
                    # 중앙값으로 대체 (0 대신)
                    median_val = df[col].median()
                    self.numerical_stats[col] = {'median': median_val, 'missing_pct': missing_pct}
                    df[col] = df[col].fillna(median_val)

                    if missing_pct > 5:
                        print(f"  {col}: {missing_pct:.2f}% missing → filled with median {median_val:.4f}")

                # 2. 극단 이상치 클리핑 (99th percentile)
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)

                outlier_mask = (df[col] > q99) | (df[col] < q01)
                outlier_pct = outlier_mask.sum() / len(df) * 100

                if outlier_pct > 1:  # 1% 이상 이상치가 있으면 클리핑
                    df[col] = np.clip(df[col], q01, q99)
                    print(f"  {col}: {outlier_pct:.2f}% outliers clipped [{q01:.4f}, {q99:.4f}]")

        # 3. RobustScaler 적용
        df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])
        print(f"  Applied RobustScaler to {len(self.numerical_features)} numerical features")

        # float32로 변환
        df[self.numerical_features] = df[self.numerical_features].astype(np.float32)

        return df

    def transform(self, df):
        """기존 fit된 정보로 변환"""
        # 범주형 피처 변환
        for col in self.categorical_features:
            df[col] = df[col].fillna('unknown').astype(str)

            # 새로운 카테고리는 'unknown'으로 처리
            unknown_mask = ~df[col].isin(self.categorical_encoders[col].classes_)
            df.loc[unknown_mask, col] = 'unknown'

            df[col] = self.categorical_encoders[col].transform(df[col])

        # 시퀀스 피처 처리 (수치형 피처 처리 전에)
        if 'seq' in df.columns:
            seq_features = self.process_sequence_features(df['seq'])
            df = pd.concat([df, seq_features], axis=1)

        # 실제 존재하는 수치형 피처만 처리 (개선된 방식)
        existing_numerical = [col for col in self.numerical_features if col in df.columns]
        if existing_numerical:
            # 훈련 시 저장된 통계 사용
            for col in existing_numerical:
                if col in self.numerical_stats:
                    median_val = self.numerical_stats[col]['median']
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna(df[col].median())

                # 같은 클리핑 범위 적용 (필요시)
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = np.clip(df[col], q01, q99)

            # 스케일러 적용
            df[existing_numerical] = self.scaler.transform(df[existing_numerical])
            df[existing_numerical] = df[existing_numerical].astype(np.float32)

        return df

class DeepFMDataset(Dataset):
    """DeepFM용 데이터셋"""
    def __init__(self, df, processor, target_col=None, has_target=True):
        self.processor = processor
        self.has_target = has_target
        self.target_col = target_col

        # 범주형 피처 (임베딩용)
        self.categorical_data = {}
        for col in processor.categorical_features:
            self.categorical_data[col] = torch.LongTensor(df[col].values)

        # 수치형 피처
        self.numerical_data = torch.FloatTensor(df[processor.numerical_features].values)

        if has_target:
            self.targets = torch.FloatTensor(df[target_col].values)

        print(f"Dataset created: {len(df)} samples")
        print(f"Categorical features shape: {[(k, v.shape) for k, v in self.categorical_data.items()]}")
        print(f"Numerical features shape: {self.numerical_data.shape}")

    def __len__(self):
        return len(self.numerical_data)

    def __getitem__(self, idx):
        categorical_vals = {k: v[idx] for k, v in self.categorical_data.items()}
        numerical_vals = self.numerical_data[idx]

        if self.has_target:
            target = self.targets[idx]
            return categorical_vals, numerical_vals, target
        else:
            return categorical_vals, numerical_vals

class DeepFM(nn.Module):
    """DeepFM 모델 구현"""
    def __init__(self, feature_dims, embedding_dim=16, hidden_units=[512, 256, 128], dropout=0.2, num_numerical=100):
        super(DeepFM, self).__init__()

        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        self.num_numerical = num_numerical

        # === FM Part ===
        # 범주형 피처용 임베딩 (1차 + 2차)
        self.fm_embeddings = nn.ModuleDict()
        self.fm_bias = nn.ModuleDict()

        for feature, dim in feature_dims.items():
            # 1차 임베딩 (bias)
            self.fm_bias[feature] = nn.Embedding(dim, 1)
            # 2차 임베딩 (interaction)
            self.fm_embeddings[feature] = nn.Embedding(dim, embedding_dim)

        # 수치형 피처용 선형 변환
        self.numerical_linear = nn.Linear(num_numerical, 1)
        self.numerical_embedding = nn.Linear(num_numerical, embedding_dim)

        # === Deep Part ===
        # 임베딩 결합 후 MLP
        total_embedding_dim = len(feature_dims) * embedding_dim + embedding_dim  # 범주형 + 수치형

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

        # 최종 출력 조합
        self.output_layer = nn.Linear(2, 1)  # FM + Deep

        print(f"🧠 DeepFM initialized:")
        print(f"  Feature dims: {feature_dims}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Numerical features: {num_numerical}")
        print(f"  Deep layers: {hidden_units}")

    def forward(self, categorical_inputs, numerical_inputs):
        batch_size = numerical_inputs.size(0)

        # === FM Part ===
        # 1차 항 (선형)
        fm_first_order = self.numerical_linear(numerical_inputs)  # 수치형 1차

        for feature, values in categorical_inputs.items():
            fm_first_order += self.fm_bias[feature](values)  # 범주형 1차

        # 2차 항 (교차)
        embeddings = []

        # 범주형 임베딩
        for feature, values in categorical_inputs.items():
            emb = self.fm_embeddings[feature](values)  # (batch, embedding_dim)
            embeddings.append(emb)

        # 수치형 임베딩
        numerical_emb = self.numerical_embedding(numerical_inputs)  # (batch, embedding_dim)
        embeddings.append(numerical_emb)

        # 임베딩 스택
        embeddings = torch.stack(embeddings, dim=1)  # (batch, num_features, embedding_dim)

        # 2차 교차항 계산
        sum_square = torch.sum(embeddings, dim=1) ** 2  # (batch, embedding_dim)
        square_sum = torch.sum(embeddings ** 2, dim=1)  # (batch, embedding_dim)
        fm_second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # (batch, 1)

        fm_output = fm_first_order + fm_second_order

        # === Deep Part ===
        # 모든 임베딩을 평탄화해서 MLP 입력으로
        deep_input = embeddings.view(batch_size, -1)  # (batch, num_features * embedding_dim)
        deep_output = self.deep_layers(deep_input)

        # === 최종 결합 ===
        combined = torch.cat([fm_output, deep_output], dim=1)
        output = self.output_layer(combined)

        return output.squeeze(1)

def train_deepfm():
    """DeepFM 훈련"""

    # WandB 초기화
    run_name = f"deepfm-improved-{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project=WANDB_CFG['PROJECT'],
        entity=WANDB_CFG['ENTITY'],
        name=run_name,
        tags=WANDB_CFG['TAGS'],
        notes=WANDB_CFG['NOTES'],
        config=DEEPFM_CFG
    )

    print("🎯 Starting DeepFM Training for CTR Prediction")
    print("="*60)
    print(f"📊 WandB Run: {run_name}")

    # 데이터 로드 (샘플링)
    print("📊 Loading and sampling data...")
    all_train = pd.read_parquet("data/train.parquet", engine="pyarrow")

    print(f"Original train size: {len(all_train):,}")

    # 전체 데이터 사용
    sample_size = DEEPFM_CFG['SAMPLE_SIZE']

    if sample_size is None:
        # 전체 데이터 사용
        train_sample = all_train.copy()
        print("🚀 Using FULL dataset!")
    else:
        # 기존 샘플링 로직
        clicked_1 = all_train[all_train['clicked'] == 1]
        clicked_0 = all_train[all_train['clicked'] == 0]

        pos_samples = len(clicked_1)
        neg_samples = min(sample_size - pos_samples, len(clicked_0))

        if neg_samples > 0:
            sampled_neg = clicked_0.sample(n=neg_samples, random_state=DEEPFM_CFG['SEED'])
            train_sample = pd.concat([clicked_1, sampled_neg], axis=0).sample(frac=1, random_state=DEEPFM_CFG['SEED']).reset_index(drop=True)
        else:
            train_sample = all_train.sample(n=sample_size, random_state=DEEPFM_CFG['SEED']).reset_index(drop=True)

        del clicked_1, clicked_0

    print(f"Train size: {len(train_sample):,}")
    print(f"Class distribution: {train_sample['clicked'].value_counts().to_dict()}")

    del all_train
    gc.collect()

    # 피처 전처리
    processor = FeatureProcessor()
    train_processed = processor.fit_transform(train_sample.copy())

    print(f"Processed features: {len(processor.numerical_features)} numerical, {len(processor.categorical_features)} categorical")

    # Train/Validation 분할
    train_df, val_df = train_test_split(
        train_processed,
        test_size=0.15,
        random_state=DEEPFM_CFG['SEED'],
        stratify=train_processed['clicked']
    )

    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}")

    # 데이터셋 생성 (훈련용만)
    train_dataset = DeepFMDataset(train_df, processor, target_col='clicked', has_target=True)
    val_dataset = DeepFMDataset(val_df, processor, target_col='clicked', has_target=True)

    # 데이터로더 (훈련용만)
    train_loader = DataLoader(train_dataset, batch_size=DEEPFM_CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=DEEPFM_CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    # 모델 초기화
    model = DeepFM(
        feature_dims=processor.feature_dims,
        embedding_dim=DEEPFM_CFG['EMBEDDING_DIM'],
        hidden_units=DEEPFM_CFG['HIDDEN_UNITS'],
        dropout=DEEPFM_CFG['DROPOUT'],
        num_numerical=len(processor.numerical_features)
    ).to(device)

    # 클래스 가중치
    pos_weight = (train_df['clicked'] == 0).sum() / (train_df['clicked'] == 1).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEEPFM_CFG['LEARNING_RATE'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    print(f"\n🏋️ Training with class weight: {pos_weight:.2f}")
    print("Epoch | Train Loss | Val Loss | Comp Score | AP     | WLL    | AUC    | LR")
    print("-" * 80)

    best_score = 0
    patience_counter = 0

    for epoch in range(1, DEEPFM_CFG['EPOCHS'] + 1):
        # === 훈련 ===
        model.train()
        train_loss = 0.0

        for categorical_inputs, numerical_inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            categorical_inputs = {k: v.to(device) for k, v in categorical_inputs.items()}
            numerical_inputs = numerical_inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(categorical_inputs, numerical_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # === 검증 ===
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for categorical_inputs, numerical_inputs, targets in val_loader:
                categorical_inputs = {k: v.to(device) for k, v in categorical_inputs.items()}
                numerical_inputs = numerical_inputs.to(device)
                targets = targets.to(device)

                outputs = model(categorical_inputs, numerical_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)

        # 성능 지표
        comp_score, ap, wll, auc = calculate_competition_metrics(
            np.array(val_targets), np.array(val_preds)
        )

        current_lr = optimizer.param_groups[0]['lr']

        # WandB 로깅
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'comp_score': comp_score,
            'ap': ap,
            'wll': wll if not np.isinf(wll) else None,  # inf 값 처리
            'auc': auc,
            'learning_rate': current_lr,
            'best_score': best_score
        })

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f} | "
              f"{comp_score:10.4f} | {ap:.4f} | {wll:.4f} | {auc:.4f} | {current_lr:.2e}")

        # 스케줄러 업데이트
        scheduler.step(val_loss)

        # 최고 모델 저장
        if comp_score > best_score:
            best_score = comp_score
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'processor': processor,
                'score': best_score,
                'epoch': epoch
            }, 'deepfm_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= DEEPFM_CFG['EARLY_STOPPING']:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\n✅ Training completed! Best score: {best_score:.6f}")
    print(f"📁 Model saved: deepfm_best_model.pth")
    print(f"🔮 For prediction, run: python deepfm_predict.py")

    # WandB 종료
    wandb.finish()

    return best_score

if __name__ == "__main__":
    try:
        best_score = train_deepfm()
        print(f"\n🏆 Final DeepFM score: {best_score:.6f}")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()