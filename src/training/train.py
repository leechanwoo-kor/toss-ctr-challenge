"""
DeepFM 훈련 로직
"""

import pandas as pd
import numpy as np
import os
import random
import gc
import time
from datetime import datetime

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..models.deepfm_core import DeepFM, DeepFMDataset
from ..models.feature_processor import TrainingFeatureProcessor
from ..utils.metrics import calculate_competition_metrics
from ..utils.config import DEEPFM_CONFIG, WANDB_CONFIG


def seed_everything(seed):
    """시드 고정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeepFMTrainer:
    """DeepFM 훈련 클래스"""

    def __init__(self, config=None, wandb_config=None):
        self.config = config or DEEPFM_CONFIG
        self.wandb_config = wandb_config or WANDB_CONFIG
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 시드 고정
        seed_everything(self.config['SEED'])

        print(f"🚀 Using device: {self.device}")

    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        print("📊 Loading and sampling data...")
        all_train = pd.read_parquet("data/train.parquet", engine="pyarrow")

        print(f"Original train size: {len(all_train):,}")

        # 샘플링 로직
        sample_size = self.config['SAMPLE_SIZE']

        if sample_size is None:
            train_sample = all_train.copy()
            print("🚀 Using FULL dataset!")
        else:
            clicked_1 = all_train[all_train['clicked'] == 1]
            clicked_0 = all_train[all_train['clicked'] == 0]

            pos_samples = len(clicked_1)
            neg_samples = min(sample_size - pos_samples, len(clicked_0))

            if neg_samples > 0:
                sampled_neg = clicked_0.sample(n=neg_samples, random_state=self.config['SEED'])
                train_sample = pd.concat([clicked_1, sampled_neg], axis=0).sample(
                    frac=1, random_state=self.config['SEED']).reset_index(drop=True)
            else:
                train_sample = all_train.sample(n=sample_size, random_state=self.config['SEED']).reset_index(drop=True)

            del clicked_1, clicked_0

        print(f"Train size: {len(train_sample):,}")
        print(f"Class distribution: {train_sample['clicked'].value_counts().to_dict()}")

        del all_train
        gc.collect()

        return train_sample

    def prepare_features(self, train_sample):
        """피처 전처리"""
        processor = TrainingFeatureProcessor()
        train_processed = processor.fit_transform(train_sample.copy())

        print(f"Processed features: {len(processor.numerical_features)} numerical, {len(processor.categorical_features)} categorical")

        return train_processed, processor

    def create_datasets(self, train_processed, processor):
        """데이터셋 생성"""
        # Train/Validation 분할
        train_df, val_df = train_test_split(
            train_processed,
            test_size=0.15,
            random_state=self.config['SEED'],
            stratify=train_processed['clicked']
        )

        print(f"Train: {len(train_df):,}, Val: {len(val_df):,}")

        # 데이터셋 생성
        train_dataset = DeepFMDataset(train_df, processor, 'clicked', True)
        val_dataset = DeepFMDataset(val_df, processor, 'clicked', True)

        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        return train_loader, val_loader, train_df, val_df

    def create_model(self, processor):
        """모델 생성"""
        model = DeepFM(
            feature_dims=processor.feature_dims,
            embedding_dim=self.config['EMBEDDING_DIM'],
            hidden_units=self.config['HIDDEN_UNITS'],
            dropout=self.config['DROPOUT'],
            num_numerical=len(processor.numerical_features)
        ).to(self.device)

        print(f"🧠 DeepFM initialized:")
        print(f"  Feature dims: {processor.feature_dims}")
        print(f"  Embedding dim: {self.config['EMBEDDING_DIM']}")
        print(f"  Numerical features: {len(processor.numerical_features)}")
        print(f"  Deep layers: {self.config['HIDDEN_UNITS']}")

        return model

    def setup_training(self, model, train_df):
        """훈련 설정"""
        # 클래스 가중치 계산
        pos_count = len(train_df[train_df['clicked'] == 1])
        neg_count = len(train_df[train_df['clicked'] == 0])
        pos_weight = torch.tensor([neg_count / pos_count]).to(self.device)

        print(f"🏋️ Training with class weight: {pos_weight.item():.2f}")

        # 손실 함수 및 최적화기
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=self.config['LEARNING_RATE'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2)

        return criterion, optimizer, scheduler

    def train_epoch(self, model, train_loader, criterion, optimizer):
        """단일 에포크 훈련"""
        model.train()
        train_loss = 0.0

        for categorical_inputs, numerical_inputs, targets in train_loader:
            categorical_inputs = {k: v.to(self.device) for k, v in categorical_inputs.items()}
            numerical_inputs = numerical_inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(categorical_inputs, numerical_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        return train_loss / len(train_loader)

    def validate_epoch(self, model, val_loader, criterion):
        """단일 에포크 검증"""
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for categorical_inputs, numerical_inputs, targets in val_loader:
                categorical_inputs = {k: v.to(self.device) for k, v in categorical_inputs.items()}
                numerical_inputs = numerical_inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(categorical_inputs, numerical_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)

        # 성능 지표 계산
        comp_score, ap, wll, auc = calculate_competition_metrics(
            np.array(val_targets), np.array(val_preds)
        )

        return val_loss, comp_score, ap, wll, auc

    def train(self):
        """전체 훈련 프로세스"""
        # WandB 초기화
        run_name = f"deepfm-modular-{datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(
            project=self.wandb_config['PROJECT'],
            entity=self.wandb_config['ENTITY'],
            name=run_name,
            tags=self.wandb_config['TAGS'] + ['modular-structure'],
            notes=self.wandb_config['NOTES'] + ' (New modular structure)',
            config=self.config
        )

        print("🎯 Starting DeepFM Training for CTR Prediction")
        print("="*60)
        print(f"📊 WandB Run: {run_name}")

        try:
            # 데이터 준비
            train_sample = self.load_and_prepare_data()
            train_processed, processor = self.prepare_features(train_sample)
            train_loader, val_loader, train_df, val_df = self.create_datasets(train_processed, processor)

            # 모델 및 훈련 설정
            model = self.create_model(processor)
            criterion, optimizer, scheduler = self.setup_training(model, train_df)

            # 훈련 루프
            print("\nEpoch | Train Loss | Val Loss | Comp Score | AP     | WLL    | AUC    | LR")
            print("-" * 80)

            best_score = 0
            patience_counter = 0

            for epoch in range(1, self.config['EPOCHS'] + 1):
                # 훈련
                train_loss = self.train_epoch(model, train_loader, criterion, optimizer)

                # 검증
                val_loss, comp_score, ap, wll, auc = self.validate_epoch(model, val_loader, criterion)

                current_lr = optimizer.param_groups[0]['lr']

                # WandB 로깅
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'comp_score': comp_score,
                    'ap': ap,
                    'wll': wll if not np.isinf(wll) else None,
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
                        'epoch': epoch,
                        'config': self.config
                    }, 'deepfm_modular_best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['EARLY_STOPPING']:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break

            print(f"\n✅ Training completed! Best score: {best_score:.6f}")
            print(f"📁 Model saved: deepfm_modular_best_model.pth")

            return best_score

        finally:
            wandb.finish()


def train_deepfm(config=None, wandb_config=None):
    """DeepFM 훈련 실행"""
    trainer = DeepFMTrainer(config, wandb_config)
    return trainer.train()


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