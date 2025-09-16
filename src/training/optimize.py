"""
Optuna 하이퍼파라미터 최적화 로직
"""

import pandas as pd
import numpy as np
import gc
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import optuna
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

from ..models.deepfm_core import DeepFM, DeepFMDataset
from ..models.feature_processor import TrainingFeatureProcessor
from ..utils.metrics import calculate_competition_metrics
from ..utils.config import OPTUNA_CONFIG


class DeepFMOptimizer:
    """DeepFM Optuna 최적화 클래스"""

    def __init__(self, config=None):
        self.config = config or OPTUNA_CONFIG
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.wandb_project = "toss-ctr-deepfm-optuna"

        print(f"🚀 Using device: {self.device}")

    def load_sample_data(self):
        """최적화용 샘플 데이터 로드"""
        print("📊 Loading sample data for optimization...")
        all_train = pd.read_parquet("data/train.parquet", engine="pyarrow")

        # 빠른 최적화를 위한 샘플링
        sample_size = self.config['SAMPLE_SIZE']
        clicked_1 = all_train[all_train['clicked'] == 1]
        clicked_0 = all_train[all_train['clicked'] == 0]

        pos_samples = len(clicked_1)
        neg_samples = min(sample_size - pos_samples, len(clicked_0))

        if neg_samples > 0:
            sampled_neg = clicked_0.sample(n=neg_samples, random_state=42)
            train_sample = pd.concat([clicked_1, sampled_neg], axis=0).sample(
                frac=1, random_state=42).reset_index(drop=True)
        else:
            train_sample = all_train.sample(n=sample_size, random_state=42).reset_index(drop=True)

        del all_train, clicked_1, clicked_0
        gc.collect()

        print(f"Sample size: {len(train_sample):,}")
        print(f"Class distribution: {train_sample['clicked'].value_counts().to_dict()}")

        return train_sample

    def objective(self, trial):
        """Optuna 목적 함수"""

        # 하이퍼파라미터 제안 (Trial 2 주변 탐색 추가)
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'embedding_dim': trial.suggest_int('embedding_dim', 16, 48, step=8),  # 24 주변
            'hidden_units': trial.suggest_categorical('hidden_units', [
                [256, 128],
                [512, 256, 128],      # Trial 2 최적값
                [768, 512, 256],
                [1024, 512, 256, 128]
            ]),
            'dropout': trial.suggest_float('dropout', 0.2, 0.5),  # 0.337 주변
            'batch_size': trial.suggest_int('batch_size', 256, 1024, step=128),  # 512 주변
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        }

        # WandB 초기화 (각 trial별로)
        run_name = f"trial-{trial.number:03d}-{datetime.now().strftime('%H%M')}"
        wandb.init(
            project=self.wandb_project,
            name=run_name,
            tags=['optuna', 'hyperopt', 'trial', 'v2'],
            config=params,
            reinit=True
        )

        try:
            # 데이터 로드
            print(f"\n🔍 Trial {trial.number}: {params}")
            train_sample = self.load_sample_data()

            # 피처 전처리
            processor = TrainingFeatureProcessor()
            train_processed = processor.fit_transform(train_sample.copy())

            # Train/Validation 분할
            train_df, val_df = train_test_split(
                train_processed,
                test_size=0.15,
                random_state=42,
                stratify=train_processed['clicked']
            )

            # 데이터셋 생성
            train_dataset = DeepFMDataset(train_df, processor, 'clicked', True)
            val_dataset = DeepFMDataset(val_df, processor, 'clicked', True)

            train_loader = DataLoader(
                train_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

            # 모델 초기화
            model = DeepFM(
                feature_dims=processor.feature_dims,
                embedding_dim=params['embedding_dim'],
                hidden_units=params['hidden_units'],
                dropout=params['dropout'],
                num_numerical=len(processor.numerical_features)
            ).to(self.device)

            # 옵티마이저 설정
            if params['optimizer_type'] == 'adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=params['weight_decay']
                )
            else:  # adamw
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=params['weight_decay']
                )

            # 손실 함수 (클래스 가중치)
            pos_weight = torch.tensor([len(train_df[train_df['clicked'] == 0]) / len(train_df[train_df['clicked'] == 1])]).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # 스케줄러
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=1)

            # 훈련 루프 (짧은 에포크)
            best_score = 0
            patience_counter = 0

            for epoch in range(1, self.config['EPOCHS_PER_TRIAL'] + 1):
                # 훈련
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

                train_loss /= len(train_loader)

                # 검증
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

                # 성능 지표
                comp_score, ap, wll, auc = calculate_competition_metrics(
                    np.array(val_targets), np.array(val_preds)
                )

                # WandB 로깅
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'comp_score': comp_score,
                    'ap': ap,
                    'wll': wll if not np.isinf(wll) else None,
                    'auc': auc,
                    'trial_number': trial.number
                })

                # 최고 점수 업데이트
                if comp_score > best_score:
                    best_score = comp_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['EARLY_STOPPING']:
                        break

                scheduler.step(val_loss)

                # Optuna pruning
                trial.report(comp_score, epoch)
                if trial.should_prune():
                    wandb.log({'pruned': True})
                    raise optuna.exceptions.TrialPruned()

            # 최종 점수 로깅
            wandb.log({'final_score': best_score})

            print(f"Trial {trial.number} completed: {best_score:.4f}")
            return best_score

        except Exception as e:
            wandb.log({'error': str(e)})
            print(f"Trial {trial.number} failed: {e}")
            return 0.0

        finally:
            wandb.finish()
            gc.collect()

    def run_optimization(self):
        """Optuna 최적화 실행"""

        print("🚀 Starting Optuna + WandB Hyperparameter Optimization (V2)")
        print("="*70)
        print(f"📊 Configuration:")
        print(f"  Trials: {self.config['N_TRIALS']}")
        print(f"  Epochs per trial: {self.config['EPOCHS_PER_TRIAL']}")
        print(f"  Sample size: {self.config['SAMPLE_SIZE']:,}")
        print(f"  Device: {self.device}")

        # WandB 콜백 설정
        wandbc = WeightsAndBiasesCallback(
            metric_name="comp_score",
            wandb_kwargs={
                "project": self.wandb_project,
                "name": f"optuna-study-v2-{datetime.now().strftime('%m%d_%H%M')}",
                "tags": ["optuna", "study-overview", "v2"]
            }
        )

        # Optuna 스터디 생성
        study = optuna.create_study(
            study_name=f"{self.config['STUDY_NAME']}-v2",
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        )

        # 최적화 실행
        study.optimize(
            self.objective,
            n_trials=self.config['N_TRIALS'],
            timeout=self.config['TIMEOUT'],
            callbacks=[wandbc]
        )

        # 결과 출력
        print("\n🏆 Optimization Results:")
        print("="*50)
        print(f"Best score: {study.best_value:.6f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # 최적 파라미터 저장
        with open('optuna_best_params_v2.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)

        print(f"\n📁 Best parameters saved: optuna_best_params_v2.json")
        print(f"🔮 Apply these parameters to config.py for final training!")

        return study.best_params


def optimize_deepfm(config=None):
    """DeepFM 최적화 실행"""
    optimizer = DeepFMOptimizer(config)
    return optimizer.run_optimization()


if __name__ == "__main__":
    try:
        best_params = optimize_deepfm()
        print(f"\n🎯 Optimization completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()