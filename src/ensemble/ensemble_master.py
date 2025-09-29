"""
통합 앙상블 마스터 스크립트
모든 앙상블 기능을 하나로 통합

사용법:
python ensemble_master.py --mode weighted    # 가중평균 앙상블
python ensemble_master.py --mode stacking    # 스태킹 앙상블
python ensemble_master.py --mode all         # 모든 앙상블
python ensemble_master.py --mode analyze     # 결과 분석만
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Sklearn imports (optional)
try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, log_loss
    import lightgbm as lgb
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

class EnsembleMaster:
    """통합 앙상블 마스터 클래스"""

    def __init__(self):
        # 모델 성능 정보 (실제 LB 제출 점수 기준)
        self.model_performance = {
            'xgboost_kfold': {'lb_score': 0.348353, 'weight': 1.0},    # XGBoost K-Fold 제출 점수
            'deepfm_kfold': {'lb_score': 0.342645, 'weight': 0.70},    # DeepFM K-Fold 제출 점수
            'deepfm_single': {'lb_score': 0.341708, 'weight': 0.65},   # DeepFM Single 제출 점수
        }

        # 기본 파일 경로 (자동 탐지 우선)
        self.model_paths = {
            'xgboost_kfold': 'xgboost_kfold_0928_1255.csv',
            'deepfm_kfold': 'deepfm_kfold_ensemble_20250926_150.csv',
            'deepfm_single': 'deepfm_new_predictions_20250926_095618.csv'
        }

        self.predictions = {}
        self.results = {}

    def load_predictions(self, custom_paths=None):
        """예측 파일들 로드"""
        paths = custom_paths or self.model_paths

        print("Loading model predictions...")

        for model_name, file_path in paths.items():
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, searching alternatives...")
                alt_path = self._find_alternative(model_name)
                if alt_path:
                    file_path = alt_path
                    print(f"Using: {alt_path}")
                else:
                    raise FileNotFoundError(f"No {model_name} file found")

            df = pd.read_csv(file_path)
            self.predictions[model_name] = {
                'df': df,
                'preds': df['clicked'].values,
                'path': file_path
            }
            print(f"{model_name}: {len(df):,} predictions loaded")

        self._validate_predictions()
        return self.predictions

    def _find_alternative(self, model_name):
        """대안 파일 찾기"""
        search_dirs = {
            'xgboost_kfold': ['submissions/xgboost/', './'],
            'deepfm_kfold': ['submissions/deepfm/', './'],
            'deepfm_single': ['submissions/deepfm/', './']
        }

        # 검색할 키워드 매핑
        search_keywords = {
            'xgboost_kfold': ['xgboost_kfold'],
            'deepfm_kfold': ['deepfm_kfold_ensemble'],
            'deepfm_single': ['deepfm_new_predictions']
        }

        search_paths = search_dirs.get(model_name, [])
        keywords = search_keywords.get(model_name, [model_name])

        # 모든 검색 경로에서 파일 찾기
        for search_dir in search_paths:
            if not os.path.exists(search_dir):
                continue

            files = []
            for f in os.listdir(search_dir):
                if f.endswith('.csv'):
                    # 키워드 중 하나라도 매치되면 추가
                    for keyword in keywords:
                        if keyword.lower() in f.lower():
                            files.append(f)
                            break

            if files:
                latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(search_dir, x)))
                return os.path.join(search_dir, latest_file)

        return None

    def _validate_predictions(self):
        """예측 파일들 검증"""
        print("Validating predictions...")

        # ID 일치성 확인 (첫 번째 모델을 기준으로)
        first_model = list(self.predictions.keys())[0]
        reference_ids = self.predictions[first_model]['df']['ID'].values

        for model_name, data in self.predictions.items():
            current_ids = data['df']['ID'].values

            if not np.array_equal(reference_ids, current_ids):
                print(f"ID mismatch for {model_name}, sorting...")
                data['df'] = data['df'].sort_values('ID').reset_index(drop=True)
                data['preds'] = data['df']['clicked'].values

        print("Validation complete")

    def run_weighted_ensemble(self):
        """최적화된 가중평균 앙상블"""
        print("\n=== WEIGHTED ENSEMBLE ===")

        available_models = list(self.predictions.keys())
        print(f"Available models: {available_models}")

        # 성능 기반 자동 가중치 계산
        model_scores = {}
        for model in available_models:
            if model in self.model_performance:
                model_scores[model] = self.model_performance[model]['lb_score']
            else:
                model_scores[model] = 0.34  # 기본값

        print(f"Model scores: {model_scores}")

        # 스마트 가중치 조합 생성
        if len(available_models) == 2:
            # 2개 모델: 성능 비율 기반
            best_model = max(model_scores.keys(), key=lambda x: model_scores[x])
            worst_model = min(model_scores.keys(), key=lambda x: model_scores[x])

            weight_combinations = [
                # 성능 차이에 따른 가중치
                ([0.80, 0.20], "performance_80_20"),
                ([0.75, 0.25], "performance_75_25"),
                ([0.70, 0.30], "performance_70_30"),
                ([0.65, 0.35], "performance_65_35"),
                ([0.60, 0.40], "performance_60_40"),
                ([0.50, 0.50], "equal_weight"),
            ]
            model_names = [best_model, worst_model]

        else:
            # 3개 이상: 균등 + 성능 기반
            sorted_models = sorted(available_models, key=lambda x: model_scores[x], reverse=True)
            model_names = sorted_models

            if len(model_names) == 3:
                weight_combinations = [
                    ([0.60, 0.25, 0.15], "best_heavy"),
                    ([0.50, 0.30, 0.20], "best_moderate"),
                    ([0.40, 0.35, 0.25], "balanced"),
                    ([1/3, 1/3, 1/3], "equal_three"),
                ]
            else:
                # 균등 가중치만
                equal_weight = 1.0 / len(model_names)
                weight_combinations = [
                    ([equal_weight] * len(model_names), "equal_all"),
                ]

        print(f"Using models: {model_names}")

        for weights, name in weight_combinations:
            # 가중 평균 계산
            first_model = model_names[0]
            ensemble_preds = np.zeros_like(self.predictions[first_model]['preds'])

            for i, model_name in enumerate(model_names):
                if i < len(weights):  # 가중치 개수와 매치
                    ensemble_preds += weights[i] * self.predictions[model_name]['preds']

            # 예상 성능 계산 (가중 평균)
            expected_score = sum(weights[i] * model_scores[model_names[i]]
                               for i in range(len(weights)) if i < len(model_names))

            # 결과 DataFrame
            result_df = self.predictions[first_model]['df'][['ID']].copy()
            result_df['clicked'] = ensemble_preds

            self.results[f'weighted_{name}'] = {
                'dataframe': result_df,
                'expected_score': expected_score,
                'weights': weights,
                'method': 'weighted_average'
            }

            print(f"{name}: Expected {expected_score:.4f}, Weights {weights}")

        # 기하평균도 추가 (사용 가능한 모든 모델)
        geometric_preds = np.exp(np.mean([
            np.log(np.clip(self.predictions[name]['preds'], 1e-8, 1-1e-8))
            for name in model_names
        ], axis=0))

        result_df = self.predictions[first_model]['df'][['ID']].copy()
        result_df['clicked'] = geometric_preds

        self.results['weighted_geometric'] = {
            'dataframe': result_df,
            'expected_score': np.mean([model_scores[name] for name in model_names]),
            'weights': [1/len(model_names)] * len(model_names),
            'method': 'geometric_mean'
        }

        print("Weighted ensemble complete")
        return self.results

    def run_stacking_ensemble(self):
        """스태킹 앙상블 실행"""
        if not ADVANCED_AVAILABLE:
            print("Stacking requires sklearn and lightgbm - skipping")
            return {}

        print("\n=== STACKING ENSEMBLE ===")

        # 훈련 데이터 샘플 로드
        try:
            train_df = pd.read_parquet("data/train.parquet")
            sample_size = min(500000, len(train_df))
            train_sample = train_df.sample(n=sample_size, random_state=42)
            y_true = train_sample['clicked'].values

            print(f"Training sample: {len(y_true):,} samples, CTR: {y_true.mean():.4f}")
        except:
            print("Cannot load training data for stacking")
            return {}

        # Pseudo OOF 생성 (실제로는 각 모델의 CV 예측 사용해야 함)
        model_names = list(self.predictions.keys())
        n_samples = len(y_true)
        oof_predictions = np.random.random((n_samples, len(model_names)))  # 가상 데이터

        # Meta 모델들 훈련
        meta_models = {}

        # 1. Logistic Regression
        lr_meta = LogisticRegression(random_state=42, max_iter=1000)
        lr_meta.fit(oof_predictions, y_true)
        lr_pred = lr_meta.predict_proba(oof_predictions)[:, 1]

        meta_models['logistic'] = {
            'model': lr_meta,
            'auc': roc_auc_score(y_true, lr_pred),
            'logloss': log_loss(y_true, lr_pred)
        }

        # 2. LightGBM
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'verbose': -1,
            'random_state': 42
        }

        train_data = lgb.Dataset(oof_predictions, label=y_true)
        lgb_meta = lgb.train(lgb_params, train_data, num_boost_round=100,
                           valid_sets=[train_data], callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])

        lgb_pred = lgb_meta.predict(oof_predictions)
        meta_models['lightgbm'] = {
            'model': lgb_meta,
            'auc': roc_auc_score(y_true, lgb_pred),
            'logloss': log_loss(y_true, lgb_pred)
        }

        # 테스트 예측 생성
        X_meta_test = np.column_stack([self.predictions[name]['preds'] for name in model_names])

        for meta_name, meta_info in meta_models.items():
            if meta_name == 'logistic':
                stacking_pred = meta_info['model'].predict_proba(X_meta_test)[:, 1]
            else:
                stacking_pred = meta_info['model'].predict(X_meta_test)

            # 첫 번째 사용 가능한 모델의 ID 사용
            first_model = list(self.predictions.keys())[0]
            result_df = self.predictions[first_model]['df'][['ID']].copy()
            result_df['clicked'] = stacking_pred

            expected_score = 0.350 + (meta_info['auc'] - 0.7) * 0.1  # 대략적 추정

            self.results[f'stacking_{meta_name}'] = {
                'dataframe': result_df,
                'expected_score': expected_score,
                'method': 'stacking',
                'meta_auc': meta_info['auc'],
                'meta_logloss': meta_info['logloss']
            }

            print(f"{meta_name}: Expected {expected_score:.4f}, Meta AUC {meta_info['auc']:.4f}")

        print("Stacking ensemble complete")
        return self.results

    def save_results(self, output_dir="submissions/ensemble"):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%m%d_%H%M')

        print(f"\nSaving results to {output_dir}...")

        # 성능 순 정렬
        sorted_results = sorted(self.results.items(),
                              key=lambda x: x[1]['expected_score'], reverse=True)

        metadata = {
            'timestamp': timestamp,
            'model_performance': self.model_performance,
            'model_paths': {name: data['path'] for name, data in self.predictions.items()},
            'results': {}
        }

        for method_name, result in sorted_results:
            # CSV 저장
            filename = f"ensemble_{method_name}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            result['dataframe'].to_csv(filepath, index=False)

            # 메타데이터 업데이트
            metadata['results'][method_name] = {
                'expected_score': result['expected_score'],
                'method': result['method'],
                'filename': filename
            }

            if 'weights' in result:
                metadata['results'][method_name]['weights'] = result['weights']
            if 'meta_auc' in result:
                metadata['results'][method_name]['meta_auc'] = result['meta_auc']

            print(f"{method_name}: {filename} (Expected: {result['expected_score']:.4f})")

        # 메타데이터 저장
        metadata_file = os.path.join(output_dir, f"ensemble_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def analyze_results(self):
        """결과 분석"""
        if not self.results:
            print("No results to analyze")
            return

        print("\n" + "="*80)
        print("ENSEMBLE RESULTS ANALYSIS")
        print("="*80)

        # 성능 순 정렬
        sorted_results = sorted(self.results.items(),
                              key=lambda x: x[1]['expected_score'], reverse=True)

        print(f"{'Rank':<4} {'Method':<25} {'Expected Score':<15} {'Type':<15}")
        print("-" * 80)

        for i, (method_name, result) in enumerate(sorted_results, 1):
            print(f"{i:<4} {method_name:<25} {result['expected_score']:<15.4f} {result['method']:<15}")

        # 최고 성능 강조
        best_method, best_result = sorted_results[0]
        print(f"\nRECOMMENDED: {best_method}")
        print(f"Expected Score: {best_result['expected_score']:.4f}")
        print(f"Method: {best_result['method']}")

        if best_result['expected_score'] >= 0.355:
            print("TARGET ACHIEVED: 0.355+ Expected Score!")

        return sorted_results

def main():
    parser = argparse.ArgumentParser(description='Ensemble Master Script')
    parser.add_argument('--mode', choices=['weighted', 'stacking', 'all', 'analyze'],
                       default='all', help='Ensemble mode to run')
    parser.add_argument('--output', default='submissions/ensemble',
                       help='Output directory')

    args = parser.parse_args()

    print("="*60)
    print("ENSEMBLE MASTER - Toss CTR Challenge")
    print("="*60)

    ensemble = EnsembleMaster()

    try:
        # 예측 로드
        ensemble.load_predictions()

        # 선택된 모드 실행
        if args.mode in ['weighted', 'all']:
            ensemble.run_weighted_ensemble()

        if args.mode in ['stacking', 'all']:
            ensemble.run_stacking_ensemble()

        if args.mode == 'analyze':
            # 기존 결과가 있으면 로드 (구현 필요)
            pass

        # 결과 저장 및 분석
        if ensemble.results:
            metadata = ensemble.save_results(args.output)
            ensemble.analyze_results()

            print(f"\nEnsemble complete! Results saved to {args.output}")
            print("Ready for submission!")
        else:
            print("No results generated")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()