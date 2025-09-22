"""
기본 프로젝트 설정 - Toss CTR Challenge
경로, 디렉토리, 기본 파라미터 등 프로젝트 인프라 설정 관리

역할:
- 프로젝트 경로 및 디렉토리 구조
- 데이터 파일 경로
- 기본 XGBoost 파라미터
- 범주형 피처 정의 (CATEGORICAL_FEATURES)
- 디렉토리 생성 및 검증 함수

모델별 하이퍼파라미터는 src/utils/config.py 참조
"""

import os
from pathlib import Path

# =============================================================================
# 프로젝트 경로 설정
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'

# 구조화된 모델 디렉토리
MODEL_SHARED_DIR = MODEL_DIR / 'shared'
MODEL_EXPERIMENTS_DIR = MODEL_DIR / 'experiments'
MODEL_PRODUCTION_DIR = MODEL_DIR / 'production'

# =============================================================================
# 데이터 파일 경로
# =============================================================================
TRAIN_PATH = DATA_DIR / 'train.parquet'
TEST_PATH = DATA_DIR / 'test.parquet'
OUTPUT_X = PROCESSED_DIR / 'X_processed.parquet'
OUTPUT_Y = PROCESSED_DIR / 'y_processed.npy'
ENCODERS_PATH = MODEL_SHARED_DIR / 'label_encoders.pkl'
BEST_PARAMS_PATH = MODEL_SHARED_DIR / 'best_hyperparameters.json'

# =============================================================================
# 모델 설정
# =============================================================================
MODEL_TYPE = "xgb"
FEATURE_VERSION = "v2"
RANDOM_STATE = 42

# =============================================================================
# 피처 정의
# =============================================================================
CATEGORICAL_FEATURES = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']

# Critical features from analysis
CRITICAL_OUTLIER_FEATURES = ['feat_e_1', 'feat_e_2', 'feat_e_3', 'l_feat_2']

# =============================================================================
# 훈련 설정
# =============================================================================
N_FOLDS = 5
N_FOLDS_TUNE = 3  # 튜닝 시에는 빠른 실행을 위해 3-fold
DEFAULT_N_ESTIMATORS = 2500
EARLY_STOPPING_ROUNDS = 50
EARLY_STOPPING_ROUNDS_TUNE = 25  # 튜닝 시에는 더 빠른 조기 종료

# =============================================================================
# 하이퍼파라미터 튜닝 설정
# =============================================================================
N_TRIALS = 20

# =============================================================================
# GPU/CPU 자동 감지 설정
# =============================================================================
def get_device_config():
    """GPU 사용 가능 여부를 확인하고 적절한 설정 반환"""
    try:
        # GPU 사용 가능성 체크
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        gpu_available = result.returncode == 0

        if gpu_available:
            return {
                'tree_method': 'gpu_hist',
                'device': 'cuda'
            }
        else:
            return {
                'tree_method': 'hist',
                'device': 'cpu'
            }
    except (FileNotFoundError, subprocess.SubprocessError):
        # nvidia-smi가 없거나 실행 실패 시 CPU 사용
        return {
            'tree_method': 'hist',
            'device': 'cpu'
        }

# 디바이스 설정 (모듈 로드 시 한 번만 실행)
DEVICE_CONFIG = get_device_config()

# =============================================================================
# XGBoost 기본 파라미터
# =============================================================================
def get_base_xgb_params(pos_ratio):
    """기본 XGBoost 파라미터 반환 (scale_pos_weight 포함)"""
    scale_pos_weight = (1 - pos_ratio) / pos_ratio

    params = {
        'objective': 'binary:logistic',
        'verbosity': 0,
        'seed': RANDOM_STATE,
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        # 기본 하이퍼파라미터
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    # 디바이스 설정 추가
    params.update(DEVICE_CONFIG)

    return params

# =============================================================================
# 환경별 설정 (선택사항)
# =============================================================================
class Environment:
    """환경별 설정 관리"""

    @staticmethod
    def is_development():
        """개발 환경 여부"""
        return os.getenv('ENV', 'development') == 'development'

    @staticmethod
    def is_production():
        """운영 환경 여부"""
        return os.getenv('ENV') == 'production'

    @classmethod
    def get_n_trials(cls):
        """환경에 따른 튜닝 시행 횟수"""
        if cls.is_development():
            return min(N_TRIALS, 5)  # 개발 시 빠른 테스트
        return N_TRIALS

# =============================================================================
# 실험 관리 함수들
# =============================================================================
def get_experiment_dir(timestamp):
    """타임스탬프 기반 실험 디렉토리 경로 반환"""
    return MODEL_EXPERIMENTS_DIR / timestamp

def get_model_paths(run_name, timestamp):
    """실험별 모델 파일 경로들 반환"""
    exp_dir = get_experiment_dir(timestamp)

    return {
        'model': exp_dir / f'{MODEL_TYPE}-{FEATURE_VERSION}-train.json',
        'results': exp_dir / 'results.json',
        'feature_importance_csv': exp_dir / 'feature_importance.csv',
        'feature_importance_png': exp_dir / 'feature_importance.png',
        'experiment_dir': exp_dir
    }

# =============================================================================
# 디렉토리 생성 함수
# =============================================================================
def create_directories():
    """필요한 디렉토리들을 생성"""
    directories = [
        PROCESSED_DIR, MODEL_DIR, SUBMISSIONS_DIR,
        MODEL_SHARED_DIR, MODEL_EXPERIMENTS_DIR,
        MODEL_PRODUCTION_DIR
    ]
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)

# =============================================================================
# 설정 검증 함수
# =============================================================================
def validate_config():
    """설정의 유효성을 검증"""
    errors = []

    # 필수 디렉토리 존재 확인
    if not DATA_DIR.exists():
        errors.append(f"Data directory not found: {DATA_DIR}")

    # 훈련 데이터 존재 확인
    if not TRAIN_PATH.exists():
        errors.append(f"Training data not found: {TRAIN_PATH}")

    # GPU 설정 출력
    device_info = f"Using device: {DEVICE_CONFIG['device']} ({DEVICE_CONFIG['tree_method']})"
    print(f"[OK] {device_info}")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

    print("[OK] Configuration validation passed")

# =============================================================================
# 모듈 로드 시 자동 실행
# =============================================================================
if __name__ == "__main__":
    # 테스트 실행
    validate_config()
    create_directories()
    print(f"[OK] Configuration module loaded successfully")
    print(f"   - Project root: {PROJECT_ROOT}")
    print(f"   - Device: {DEVICE_CONFIG['device']}")
    print(f"   - Environment trials: {Environment.get_n_trials()}")