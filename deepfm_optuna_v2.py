"""
DeepFM Optuna 최적화 스크립트 (모듈화 버전)
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.optimize import optimize_deepfm
from src.utils.config import OPTUNA_CONFIG


if __name__ == "__main__":
    try:
        print("🚀 Starting DeepFM Optuna Optimization (Modular Version)")
        print("="*70)

        # 설정 출력
        print("📊 Optuna Configuration:")
        for key, value in OPTUNA_CONFIG.items():
            print(f"  {key}: {value}")

        print("\n" + "="*70)

        best_params = optimize_deepfm(OPTUNA_CONFIG)

        print(f"\n🏆 Best hyperparameters found!")
        print("="*50)
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        print(f"\n💡 Next steps:")
        print(f"  1. Update src/utils/config.py with these parameters")
        print(f"  2. Run: python deepfm_train_v2.py")
        print(f"  3. Expected score: 0.38+ 🎯")

    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()