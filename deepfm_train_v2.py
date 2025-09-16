"""
DeepFM 훈련 스크립트 (모듈화 버전)
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.train import train_deepfm
from src.utils.config import DEEPFM_CONFIG, WANDB_CONFIG


if __name__ == "__main__":
    try:
        print("🚀 Starting DeepFM Training (Modular Version)")
        print("="*60)

        # 설정 출력
        print("📊 Configuration:")
        for key, value in DEEPFM_CONFIG.items():
            print(f"  {key}: {value}")

        print("\n" + "="*60)

        best_score = train_deepfm(DEEPFM_CONFIG, WANDB_CONFIG)
        print(f"\n🏆 Final DeepFM score: {best_score:.6f}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()