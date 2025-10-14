"""
AI-VAD 모델 학습 스크립트
체크포인트가 없을 때 모델을 학습시켜 체크포인트를 생성합니다.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# 환경 설정
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")

# cuDNN 설정
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def setup_gpu():
    """GPU 설정 및 확인"""
    if torch.cuda.is_available():
        print(f"✅ GPU 사용 가능: {torch.cuda.device_count()}개")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return True
    else:
        print("❌ GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        return False

def create_training_script():
    """학습 스크립트 생성"""
    
    script_content = '''
import os
import torch
from anomalib.models.video import AiVad
from anomalib.data import Avenue
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.engine import Engine

def main():
    print("🚀 AI-VAD 모델 학습 시작...")
    
    # GPU 설정
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("CPU 사용")
    
    # 데이터 모듈 설정
    print("📁 데이터셋 로드 중...")
    datamodule = Avenue(
        root="/tmp/anomalib/data",  # 데이터 저장 경로
        clip_length_in_frames=2,
        frames_between_clips=1,
        target_frame=VideoTargetFrame.LAST,
        num_workers=4,  # 데이터 로더 워커 수
        batch_size=8,   # 배치 크기 (GPU 메모리에 따라 조정)
        train_batch_size=8,
        eval_batch_size=8,
        test_batch_size=8,
    )
    
    # 모델 초기화
    print("🤖 AI-VAD 모델 초기화...")
    model = AiVad()
    
    # 학습 엔진 설정
    print("⚙️  학습 엔진 설정...")
    engine = Engine(
        devices=1 if device == "cuda" else "auto",
        accelerator="gpu" if device == "cuda" else "cpu",
        precision="16-mixed" if device == "cuda" else "32",  # GPU에서는 mixed precision 사용
        max_epochs=50,  # 에포크 수
        gradient_clip_val=1.0,  # 그래디언트 클리핑
        accumulate_grad_batches=1,  # 그래디언트 누적
        log_every_n_steps=10,  # 로그 출력 주기
        val_check_interval=0.5,  # 검증 주기 (에포크의 50%마다)
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # 학습 시작
    print("🎯 학습 시작!")
    try:
        engine.fit(model=model, datamodule=datamodule)
        print("✅ 학습 완료!")
        
        # 체크포인트 저장
        checkpoint_path = "aivad_checkpoint.ckpt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 체크포인트 저장: {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎉 학습이 성공적으로 완료되었습니다!")
        print("이제 realtime_ui_advanced_windows.py에서 체크포인트를 로드할 수 있습니다.")
    else:
        print("\\n💥 학습에 실패했습니다.")
        sys.exit(1)
'''
    
    with open("train_aivad_main.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ 학습 스크립트 생성 완료: train_aivad_main.py")

def create_quick_training_script():
    """빠른 테스트용 학습 스크립트"""
    
    script_content = '''
import os
import torch
from anomalib.models.video import AiVad
from anomalib.data import Avenue
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.engine import Engine

def main():
    print("🚀 AI-VAD 빠른 학습 (테스트용)...")
    
    # GPU 설정
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("CPU 사용")
    
    # 데이터 모듈 설정 (작은 배치로 빠른 테스트)
    print("📁 데이터셋 로드 중...")
    datamodule = Avenue(
        root="/tmp/anomalib/data",
        clip_length_in_frames=2,
        frames_between_clips=1,
        target_frame=VideoTargetFrame.LAST,
        num_workers=2,
        batch_size=4,  # 작은 배치 크기
        train_batch_size=4,
        eval_batch_size=4,
        test_batch_size=4,
    )
    
    # 모델 초기화
    print("🤖 AI-VAD 모델 초기화...")
    model = AiVad()
    
    # 학습 엔진 설정 (빠른 테스트용)
    print("⚙️  학습 엔진 설정...")
    engine = Engine(
        devices=1 if device == "cuda" else "auto",
        accelerator="gpu" if device == "cuda" else "cpu",
        precision="16-mixed" if device == "cuda" else "32",
        max_epochs=3,  # 3 에포크만
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=5,
        val_check_interval=1.0,  # 매 에포크마다 검증
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_train_batches=10,  # 훈련 배치 수 제한
        limit_val_batches=5,     # 검증 배치 수 제한
    )
    
    # 학습 시작
    print("🎯 빠른 학습 시작!")
    try:
        engine.fit(model=model, datamodule=datamodule)
        print("✅ 빠른 학습 완료!")
        
        # 체크포인트 저장
        checkpoint_path = "aivad_quick_checkpoint.ckpt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 체크포인트 저장: {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎉 빠른 학습이 성공적으로 완료되었습니다!")
        print("이제 realtime_ui_advanced_windows.py에서 체크포인트를 로드할 수 있습니다.")
    else:
        print("\\n💥 학습에 실패했습니다.")
        sys.exit(1)
'''
    
    with open("train_aivad_quick.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ 빠른 테스트용 학습 스크립트 생성 완료: train_aivad_quick.py")

def main():
    parser = argparse.ArgumentParser(description="AI-VAD 모델 학습 스크립트 생성")
    parser.add_argument("--quick", action="store_true", help="빠른 테스트용 스크립트 생성")
    parser.add_argument("--gpu-info", action="store_true", help="GPU 정보만 출력")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI-VAD 모델 학습 스크립트 생성기")
    print("=" * 60)
    
    # GPU 정보 출력
    gpu_available = setup_gpu()
    
    if args.gpu_info:
        return
    
    if args.quick:
        create_quick_training_script()
    else:
        create_training_script()
    
    print("\n📋 다음 단계:")
    print("1. 데이터셋 다운로드 (자동으로 됩니다)")
    print("2. 학습 스크립트 실행:")
    
    if args.quick:
        print("   python train_aivad_quick.py")
        print("   (빠른 테스트용 - 3 에포크, 작은 배치)")
    else:
        print("   python train_aivad_main.py")
        print("   (전체 학습 - 50 에포크)")
    
    print("3. 학습 완료 후 체크포인트 파일 확인")
    print("4. realtime_ui_advanced_windows.py에서 체크포인트 로드")
    
    print("\n💡 GPU 사용량이 높으면:")
    print("   - batch_size를 줄이세요 (8 → 4 → 2)")
    print("   - num_workers를 줄이세요 (4 → 2 → 1)")
    print("   - precision을 '32'로 변경하세요")

if __name__ == "__main__":
    main()
