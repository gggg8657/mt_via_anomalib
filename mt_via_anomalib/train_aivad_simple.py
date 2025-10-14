"""
AI-VAD 모델 학습 스크립트 (간단한 설정)
"""

import os
import torch
from anomalib.models.video import AiVad
from anomalib.data import Avenue
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.engine import Engine

def main():
    print("🚀 AI-VAD 모델 학습 시작 (간단한 설정)...")
    
    # GPU 설정
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print("CPU 사용")
    
    # 데이터 모듈 설정
    print("📁 데이터셋 로드 중...")
    try:
        datamodule = Avenue(
            root="/tmp/anomalib/data",
            clip_length_in_frames=2,
            frames_between_clips=1,
            target_frame=VideoTargetFrame.LAST,
            num_workers=2,  # 워커 수 줄임
        )
        
        # 배치 크기 설정
        datamodule.train_batch_size = 4  # 배치 크기 줄임
        datamodule.eval_batch_size = 4
        datamodule.test_batch_size = 4
        
        print("✅ 데이터셋 로드 완료")
        
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return False
    
    # 모델 초기화
    print("🤖 AI-VAD 모델 초기화...")
    try:
        model = AiVad()
        print("✅ 모델 초기화 완료")
    except Exception as e:
        print(f"❌ 모델 초기화 실패: {e}")
        return False
    
    # 학습 엔진 설정 (간단한 설정)
    print("⚙️  학습 엔진 설정...")
    try:
        engine = Engine(
            devices=1 if device == "cuda" else "auto",
            accelerator="gpu" if device == "cuda" else "cpu",
            precision=32,  # 32비트 정밀도 사용 (AMP 비활성화)
            max_epochs=2,  # 에포크 수 줄임
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            log_every_n_steps=5,
            val_check_interval=1.0,
            enable_progress_bar=True,
            enable_model_summary=True,
            limit_train_batches=5,  # 훈련 배치 수 더 줄임
            limit_val_batches=3,    # 검증 배치 수 더 줄임
        )
        print("✅ 학습 엔진 설정 완료")
        
    except Exception as e:
        print(f"❌ 학습 엔진 설정 실패: {e}")
        return False
    
    # 학습 시작
    print("🎯 학습 시작!")
    try:
        engine.fit(model=model, datamodule=datamodule)
        print("✅ 학습 완료!")
        
        # 체크포인트 저장
        checkpoint_path = "aivad_checkpoint_simple.ckpt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 체크포인트 저장: {checkpoint_path}")
        
        # 체크포인트 파일 크기 확인
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
            print(f"📊 체크포인트 크기: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 학습이 성공적으로 완료되었습니다!")
        print("이제 realtime_ui_advanced_windows.py에서 체크포인트를 로드할 수 있습니다.")
        print("\n체크포인트 파일:")
        print("- aivad_checkpoint_simple.ckpt")
    else:
        print("\n💥 학습에 실패했습니다.")
        exit(1)
