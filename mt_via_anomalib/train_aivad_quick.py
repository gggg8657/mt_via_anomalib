
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
        print("\n🎉 빠른 학습이 성공적으로 완료되었습니다!")
        print("이제 realtime_ui_advanced_windows.py에서 체크포인트를 로드할 수 있습니다.")
    else:
        print("\n💥 학습에 실패했습니다.")
        sys.exit(1)
