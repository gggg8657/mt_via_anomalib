"""
Accurate Interpretable VAD (AiVAD) 공식 학습 방법
anomalib의 공식 문서와 GitHub 예제를 따라 구현
"""

import os
import torch
from pathlib import Path
from anomalib.models.video import AiVad
from anomalib.data import Avenue
from anomalib.engine import Engine
from anomalib.data.datasets.base.video import VideoTargetFrame

def main():
    """AiVAD 공식 학습 방법"""
    print("=" * 60)
    print("🚀 Accurate Interpretable VAD (AiVAD) 공식 학습 시작")
    print("=" * 60)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ 사용 디바이스: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 1. Avenue 데이터셋 준비 (공식 데이터셋)
    print("\n📁 Avenue 데이터셋 준비 중...")
    print("💡 Avenue 데이터셋은 자동으로 다운로드됩니다.")
    print("💡 처음 실행 시 시간이 걸릴 수 있습니다.")
    
    try:
        datamodule = Avenue(
            root="./data/anomalib",  # 데이터 저장 경로
            clip_length_in_frames=2,  # AiVAD는 2프레임 클립 사용
            frames_between_clips=1,   # 클립 간 1프레임 간격
            target_frame=VideoTargetFrame.LAST,  # 마지막 프레임 타겟
            num_workers=4,            # 데이터 로더 워커 수
            train_batch_size=8,       # 훈련 배치 크기 (GPU 메모리에 따라 조정)
            eval_batch_size=8,        # 검증 배치 크기
        )
        print("✅ Avenue 데이터 모듈 생성 완료")
        
    except Exception as e:
        print(f"❌ 데이터셋 준비 실패: {e}")
        print("\n💡 대안: 커스텀 비디오 데이터 사용")
        print("💡 또는 Avenue 데이터셋이 자동 다운로드되기를 기다리세요.")
        return
    
    # 2. AiVAD 모델 초기화 (공식 설정)
    print("\n🤖 AiVAD 모델 초기화 (공식 설정)...")
    try:
        model = AiVad(
            # 공식 논문의 기본 설정
            use_velocity_features=True,   # 속도 특성 사용
            use_pose_features=True,       # 포즈 특성 사용
            use_deep_features=True,       # 딥 특성 사용
            # Density estimation 설정
            n_components_velocity=2,      # 속도 특성의 GMM 컴포넌트 수
            n_neighbors_pose=1,           # 포즈 특성의 k-NN
            n_neighbors_deep=1,           # 딥 특성의 k-NN
            # 객체 감지 설정 (기본값)
            box_score_thresh=0.7,
            min_bbox_area=100,
            max_bbox_overlap=0.65,
            foreground_binary_threshold=18,
        )
        model = model.to(device)
        print("✅ AiVAD 모델 생성 완료")
        
    except Exception as e:
        print(f"❌ 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 학습 엔진 설정 (공식 방법)
    print("\n🔧 PyTorch Lightning Engine 설정 (공식 방법)...")
    try:
        engine = Engine(
            devices=1 if device == "cuda" else "auto",
            accelerator="gpu" if device == "cuda" else "cpu",
            precision="16-mixed" if device == "cuda" else "32",  # Mixed precision (GPU 성능 향상)
            max_epochs=50,                # 공식 논문에서 권장하는 에포크 수
            gradient_clip_val=1.0,        # 그래디언트 클리핑
            accumulate_grad_batches=1,    # 그래디언트 누적
            log_every_n_steps=10,         # 로그 출력 주기
            val_check_interval=0.5,       # 검증 주기 (에포크의 50%마다)
            enable_progress_bar=True,      # 진행 표시줄
            enable_model_summary=True,     # 모델 요약 출력
        )
        print("✅ Engine 설정 완료")
        
    except Exception as e:
        print(f"❌ Engine 설정 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 학습 시작
    print("\n🎯 AiVAD 학습 시작!")
    print("💡 학습 과정:")
    print("   1. 정상 비디오 클립으로 Feature Extraction (Flow, Region, Pose, Deep)")
    print("   2. Density Estimator로 정상 데이터의 분포 학습")
    print("   3. One-Class Learning (정상 데이터만 사용)")
    print("   4. 이상 탐지를 위한 분포 모델 생성")
    print()
    
    try:
        # 공식 학습 방법: engine.fit() 사용
        engine.fit(model=model, datamodule=datamodule)
        
        print("\n✅ 학습 완료!")
        
    except Exception as e:
        print(f"\n❌ 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 체크포인트 저장 (공식 방법)
    checkpoint_path = "aivad_official_trained.ckpt"
    print(f"\n💾 체크포인트 저장 중: {checkpoint_path}")
    
    try:
        # PyTorch Lightning의 공식 저장 방법
        if hasattr(engine, 'save_checkpoint'):
            engine.save_checkpoint(checkpoint_path)
        else:
            # 수동 저장 (백업)
            torch.save({
                'state_dict': model.state_dict(),
                'hyper_parameters': model.hparams if hasattr(model, 'hparams') else {},
                'pytorch-lightning_version': '2.0.0',
                'model_class': 'AiVad',
                'training_type': 'official_anomalib_method',
            }, checkpoint_path)
        
        print(f"✅ 체크포인트 저장 완료: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
        # 최소한의 저장 시도
        try:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ 최소 저장 완료 (state_dict만): {checkpoint_path}")
        except Exception as e2:
            print(f"❌ 최소 저장도 실패: {e2}")
    
    print("\n" + "=" * 60)
    print("🎉 AiVAD 공식 학습 완료!")
    print("=" * 60)
    print(f"\n💡 다음 단계:")
    print(f"1. UI에서 '{checkpoint_path}' 파일을 로드하여 사용")
    print(f"2. 이상 탐지 테스트 수행")
    print(f"3. 필요시 추가 파인튜닝")
    print()

if __name__ == "__main__":
    main()

