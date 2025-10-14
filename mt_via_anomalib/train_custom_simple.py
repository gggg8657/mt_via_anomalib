"""
커스텀 비디오 데이터셋으로 AI-VAD 모델 학습 (간단한 버전)
"""

import os
import pathlib
import torch
from anomalib.models.video import AiVad
from anomalib.data.datamodules.video.avenue import Avenue
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.engine import Engine


def prepare_custom_dataset(dataset_path: str, video_files: list):
    """
    커스텀 비디오 파일들을 Avenue 형식으로 준비
    
    Args:
        dataset_path: 데이터셋이 저장될 경로
        video_files: 비디오 파일 경로 리스트
    """
    dataset_path = pathlib.Path(dataset_path)
    
    # Avenue 형식의 디렉토리 구조 생성
    train_path = dataset_path / "train"
    train_path.mkdir(parents=True, exist_ok=True)
    
    # 비디오 파일들을 train 디렉토리로 복사 또는 심볼릭 링크 생성
    for i, video_file in enumerate(video_files):
        if not os.path.exists(video_file):
            print(f"⚠️  비디오 파일을 찾을 수 없습니다: {video_file}")
            continue
            
        # 파일 확장자 유지하며 복사
        dest_path = train_path / f"video_{i:04d}{pathlib.Path(video_file).suffix}"
        
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            
            # 심볼릭 링크 생성 (실제 복사보다 빠름)
            os.symlink(os.path.abspath(video_file), dest_path)
            print(f"✅ 링크 생성: {video_file} -> {dest_path.name}")
            
        except Exception as e:
            print(f"❌ 링크 생성 실패: {e}")
            # 심볼릭 링크가 실패하면 복사 시도
            try:
                import shutil
                shutil.copy2(video_file, dest_path)
                print(f"✅ 복사 완료: {video_file} -> {dest_path.name}")
            except Exception as e2:
                print(f"❌ 복사도 실패: {e2}")
    
    print(f"✅ 커스텀 데이터셋 준비 완료: {dataset_path}")


def main():
    print("🚀 커스텀 비디오 데이터셋으로 AI-VAD 모델 학습 시작...")
    
    # ===== 여기를 수정하세요 =====
    # 1. 비디오 파일 경로들을 여기에 추가하세요
    video_files = [
        # 예시:
        # "/path/to/your/video1.mp4",
        # "/path/to/your/video2.avi",
        # "/path/to/your/video3.mov",
    ]
    
    # 2. 데이터셋이 저장될 경로
    dataset_path = "./custom_video_dataset"
    
    # 3. 학습 설정
    max_epochs = 3
    batch_size = 4
    # =============================
    
    # 비디오 파일이 없으면 샘플 구조만 생성
    if not video_files:
        print("⚠️  비디오 파일이 지정되지 않았습니다.")
        print("\n📋 사용 방법:")
        print("1. train_custom_simple.py 파일을 열어서 video_files 리스트에 비디오 파일 경로를 추가하세요")
        print("2. 지원되는 형식: .mp4, .avi, .mov, .mkv, .flv, .wmv")
        print("3. 스크립트를 다시 실행하세요")
        
        # 샘플 디렉토리 구조 생성
        dataset_path = pathlib.Path(dataset_path)
        train_path = dataset_path / "train"
        train_path.mkdir(parents=True, exist_ok=True)
        
        # README 파일 생성
        readme_path = train_path / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("""# 커스텀 비디오 데이터셋

## 사용 방법

1. train_custom_simple.py 파일을 편집하여 video_files 리스트에 비디오 파일 경로를 추가하세요:

```python
video_files = [
    "/path/to/your/video1.mp4",
    "/path/to/your/video2.avi",
    "/path/to/your/video3.mov",
]
```

2. 스크립트를 실행하세요:

```bash
python train_custom_simple.py
```

## 지원되는 비디오 형식
- .mp4
- .avi  
- .mov
- .mkv
- .flv
- .wmv

## 주의사항
- 비디오 파일들이 정상적인 상황(이상이 없는 상황)을 보여주는 것이 좋습니다
- 이상 상황이 포함된 비디오가 있다면 별도로 관리하세요
""")
        
        print(f"✅ 샘플 구조 생성 완료: {dataset_path}")
        print("📁 생성된 파일:")
        print(f"   - {readme_path}")
        return False
    
    # 커스텀 데이터셋 준비
    print("📁 커스텀 데이터셋 준비 중...")
    prepare_custom_dataset(dataset_path, video_files)
    
    # GPU 설정
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print("CPU 사용")
    
    # Avenue 데이터 모듈 설정 (커스텀 데이터셋 경로 사용)
    print("📁 데이터셋 로드 중...")
    try:
        datamodule = Avenue(
            root=dataset_path,
            clip_length_in_frames=2,
            frames_between_clips=1,
            target_frame=VideoTargetFrame.LAST,
            num_workers=2,
        )
        
        # 배치 크기 설정
        datamodule.train_batch_size = batch_size
        datamodule.eval_batch_size = batch_size
        datamodule.test_batch_size = batch_size
        
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
    
    # 학습 엔진 설정
    print("⚙️  학습 엔진 설정...")
    try:
        engine = Engine(
            devices=1 if device == "cuda" else "auto",
            accelerator="gpu" if device == "cuda" else "cpu",
            precision=32,  # 32비트 정밀도 사용
            max_epochs=max_epochs,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            log_every_n_steps=5,
            val_check_interval=1.0,
            enable_progress_bar=True,
            enable_model_summary=True,
            limit_train_batches=10,  # 훈련 배치 수 제한 (테스트용)
            limit_val_batches=5,     # 검증 배치 수 제한 (테스트용)
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
        checkpoint_path = "aivad_custom_checkpoint.ckpt"
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
    print("=" * 60)
    print("🎬 커스텀 비디오 데이터셋으로 AI-VAD 학습 (간단한 버전)")
    print("=" * 60)
    
    success = main()
    if success:
        print("\n🎉 학습이 성공적으로 완료되었습니다!")
        print("이제 realtime_ui_advanced_windows.py에서 체크포인트를 로드할 수 있습니다.")
        print("\n체크포인트 파일:")
        print("- aivad_custom_checkpoint.ckpt")
    else:
        print("\n💥 학습에 실패했습니다.")
        exit(1)
