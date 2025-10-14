"""
Windows 권한 문제를 해결한 커스텀 비디오 데이터셋 학습 스크립트
"""

import os
import pathlib
import torch
import shutil
from anomalib.models.video import AiVad
from anomalib.data.datamodules.video.avenue import Avenue
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.engine import Engine


def fix_windows_permissions():
    """Windows 권한 문제 해결을 위한 설정"""
    
    print("🔧 Windows 권한 문제 해결 중...")
    
    # 1. 결과 디렉토리 권한 설정
    results_dir = "./results"
    if os.path.exists(results_dir):
        try:
            # 기존 results 디렉토리 삭제 (권한 문제 방지)
            shutil.rmtree(results_dir, ignore_errors=True)
            print("✅ 기존 results 디렉토리 정리 완료")
        except Exception as e:
            print(f"⚠️  results 디렉토리 정리 중 오류: {e}")
    
    # 2. 로그 디렉토리 설정
    os.environ.setdefault("ANOMALIB_LOGGER", "false")  # 로거 비활성화
    os.environ.setdefault("ANOMALIB_RESULTS_PATH", "./custom_results")  # 결과 경로 변경
    
    # 3. 임시 디렉토리 설정
    import tempfile
    temp_dir = tempfile.mkdtemp()
    os.environ.setdefault("TMPDIR", temp_dir)
    os.environ.setdefault("TEMP", temp_dir)
    
    print("✅ Windows 권한 설정 완료")


def prepare_custom_dataset_safe(dataset_path: str, video_files: list):
    """
    Windows에서 안전하게 커스텀 비디오 파일들을 준비
    
    Args:
        dataset_path: 데이터셋이 저장될 경로
        video_files: 비디오 파일 경로 리스트
    """
    dataset_path = pathlib.Path(dataset_path)
    
    # 기존 데이터셋 정리
    if dataset_path.exists():
        try:
            shutil.rmtree(dataset_path, ignore_errors=True)
        except Exception as e:
            print(f"⚠️  기존 데이터셋 정리 중 오류: {e}")
    
    # Avenue 형식의 디렉토리 구조 생성
    train_path = dataset_path / "train"
    train_path.mkdir(parents=True, exist_ok=True)
    
    successful_files = 0
    
    # 비디오 파일들을 train 디렉토리로 복사
    for i, video_file in enumerate(video_files):
        if not os.path.exists(video_file):
            print(f"⚠️  비디오 파일을 찾을 수 없습니다: {video_file}")
            continue
            
        # 파일 확장자 유지하며 복사
        dest_path = train_path / f"video_{i:04d}{pathlib.Path(video_file).suffix}"
        
        try:
            # 기존 파일이 있으면 삭제
            if os.path.exists(dest_path):
                os.remove(dest_path)
            
            # 파일 복사 (심볼릭 링크 대신 복사 사용)
            shutil.copy2(video_file, dest_path)
            print(f"✅ 복사 완료: {os.path.basename(video_file)} -> {dest_path.name}")
            successful_files += 1
            
        except Exception as e:
            print(f"❌ 복사 실패: {video_file} - {e}")
    
    if successful_files == 0:
        raise FileNotFoundError("복사된 비디오 파일이 없습니다. 파일 경로를 확인하세요.")
    
    print(f"✅ 커스텀 데이터셋 준비 완료: {successful_files}개 파일 처리됨")


def main():
    print("🚀 Windows 호환 커스텀 비디오 데이터셋으로 AI-VAD 모델 학습 시작...")
    
    # Windows 권한 문제 해결
    fix_windows_permissions()
    
    # ===== 여기를 수정하세요 =====
    # 1. 비디오 파일 경로들을 여기에 추가하세요
    video_files = [
        # 예시:
        # "C:\\Users\\YourName\\Videos\\normal_video1.mp4",
        # "C:\\Users\\YourName\\Videos\\normal_video2.avi",
        # "D:\\SecurityCameras\\normal_footage.mp4",
    ]
    
    # 2. 데이터셋이 저장될 경로
    dataset_path = "./custom_video_dataset"
    
    # 3. 학습 설정
    max_epochs = 3
    batch_size = 4
    # =============================
    
    # 비디오 파일이 없으면 안내 메시지
    if not video_files:
        print("⚠️  비디오 파일이 지정되지 않았습니다.")
        print("\n📋 Windows에서 사용 방법:")
        print("1. train_custom_windows_fix.py 파일을 열어서 video_files 리스트에 비디오 파일 경로를 추가하세요")
        print("2. Windows 경로 예시:")
        print('   "C:\\\\Users\\\\YourName\\\\Videos\\\\normal_video1.mp4"')
        print('   "D:\\\\SecurityCameras\\\\normal_footage.mp4"')
        print("3. 스크립트를 다시 실행하세요")
        
        # 샘플 디렉토리 구조 생성
        dataset_path = pathlib.Path(dataset_path)
        train_path = dataset_path / "train"
        train_path.mkdir(parents=True, exist_ok=True)
        
        # README 파일 생성
        readme_path = train_path / "README_Windows.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("""# Windows용 커스텀 비디오 데이터셋

## 사용 방법

1. train_custom_windows_fix.py 파일을 편집하여 video_files 리스트에 비디오 파일 경로를 추가하세요:

```python
video_files = [
    "C:\\\\Users\\\\YourName\\\\Videos\\\\normal_video1.mp4",
    "D:\\\\SecurityCameras\\\\normal_footage.mp4",
    "E:\\\\MyVideos\\\\normal_clip.avi",
]
```

2. 스크립트를 실행하세요:

```bash
python train_custom_windows_fix.py
```

## Windows 경로 주의사항
- 백슬래시를 두 개 사용하세요: \\\\
- 또는 raw string 사용: r"C:\\Users\\..."
- 경로에 공백이 있으면 따옴표로 감싸세요

## 지원되는 비디오 형식
- .mp4
- .avi  
- .mov
- .mkv
- .flv
- .wmv

## 권한 문제 해결
이 스크립트는 Windows 권한 문제를 자동으로 해결합니다:
- 결과 디렉토리 자동 정리
- 파일 복사 방식 사용 (심볼릭 링크 대신)
- 임시 디렉토리 설정
""")
        
        print(f"✅ 샘플 구조 생성 완료: {dataset_path}")
        print("📁 생성된 파일:")
        print(f"   - {readme_path}")
        return False
    
    # 커스텀 데이터셋 준비
    print("📁 커스텀 데이터셋 준비 중...")
    try:
        prepare_custom_dataset_safe(dataset_path, video_files)
    except Exception as e:
        print(f"❌ 데이터셋 준비 실패: {e}")
        return False
    
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
            num_workers=2,  # Windows에서 워커 수 줄임
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
    
    # 학습 엔진 설정 (Windows 최적화)
    print("⚙️  학습 엔진 설정...")
    try:
        engine = Engine(
            devices=1 if device == "cuda" else "auto",
            accelerator="gpu" if device == "cuda" else "cpu",
            precision=32,  # 32비트 정밀도 사용 (Windows 안정성)
            max_epochs=max_epochs,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            log_every_n_steps=5,
            val_check_interval=1.0,
            enable_progress_bar=True,
            enable_model_summary=True,
            limit_train_batches=10,  # 훈련 배치 수 제한 (테스트용)
            limit_val_batches=5,     # 검증 배치 수 제한 (테스트용)
            # Windows 특화 설정
            logger=False,  # 로거 비활성화 (권한 문제 방지)
            default_root_dir="./custom_results",  # 결과 디렉토리 변경
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
        checkpoint_path = "aivad_custom_windows_checkpoint.ckpt"
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
    print("🪟 Windows 호환 커스텀 비디오 데이터셋으로 AI-VAD 학습")
    print("=" * 60)
    
    success = main()
    if success:
        print("\n🎉 학습이 성공적으로 완료되었습니다!")
        print("이제 realtime_ui_advanced_windows.py에서 체크포인트를 로드할 수 있습니다.")
        print("\n체크포인트 파일:")
        print("- aivad_custom_windows_checkpoint.ckpt")
    else:
        print("\n💥 학습에 실패했습니다.")
        print("\n📋 Windows 권한 문제 해결 방법:")
        print("1. 관리자 권한으로 실행: 명령 프롬프트를 관리자로 실행")
        print("2. 바이러스 백신 예외 설정: 프로젝트 폴더를 예외 목록에 추가")
        print("3. 폴더 권한 설정: 프로젝트 폴더의 권한을 전체 제어로 변경")
        print("4. 다른 드라이브 사용: C: 드라이브 대신 D: 드라이브 사용")
        exit(1)
