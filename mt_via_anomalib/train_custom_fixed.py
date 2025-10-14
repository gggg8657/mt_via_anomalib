"""
커스텀 비디오 데이터셋으로 AI-VAD 모델 학습 (수정된 버전)
Avenue 데이터 모듈을 그대로 사용하여 안정성 확보
"""

import os
import pathlib
import torch
import shutil
from anomalib.models.video import AiVad
from anomalib.data.datamodules.video.avenue import Avenue
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.engine import Engine


def prepare_custom_dataset_for_avenue(dataset_path: str, video_files: list):
    """
    커스텀 비디오 파일들을 Avenue 형식으로 준비
    
    Args:
        dataset_path: 데이터셋이 저장될 경로
        video_files: 비디오 파일 경로 리스트
    """
    dataset_path = pathlib.Path(dataset_path)
    
    # 기존 데이터셋 정리
    if dataset_path.exists():
        try:
            shutil.rmtree(dataset_path, ignore_errors=True)
            print("✅ 기존 데이터셋 정리 완료")
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
            
        # Avenue 형식의 파일명 생성 (숫자로 시작)
        dest_path = train_path / f"{i:02d}.mp4"
        
        try:
            # 기존 파일이 있으면 삭제
            if os.path.exists(dest_path):
                os.remove(dest_path)
            
            # 파일 복사
            shutil.copy2(video_file, dest_path)
            print(f"✅ 복사 완료: {os.path.basename(video_file)} -> {dest_path.name}")
            successful_files += 1
            
        except Exception as e:
            print(f"❌ 복사 실패: {video_file} - {e}")
    
    if successful_files == 0:
        raise FileNotFoundError("복사된 비디오 파일이 없습니다. 파일 경로를 확인하세요.")
    
    print(f"✅ 커스텀 데이터셋 준비 완료: {successful_files}개 파일 처리됨")
    return successful_files


def main():
    print("🚀 커스텀 비디오 데이터셋으로 AI-VAD 모델 학습 시작 (수정된 버전)...")
    
    # ===== 여기를 수정하세요 =====
    # 1. 비디오 파일 경로들을 여기에 추가하세요
    video_files = [
        # 예시:
        # "C:\\Users\\YourName\\Videos\\normal_video1.mp4",
        # "C:\\Users\\YourName\\Videos\\normal_video2.avi",
        # "D:\\SecurityCameras\\normal_footage.mp4",
        "C:\\Users\\User\Documents\\repos\VAD\CV_module_test_tmp\STEAD\\videos\\normal_video.mp4",
        "C:\\Users\\User\Documents\\repos\VAD\CV_module_test_tmp\STEAD\\videos\\unknown_video.mp4",
        # "C:\Users\User\Documents\repos\VAD\CV_module_test_tmp\STEAD\videos\rest_video.mp4"
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
        print("\n📋 사용 방법:")
        print("1. train_custom_fixed.py 파일을 열어서 video_files 리스트에 비디오 파일 경로를 추가하세요")
        print("2. Windows 경로 예시:")
        print('   "C:\\\\Users\\\\YourName\\\\Videos\\\\normal_video1.mp4"')
        print('   "D:\\\\SecurityCameras\\\\normal_footage.mp4"')
        print("3. 스크립트를 다시 실행하세요")
        
        # 샘플 디렉토리 구조 생성
        dataset_path = pathlib.Path(dataset_path)
        train_path = dataset_path / "train"
        train_path.mkdir(parents=True, exist_ok=True)
        
        # README 파일 생성
        readme_path = train_path / "README_Fixed.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("""# 수정된 커스텀 비디오 데이터셋 학습

## 사용 방법

1. train_custom_fixed.py 파일을 편집하여 video_files 리스트에 비디오 파일 경로를 추가하세요:

```python
video_files = [
    "C:\\\\Users\\\\YourName\\\\Videos\\\\normal_video1.mp4",
    "D:\\\\SecurityCameras\\\\normal_footage.mp4",
    "E:\\\\MyVideos\\\\normal_clip.avi",
]
```

2. 스크립트를 실행하세요:

```bash
python train_custom_fixed.py
```

## 수정 사항
- CustomVideoDataModule 대신 기존 Avenue 데이터 모듈 사용
- train_data 속성 오류 해결
- 더 안정적인 학습 환경 제공

## 지원되는 비디오 형식
- .mp4 (권장)
- .avi  
- .mov
- .mkv
- .flv
- .wmv

## 주의사항
- 비디오 파일들이 정상적인 상황을 보여주는 것이 좋습니다
- 이상 상황이 포함된 비디오가 있다면 별도로 관리하세요
""")
        
        print(f"✅ 샘플 구조 생성 완료: {dataset_path}")
        print("📁 생성된 파일:")
        print(f"   - {readme_path}")
        return False
    
    # 커스텀 데이터셋 준비
    print("📁 커스텀 데이터셋 준비 중...")
    try:
        num_files = prepare_custom_dataset_for_avenue(dataset_path, video_files)
        if num_files == 0:
            print("❌ 처리된 비디오 파일이 없습니다.")
            return False
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
        checkpoint_path = "aivad_custom_fixed_checkpoint.ckpt"
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
    print("🔧 수정된 커스텀 비디오 데이터셋으로 AI-VAD 학습")
    print("=" * 60)
    
    success = main()
    if success:
        print("\n🎉 학습이 성공적으로 완료되었습니다!")
        print("이제 realtime_ui_advanced_windows.py에서 체크포인트를 로드할 수 있습니다.")
        print("\n체크포인트 파일:")
        print("- aivad_custom_fixed_checkpoint.ckpt")
    else:
        print("\n💥 학습에 실패했습니다.")
        print("\n📋 해결 방법:")
        print("1. video_files 리스트에 올바른 비디오 파일 경로를 추가하세요")
        print("2. 비디오 파일이 존재하는지 확인하세요")
        print("3. 지원되는 형식인지 확인하세요 (.mp4, .avi, .mov, .mkv, .flv, .wmv)")
        print("4. 관리자 권한으로 실행하세요")
        exit(1)
