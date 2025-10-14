"""
커스텀 비디오 데이터셋으로 AI-VAD 모델 학습 스크립트
"""

import os
import pathlib
import torch
from anomalib.models.video import AiVad
from anomalib.data.datamodules.base.video import AnomalibVideoDataModule
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.engine import Engine


class CustomVideoDataModule(AnomalibVideoDataModule):
    """
    커스텀 비디오 데이터셋을 위한 데이터 모듈
    
    데이터셋 구조:
    custom_video_dataset/
    ├── train/
    │   ├── normal/          # 정상 비디오 파일들
    │   │   ├── video1.mp4
    │   │   ├── video2.avi
    │   │   └── ...
    │   └── abnormal/        # 이상 비디오 파일들 (선택사항)
    │       ├── anomaly1.mp4
    │       └── ...
    ├── val/                 # 검증 데이터 (선택사항)
    │   ├── normal/
    │   └── abnormal/
    └── test/                # 테스트 데이터 (선택사항)
        ├── normal/
        └── abnormal/
    """
    
    def __init__(
        self,
        root: str = "./custom_video_dataset",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 4,
        **kwargs
    ):
        self.root = pathlib.Path(root)
        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.target_frame = target_frame
        
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            **kwargs
        )
    
    def setup(self, stage: str = None) -> None:
        """데이터셋 설정"""
        print(f"📁 커스텀 비디오 데이터셋 설정 중... (stage: {stage})")
        
        # 데이터셋 경로 확인
        if not self.root.exists():
            raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {self.root}")
        
        # 훈련 데이터 확인
        train_normal_path = self.root / "train" / "normal"
        if not train_normal_path.exists():
            raise FileNotFoundError(f"훈련 데이터 경로를 찾을 수 없습니다: {train_normal_path}")
        
        # 비디오 파일 목록 생성
        train_videos = list(train_normal_path.glob("*"))
        train_videos = [v for v in train_videos if v.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']]
        
        if len(train_videos) == 0:
            raise FileNotFoundError(f"훈련용 비디오 파일을 찾을 수 없습니다: {train_normal_path}")
        
        print(f"✅ 훈련용 비디오 파일 {len(train_videos)}개 발견")
        for video in train_videos[:5]:  # 처음 5개만 표시
            print(f"   - {video.name}")
        if len(train_videos) > 5:
            print(f"   ... ({len(train_videos)-5}개 더)")
        
        # 검증 데이터 확인 (선택사항)
        val_normal_path = self.root / "val" / "normal"
        val_videos = []
        if val_normal_path.exists():
            val_videos = list(val_normal_path.glob("*"))
            val_videos = [v for v in val_videos if v.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']]
            print(f"✅ 검증용 비디오 파일 {len(val_videos)}개 발견")
        
        # 테스트 데이터 확인 (선택사항)
        test_normal_path = self.root / "test" / "normal"
        test_videos = []
        if test_normal_path.exists():
            test_videos = list(test_normal_path.glob("*"))
            test_videos = [v for v in test_videos if v.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']]
            print(f"✅ 테스트용 비디오 파일 {len(test_videos)}개 발견")
        
        # 이상 데이터 확인 (선택사항)
        train_abnormal_path = self.root / "train" / "abnormal"
        train_abnormal_videos = []
        if train_abnormal_path.exists():
            train_abnormal_videos = list(train_abnormal_path.glob("*"))
            train_abnormal_videos = [v for v in train_abnormal_videos if v.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']]
            print(f"✅ 훈련용 이상 비디오 파일 {len(train_abnormal_videos)}개 발견")
        
        # 데이터셋 생성 (여기서는 간단히 파일 경로만 저장)
        self.train_videos = train_videos
        self.val_videos = val_videos
        self.test_videos = test_videos
        self.train_abnormal_videos = train_abnormal_videos
        
        print("✅ 커스텀 비디오 데이터셋 설정 완료")


def create_sample_dataset_structure(dataset_path: str = "./custom_video_dataset"):
    """샘플 데이터셋 구조 생성"""
    dataset_path = pathlib.Path(dataset_path)
    
    # 디렉토리 구조 생성
    dirs_to_create = [
        dataset_path / "train" / "normal",
        dataset_path / "train" / "abnormal",
        dataset_path / "val" / "normal",
        dataset_path / "val" / "abnormal",
        dataset_path / "test" / "normal",
        dataset_path / "test" / "abnormal",
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        # README 파일 생성
        readme_path = dir_path / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                if "normal" in str(dir_path):
                    f.write("# 정상 비디오 파일들\n\n이 폴더에는 정상적인 비디오 파일들을 저장하세요.\n\n지원되는 형식: .mp4, .avi, .mov, .mkv, .flv, .wmv")
                else:
                    f.write("# 이상 비디오 파일들 (선택사항)\n\n이 폴더에는 이상한 비디오 파일들을 저장하세요.\n\n지원되는 형식: .mp4, .avi, .mov, .mkv, .flv, .wmv")
    
    print(f"✅ 샘플 데이터셋 구조 생성 완료: {dataset_path}")
    print("\n📁 생성된 디렉토리 구조:")
    print("custom_video_dataset/")
    print("├── train/")
    print("│   ├── normal/          # 정상 비디오 파일들")
    print("│   └── abnormal/        # 이상 비디오 파일들 (선택사항)")
    print("├── val/                 # 검증 데이터 (선택사항)")
    print("│   ├── normal/")
    print("│   └── abnormal/")
    print("└── test/                # 테스트 데이터 (선택사항)")
    print("    ├── normal/")
    print("    └── abnormal/")


def main():
    print("🚀 커스텀 비디오 데이터셋으로 AI-VAD 모델 학습 시작...")
    
    # 샘플 데이터셋 구조 생성 (실제 사용 시에는 이 부분을 제거하고 실제 데이터를 준비)
    dataset_path = "./custom_video_dataset"
    create_sample_dataset_structure(dataset_path)
    
    # GPU 설정
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU 사용: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print("CPU 사용")
    
    # 커스텀 데이터 모듈 설정
    print("📁 커스텀 데이터셋 로드 중...")
    try:
        datamodule = CustomVideoDataModule(
            root=dataset_path,
            clip_length_in_frames=2,
            frames_between_clips=1,
            target_frame=VideoTargetFrame.LAST,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=2,
        )
        
        # 데이터셋 설정
        datamodule.setup()
        print("✅ 커스텀 데이터셋 로드 완료")
        
    except Exception as e:
        print(f"❌ 커스텀 데이터셋 로드 실패: {e}")
        print("\n💡 해결 방법:")
        print("1. custom_video_dataset/train/normal/ 폴더에 비디오 파일들을 저장하세요")
        print("2. 지원되는 형식: .mp4, .avi, .mov, .mkv, .flv, .wmv")
        print("3. 이상 데이터가 있다면 custom_video_dataset/train/abnormal/ 폴더에도 저장할 수 있습니다")
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
            max_epochs=3,  # 에포크 수
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
    print("🎬 커스텀 비디오 데이터셋으로 AI-VAD 학습")
    print("=" * 60)
    
    success = main()
    if success:
        print("\n🎉 학습이 성공적으로 완료되었습니다!")
        print("이제 realtime_ui_advanced_windows.py에서 체크포인트를 로드할 수 있습니다.")
        print("\n체크포인트 파일:")
        print("- aivad_custom_checkpoint.ckpt")
    else:
        print("\n💥 학습에 실패했습니다.")
        print("\n📋 체크리스트:")
        print("1. custom_video_dataset/train/normal/ 폴더에 비디오 파일이 있는지 확인")
        print("2. 비디오 파일 형식이 지원되는지 확인 (.mp4, .avi, .mov, .mkv, .flv, .wmv)")
        print("3. 비디오 파일이 손상되지 않았는지 확인")
        exit(1)
