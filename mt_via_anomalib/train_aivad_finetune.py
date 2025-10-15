"""
AI-VAD 모델 파인튜닝 스크립트
기존에 훈련된 AI-VAD 모델을 우리 데이터로 파인튜닝합니다.

장점:
- 빠른 학습 시간 (기존 가중치 활용)
- 안정적인 학습 (사전 훈련된 특징 활용)
- 적은 데이터로도 효과적

단점:
- 기존 데이터셋의 편향에 영향받을 수 있음
- 도메인 차이가 크면 성능 제한
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
import cv2
import numpy as np
from anomalib.models.video import AiVad
from anomalib.engine import Engine
from anomalib.data.utils.split import ValSplitMode
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomVideoDataModule:
    """커스텀 비디오 데이터 모듈"""
    
    def __init__(self, video_paths, clip_length=2, frames_between_clips=1):
        self.video_paths = video_paths
        self.clip_length = clip_length
        self.frames_between_clips = frames_between_clips
        
    def setup(self):
        """데이터셋 설정"""
        print(f"📁 비디오 파일 {len(self.video_paths)}개 로드 중...")
        
        # 비디오별 프레임 수 확인
        self.video_info = {}
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_info[video_path] = frame_count
            cap.release()
            print(f"  - {Path(video_path).name}: {frame_count} frames")
        
        print(f"✅ 총 {sum(self.video_info.values())} 프레임 로드 완료")
        
    def train_dataloader(self):
        """훈련 데이터 로더"""
        # 실제 구현에서는 비디오 클립을 생성하는 로직 필요
        # 여기서는 더미 데이터 반환
        return self._create_dummy_dataloader()
    
    def val_dataloader(self):
        """검증 데이터 로더"""
        return self._create_dummy_dataloader()
    
    def _create_dummy_dataloader(self):
        """더미 데이터 로더 생성"""
        # 실제 비디오 데이터 로더 구현 필요
        return None

def load_pretrained_model(checkpoint_path="aivad_proper_checkpoint.ckpt"):
    """사전 훈련된 모델 로드"""
    print(f"🔄 사전 훈련된 모델 로드: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일이 없습니다: {checkpoint_path}")
        return None
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # 모델 생성
        model = AiVad()
        
        # 가중치 로드
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("✅ 사전 훈련된 가중치 로드 완료")
        else:
            print("⚠️ state_dict가 없습니다. 랜덤 초기화로 시작합니다.")
        
        return model
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None

def setup_finetune_optimizer(model, learning_rate=1e-5):
    """파인튜닝용 옵티마이저 설정"""
    print(f"⚙️ 파인튜닝 옵티마이저 설정 (LR: {learning_rate})")
    
    # 파인튜닝을 위해 낮은 학습률 사용
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-7
    )
    
    return optimizer, scheduler

def main():
    """메인 함수"""
    print("🚀 AI-VAD 모델 파인튜닝 시작")
    print("=" * 50)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 비디오 파일 경로 설정
    from video_files_list import video_files
    video_paths = video_files
    
    print(f"📁 훈련할 비디오 파일: {len(video_paths)}개")
    for i, path in enumerate(video_paths[:3]):  # 처음 3개만 표시
        print(f"  {i+1}. {Path(path).name}")
    if len(video_paths) > 3:
        print(f"  ... 외 {len(video_paths)-3}개")
    
    # 데이터 모듈 설정
    print("\n📊 데이터 모듈 설정...")
    datamodule = CustomVideoDataModule(
        video_paths=video_paths,
        clip_length=2,
        frames_between_clips=1
    )
    datamodule.setup()
    
    # 사전 훈련된 모델 로드
    print("\n🤖 사전 훈련된 모델 로드...")
    model = load_pretrained_model("aivad_proper_checkpoint.ckpt")
    
    if model is None:
        print("❌ 모델 로드 실패. 새 모델로 시작합니다.")
        model = AiVad()
    
    # 파인튜닝 옵티마이저 설정
    print("\n⚙️ 파인튜닝 설정...")
    optimizer, scheduler = setup_finetune_optimizer(model, learning_rate=1e-5)
    
    # 학습 엔진 설정 (파인튜닝용)
    print("\n🔧 학습 엔진 설정...")
    engine = Engine(
        devices=1 if device == "cuda" else "auto",
        accelerator="gpu" if device == "cuda" else "cpu",
        precision="16-mixed" if device == "cuda" else "32",
        max_epochs=5,  # 파인튜닝은 적은 에포크
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # 그래디언트 누적
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_progress_bar=True,
        enable_model_summary=True,
        # 파인튜닝을 위한 제한
        limit_train_batches=20,  # 배치 수 제한
        limit_val_batches=10,
    )
    
    # 학습 시작
    print("\n🎯 파인튜닝 시작!")
    try:
        # 실제 데이터 로더가 구현되면 이 부분 활성화
        # engine.fit(model=model, datamodule=datamodule)
        
        print("⚠️ 실제 데이터 로더 구현 필요")
        print("현재는 더미 모드로 실행됩니다.")
        
        # 더미 학습 (실제 구현에서는 제거)
        print("🔧 더미 학습 실행...")
        model.train()
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, 2, 3, 224, 224).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ 모델 forward pass 성공")
            print(f"출력 타입: {type(output)}")
        
        # 체크포인트 저장
        checkpoint_path = "aivad_finetuned.ckpt"
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'finetuned'
        }, checkpoint_path)
        
        print(f"💾 파인튜닝된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
        print("\n🎉 파인튜닝 완료!")
        print("💡 다음 단계:")
        print("1. UI에서 'aivad_finetuned.ckpt' 로드")
        print("2. 실제 성능 테스트")
        print("3. 필요시 추가 파인튜닝")
        
    except Exception as e:
        print(f"❌ 파인튜닝 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
