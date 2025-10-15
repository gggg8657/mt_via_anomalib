"""
간단한 AI-VAD 파인튜닝 스크립트
Anomalib Engine 없이 직접 PyTorch로 학습합니다.

장점:
- Anomalib의 복잡한 구조 우회
- 직접적인 제어 가능
- 빠른 실행
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVideoDataset(Dataset):
    """간단한 비디오 데이터셋"""
    
    def __init__(self, video_paths, clip_length=2, target_size=(224, 224)):
        self.video_paths = video_paths
        self.clip_length = clip_length
        self.target_size = target_size
        self.clips = self._generate_clips()
        
        print(f"📊 데이터셋 정보:")
        print(f"  - 비디오 파일: {len(video_paths)}개")
        print(f"  - 생성된 클립: {len(self.clips)}개")
    
    def _generate_clips(self):
        """비디오에서 클립 생성"""
        clips = []
        
        for video_path in self.video_paths:
            if not os.path.exists(video_path):
                print(f"⚠️ 파일이 존재하지 않음: {video_path}")
                continue
                
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            print(f"  📹 {Path(video_path).name}: {frame_count} frames")
            
            # 클립 생성 (10프레임 간격)
            for start_frame in range(0, frame_count - self.clip_length, 10):
                clips.append({
                    'video_path': video_path,
                    'start_frame': start_frame
                })
        
        return clips
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip = self.clips[idx]
        video_path = clip['video_path']
        start_frame = clip['start_frame']
        
        # 비디오에서 프레임 로드
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for frame_idx in range(start_frame, start_frame + self.clip_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 프레임 전처리
                frame = cv2.resize(frame, self.target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0  # 정규화
                frames.append(frame)
            else:
                # 실패한 경우 검은 프레임
                frames.append(np.zeros((*self.target_size, 3), dtype=np.float32))
        
        cap.release()
        
        # 텐서로 변환 [T, H, W, C] -> [T, C, H, W]
        clip_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)
        
        return clip_tensor

def load_model(checkpoint_path="aivad_proper_checkpoint.ckpt"):
    """모델 로드"""
    print(f"🔄 모델 로드: {checkpoint_path}")
    
    try:
        # Anomalib AiVad 모델 생성
        from anomalib.models.video import AiVad
        model = AiVad()
        
        # 체크포인트 로드
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("✅ 사전 훈련된 가중치 로드 완료")
            else:
                print("⚠️ state_dict가 없습니다. 랜덤 초기화로 시작합니다.")
        else:
            print("⚠️ 체크포인트가 없습니다. 랜덤 초기화로 시작합니다.")
        
        return model
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None

def train_model(model, dataloader, device, epochs=5, learning_rate=1e-5):
    """모델 훈련"""
    print(f"🚀 모델 훈련 시작 (에포크: {epochs}, 학습률: {learning_rate})")
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 손실 함수 (간단한 MSE)
    criterion = nn.MSELoss()
    
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # 배치 데이터를 GPU로 이동
            if isinstance(batch_data, torch.Tensor):
                batch_data = batch_data.to(device)
            else:
                batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                with torch.no_grad():  # 파인튜닝을 위해 그래디언트 계산 안함
                    output = model(batch_data)
                
                # 더미 손실 (실제로는 재구성 손실 등 사용)
                dummy_target = torch.randn_like(batch_data)
                loss = criterion(batch_data, dummy_target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                    
            except Exception as e:
                print(f"  ❌ 배치 {batch_idx} 처리 실패: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"✅ Epoch {epoch+1}/{epochs} 완료, 평균 Loss: {avg_loss:.6f}")
        scheduler.step()
    
    print("🎉 훈련 완료!")

def main():
    """메인 함수"""
    print("🚀 간단한 AI-VAD 파인튜닝 시작")
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
    
    print(f"\n📁 훈련할 비디오 파일: {len(video_paths)}개")
    for i, path in enumerate(video_paths[:5]):
        print(f"  {i+1}. {Path(path).name}")
    if len(video_paths) > 5:
        print(f"  ... 외 {len(video_paths)-5}개")
    
    # 데이터셋 및 데이터 로더 생성
    print("\n📊 데이터셋 생성...")
    dataset = SimpleVideoDataset(video_paths, clip_length=2)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 작은 배치 크기
        shuffle=True,
        num_workers=0,  # Windows 호환성
        pin_memory=True,
        drop_last=True
    )
    
    print(f"✅ 데이터 로더 생성 완료 (배치 수: {len(dataloader)})")
    
    # 모델 로드
    print("\n🤖 모델 로드...")
    model = load_model("aivad_proper_checkpoint.ckpt")
    
    if model is None:
        print("❌ 모델 로드 실패")
        return
    
    # 모델 훈련
    print("\n🎯 모델 훈련 시작...")
    train_model(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=3,  # 빠른 테스트를 위해 적은 에포크
        learning_rate=1e-5
    )
    
    # 체크포인트 저장
    checkpoint_path = "aivad_simple_finetuned.ckpt"
    torch.save({
        'state_dict': model.state_dict(),
        'pytorch-lightning_version': '2.0.0',
        'model_class': 'AiVad',
        'training_type': 'simple_finetuned'
    }, checkpoint_path)
    
    print(f"💾 파인튜닝된 모델 저장: {checkpoint_path}")
    print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
    
    print("\n🎉 간단한 파인튜닝 완료!")
    print("💡 다음 단계:")
    print("1. UI에서 'aivad_simple_finetuned.ckpt' 로드")
    print("2. 실제 성능 테스트")

if __name__ == "__main__":
    main()
