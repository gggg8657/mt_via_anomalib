"""
대조 학습을 통한 AI-VAD 파인튜닝
정상 데이터 + 생성된 비정상 데이터로 대조 학습
"""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models.video import AiVad
import shutil
from torch.utils.data import Dataset, DataLoader
import random

class ContrastiveVideoDataset(Dataset):
    """정상 + 비정상 비디오 시퀀스 대조 학습 데이터셋"""
    
    def __init__(self, image_folder, sequence_length=2, transform=None):
        self.image_folder = Path(image_folder)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # 모든 이미지 파일 수집
        self.image_files = sorted([f for f in self.image_folder.glob("*.jpg")])
        print(f"📸 총 {len(self.image_files)}개 이미지 발견")
        
        # 시퀀스 생성 (연속된 프레임들)
        self.sequences = []
        for i in range(len(self.image_files) - sequence_length + 1):
            sequence = self.image_files[i:i + sequence_length]
            self.sequences.append(sequence)
        
        print(f"🎬 {len(self.sequences)}개 비디오 시퀀스 생성")
        
        # 정상/비정상 라벨 생성 (50:50 비율)
        self.labels = []
        self.data_pairs = []
        
        for i, sequence in enumerate(self.sequences):
            # 정상 데이터 (원본)
            normal_frames = self._load_sequence(sequence)
            self.data_pairs.append((normal_frames, 0))  # 0 = 정상
            
            # 비정상 데이터 생성 (노이즈, 회전, 밝기 변화 등)
            if len(self.image_files) > sequence_length:
                # 다른 프레임들로 랜덤 시퀀스 생성 (비정상)
                anomaly_sequence = random.sample(self.image_files, sequence_length)
                anomaly_frames = self._load_sequence(anomaly_sequence)
                self.data_pairs.append((anomaly_frames, 1))  # 1 = 비정상
        
        print(f"🎯 총 {len(self.data_pairs)}개 데이터 페어 생성")
        print(f"   📊 정상: {sum(1 for _, label in self.data_pairs if label == 0)}개")
        print(f"   📊 비정상: {sum(1 for _, label in self.data_pairs if label == 1)}개")
    
    def _load_sequence(self, sequence_paths):
        """시퀀스 로드 및 전처리"""
        frames = []
        
        for path in sequence_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                # 빈 프레임 생성
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # 224x224로 리사이즈
                frame = cv2.resize(frame, (224, 224))
            
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 정규화 [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            # CHW 형태로 변환
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)
        
        return np.array(frames)
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        frames, label = self.data_pairs[idx]
        
        # 텐서로 변환 [sequence_length, 3, 224, 224]
        video_tensor = torch.from_numpy(frames)
        
        # 배치 차원 추가 [1, sequence_length, 3, 224, 224]
        video_tensor = video_tensor.unsqueeze(0)
        
        return {
            'image': video_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'image_path': str(self.image_files[0]) if len(self.image_files) > 0 else "dummy"
        }

def create_contrastive_dataset_from_json(json_path="image_segments.json", target_dir="contrastive_dataset"):
    """JSON에서 정상 프레임들을 대조 학습용으로 변환"""
    print(f"📁 대조 학습 데이터셋 생성: {json_path}")
    
    # 폴더 생성
    normal_dir = Path(target_dir) / "train" / "good"
    normal_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📂 정상 이미지 폴더: {normal_dir}")
    
    # JSON 파일 로드
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        print(f"  📊 JSON 로드 완료: {len(segments)}개 세그먼트")
    except Exception as e:
        print(f"❌ JSON 로드 실패: {e}")
        return None
    
    # 정상 프레임들 추출
    copied_count = 0
    normal_count = 0
    
    for i, segment in enumerate(segments):
        if segment.get('category') == 'normal' and 'images' in segment:
            normal_count += 1
            images = segment['images']
            
            # 각 세그먼트에서 연속된 프레임들 사용
            for j in range(min(3, len(images) - 1)):  # 최대 3개 시퀀스
                img1_path = images[j]
                img2_path = images[j + 1]
                
                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    # 파일명 생성
                    name1 = f"seq_{normal_count:03d}_{j:02d}_frame1_{Path(img1_path).name}"
                    name2 = f"seq_{normal_count:03d}_{j:02d}_frame2_{Path(img2_path).name}"
                    
                    target1 = normal_dir / name1
                    target2 = normal_dir / name2
                    
                    try:
                        # 파일 복사
                        shutil.copy2(img1_path, target1)
                        shutil.copy2(img2_path, target2)
                        copied_count += 2
                        
                        if copied_count <= 20:  # 처음 20개만 표시
                            print(f"    🎬 {name1}")
                            print(f"    🎬 {name2}")
                            
                    except Exception as e:
                        print(f"    ⚠️ {img1_path} 복사 실패: {e}")
    
    print(f"  ✅ 정상 세그먼트: {normal_count}개")
    print(f"  ✅ 복사된 이미지: {copied_count}개")
    
    if copied_count == 0:
        print("❌ 복사된 이미지가 없습니다!")
        return None
    
    return str(Path(target_dir).absolute())

def contrastive_loss(pred_scores, labels, margin=0.5):
    """대조 학습 손실 함수"""
    # 정상 데이터는 낮은 점수, 비정상 데이터는 높은 점수 목표
    target_scores = torch.where(labels == 0, torch.tensor(0.1), torch.tensor(0.9))
    
    # MSE 손실
    mse_loss = torch.nn.functional.mse_loss(pred_scores.squeeze(), target_scores)
    
    # 대조 손실 (정상과 비정상 간 거리)
    normal_scores = pred_scores[labels == 0]
    anomaly_scores = pred_scores[labels == 1]
    
    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        contrastive_loss = torch.nn.functional.relu(margin - (anomaly_scores.mean() - normal_scores.mean()))
        return mse_loss + contrastive_loss
    else:
        return mse_loss

def main():
    """메인 함수"""
    print("🚀 대조 학습으로 AI-VAD 파인튜닝")
    print("=" * 50)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 1. 대조 학습 데이터셋 생성
    dataset_root = create_contrastive_dataset_from_json()
    if dataset_root is None:
        print("❌ 대조 학습 데이터셋 생성 실패")
        return
    
    # 2. 대조 학습 데이터셋 생성
    print(f"\n📊 대조 학습 데이터셋 생성...")
    try:
        normal_dir = Path(dataset_root) / "train" / "good"
        contrastive_dataset = ContrastiveVideoDataset(normal_dir, sequence_length=2)
        
        # DataLoader 생성
        dataloader = DataLoader(
            contrastive_dataset,
            batch_size=4,  # 더 큰 배치 크기
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        print(f"✅ 대조 학습 데이터셋 생성 완료")
        print(f"   📊 배치 크기: {dataloader.batch_size}")
        print(f"   📊 총 배치 수: {len(dataloader)}")
        
    except Exception as e:
        print(f"❌ 대조 학습 데이터셋 생성 실패: {e}")
        return
    
    # 3. AI-VAD 모델 생성
    print(f"\n🤖 AI-VAD 모델 생성...")
    try:
        model = AiVad()
        print("✅ AI-VAD 모델 생성 완료")
        
    except Exception as e:
        print(f"❌ AI-VAD 모델 생성 실패: {e}")
        return
    
    # 4. 사전 훈련된 가중치 로드
    checkpoint_path = "aivad_proper_checkpoint.ckpt"
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 사전 훈련된 가중치 로드: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("✅ 사전 훈련된 가중치 로드 완료")
            else:
                print("⚠️ state_dict가 없습니다.")
        except Exception as e:
            print(f"⚠️ 가중치 로드 실패: {e}")
    
    # 5. 대조 학습 훈련
    print(f"\n🎯 대조 학습 시작...")
    try:
        model.to(device)
        
        # 모델을 train 모드로 설정
        model.train()
        model.model.train()
        
        # 모든 파라미터의 gradient 활성화
        for param in model.parameters():
            param.requires_grad = True
        
        # 옵티마이저 설정
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 훈련 루프
        for epoch in range(5):  # 5 에포크
            print(f"\n📈 Epoch {epoch + 1}/5")
            
            total_loss = 0
            batch_count = 0
            normal_count = 0
            anomaly_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # 데이터를 GPU로 이동
                    video_tensor = batch['image'].to(device)  # [batch_size, 1, 2, 3, 224, 224]
                    labels = batch['label'].to(device)  # [batch_size]
                    
                    # 배치 차원 조정 [batch_size, 2, 3, 224, 224]
                    video_tensor = video_tensor.squeeze(1)
                    
                    # 모델 순전파
                    optimizer.zero_grad()
                    
                    # AI-VAD 모델 호출
                    output = model.model(video_tensor)
                    
                    # 출력에서 pred_score 추출
                    if hasattr(output, 'pred_score'):
                        pred_scores = output.pred_score
                    elif isinstance(output, dict) and 'pred_score' in output:
                        pred_scores = output['pred_score']
                    elif torch.is_tensor(output):
                        pred_scores = output
                    else:
                        print(f"    ⚠️ 알 수 없는 출력 형태")
                        continue
                    
                    # 대조 학습 손실 계산
                    loss = contrastive_loss(pred_scores, labels)
                    
                    # 역전파
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # 라벨 통계
                    normal_count += (labels == 0).sum().item()
                    anomaly_count += (labels == 1).sum().item()
                    
                    # 진행률 표시 (5배치마다)
                    if batch_idx % 5 == 0:
                        normal_score = pred_scores[labels == 0].mean().item() if (labels == 0).sum() > 0 else 0
                        anomaly_score = pred_scores[labels == 1].mean().item() if (labels == 1).sum() > 0 else 0
                        print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}, Normal = {normal_score:.3f}, Anomaly = {anomaly_score:.3f}")
                    
                except Exception as e:
                    print(f"  ⚠️ Batch {batch_idx + 1} 실패: {e}")
                    continue
            
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                print(f"  📊 평균 손실: {avg_loss:.4f} (총 {batch_count}개 배치)")
                print(f"  📊 정상 데이터: {normal_count}개, 비정상 데이터: {anomaly_count}개")
            else:
                print(f"  ⚠️ 성공한 배치가 없습니다.")
        
        print("✅ 대조 학습 완료!")
        
    except Exception as e:
        print(f"❌ 대조 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 체크포인트 저장
    checkpoint_path = "aivad_contrastive_finetuned.ckpt"
    try:
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'contrastive_finetuned'
        }, checkpoint_path)
        
        print(f"💾 대조 학습된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    print("\n🎉 대조 학습으로 AI-VAD 파인튜닝 완료!")
    print("💡 다음 단계:")
    print("1. UI에서 'aivad_contrastive_finetuned.ckpt' 로드")
    print("2. 실제 성능 테스트")
    print("3. 정상/비정상 구분 성능 확인")

if __name__ == "__main__":
    main()
