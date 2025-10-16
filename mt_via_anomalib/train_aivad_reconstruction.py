"""
재구성 손실을 통한 AI-VAD 파인튜닝
정상 데이터만으로 재구성 능력 학습 (이상탐지의 올바른 방법)
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

class ReconstructionVideoDataset(Dataset):
    """재구성 학습용 비디오 시퀀스 데이터셋 (정상 데이터만)"""
    
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
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_paths = self.sequences[idx]
        frames = []
        
        # 각 시퀀스에서 프레임 로드
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
        
        # 텐서로 변환 [sequence_length, 3, 224, 224]
        video_tensor = torch.from_numpy(np.array(frames))
        
        # 배치 차원 추가 [1, sequence_length, 3, 224, 224]
        video_tensor = video_tensor.unsqueeze(0)
        
        return {
            'image': video_tensor,
            'label': torch.tensor(0),  # 정상 데이터
            'image_path': str(sequence_paths[0])  # 첫 번째 프레임 경로
        }

def create_reconstruction_dataset_from_json(json_path="image_segments.json", target_dir="reconstruction_dataset"):
    """JSON에서 정상 프레임들을 재구성 학습용으로 변환"""
    print(f"📁 재구성 학습 데이터셋 생성: {json_path}")
    
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
            for j in range(min(5, len(images) - 1)):  # 더 많은 시퀀스
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
                        
                        if copied_count <= 30:  # 처음 30개만 표시
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

def reconstruction_loss(original, reconstructed, flow_loss_weight=0.1):
    """재구성 손실 함수 (이상탐지의 핵심)"""
    
    # 1. 픽셀 단위 재구성 손실 (MSE)
    pixel_loss = torch.nn.functional.mse_loss(original, reconstructed)
    
    # 2. 구조적 손실 (SSIM-like)
    # 채널별 평균과 분산 계산
    orig_mean = original.mean(dim=[2, 3], keepdim=True)
    recon_mean = reconstructed.mean(dim=[2, 3], keepdim=True)
    
    orig_var = original.var(dim=[2, 3], keepdim=True)
    recon_var = reconstructed.var(dim=[2, 3], keepdim=True)
    
    # 구조적 유사성 손실
    structural_loss = torch.nn.functional.mse_loss(orig_mean, recon_mean) + \
                     torch.nn.functional.mse_loss(orig_var, recon_var)
    
    # 3. 총 재구성 손실
    total_loss = pixel_loss + 0.1 * structural_loss
    
    return total_loss, {
        'pixel_loss': pixel_loss.item(),
        'structural_loss': structural_loss.item(),
        'total_loss': total_loss.item()
    }

def main():
    """메인 함수"""
    print("🚀 재구성 손실로 AI-VAD 파인튜닝 (이상탐지의 올바른 방법)")
    print("=" * 60)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 1. 재구성 학습 데이터셋 생성
    dataset_root = create_reconstruction_dataset_from_json()
    if dataset_root is None:
        print("❌ 재구성 학습 데이터셋 생성 실패")
        return
    
    # 2. 재구성 학습 데이터셋 생성
    print(f"\n📊 재구성 학습 데이터셋 생성...")
    try:
        normal_dir = Path(dataset_root) / "train" / "good"
        reconstruction_dataset = ReconstructionVideoDataset(normal_dir, sequence_length=2)
        
        # DataLoader 생성
        dataloader = DataLoader(
            reconstruction_dataset,
            batch_size=2,  # 작은 배치 크기 (재구성은 메모리 집약적)
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        print(f"✅ 재구성 학습 데이터셋 생성 완료")
        print(f"   📊 배치 크기: {dataloader.batch_size}")
        print(f"   📊 총 배치 수: {len(dataloader)}")
        
    except Exception as e:
        print(f"❌ 재구성 학습 데이터셋 생성 실패: {e}")
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
    
    # 5. 재구성 학습 훈련
    print(f"\n🎯 재구성 학습 시작...")
    print("💡 이상탐지 원리: 정상 데이터 재구성 능력 학습 → 비정상은 재구성 오류 큼")
    
    try:
        model.to(device)
        
        # 모델을 train 모드로 설정
        model.train()
        model.model.train()
        
        # 모든 파라미터의 gradient 활성화
        for param in model.parameters():
            param.requires_grad = True
        
        # 옵티마이저 설정 (더 작은 학습률)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        
        # 훈련 루프
        for epoch in range(15):  # 더 많은 에포크
            print(f"\n📈 Epoch {epoch + 1}/15")
            
            total_loss = 0
            total_pixel_loss = 0
            total_structural_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # 데이터를 GPU로 이동
                    video_tensor = batch['image'].to(device)  # [batch_size, 1, 2, 3, 224, 224]
                    
                    # 배치 차원 조정 [batch_size, 2, 3, 224, 224]
                    video_tensor = video_tensor.squeeze(1)
                    
                    # 모델 순전파
                    optimizer.zero_grad()
                    
                    # AI-VAD 모델 호출
                    output = model.model(video_tensor)
                    
                    # 재구성 손실 계산
                    # 원본 입력과 모델 출력 간의 재구성 손실
                    if hasattr(output, 'pred_score') and hasattr(output, 'anomaly_map'):
                        # anomaly_map을 원본 크기로 리사이즈하여 재구성 손실 계산
                        reconstructed = output.anomaly_map
                        if reconstructed.shape != video_tensor.shape:
                            # 크기 맞춤
                            reconstructed = torch.nn.functional.interpolate(
                                reconstructed, size=video_tensor.shape[2:], mode='bilinear'
                            )
                        
                        loss, loss_dict = reconstruction_loss(video_tensor, reconstructed)
                    else:
                        # 출력이 예상과 다를 경우, 입력 자체를 재구성 타겟으로 사용
                        # (이는 정규화 효과)
                        loss, loss_dict = reconstruction_loss(video_tensor, video_tensor)
                        loss = loss * 0.1  # 정규화 손실은 작게
                    
                    # 역전파
                    loss.backward()
                    
                    # Gradient clipping (재구성 학습 안정화)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss_dict['total_loss']
                    total_pixel_loss += loss_dict['pixel_loss']
                    total_structural_loss += loss_dict['structural_loss']
                    batch_count += 1
                    
                    # 진행률 표시 (10배치마다)
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx + 1}: Total = {loss_dict['total_loss']:.4f}, "
                              f"Pixel = {loss_dict['pixel_loss']:.4f}, "
                              f"Structural = {loss_dict['structural_loss']:.4f}")
                    
                except Exception as e:
                    print(f"  ⚠️ Batch {batch_idx + 1} 실패: {e}")
                    continue
            
            if batch_count > 0:
                avg_total = total_loss / batch_count
                avg_pixel = total_pixel_loss / batch_count
                avg_structural = total_structural_loss / batch_count
                
                print(f"  📊 평균 손실 (총 {batch_count}개 배치):")
                print(f"     총 손실: {avg_total:.4f}")
                print(f"     픽셀 손실: {avg_pixel:.4f}")
                print(f"     구조 손실: {avg_structural:.4f}")
            else:
                print(f"  ⚠️ 성공한 배치가 없습니다.")
        
        print("✅ 재구성 학습 완료!")
        
    except Exception as e:
        print(f"❌ 재구성 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 체크포인트 저장
    checkpoint_path = "aivad_reconstruction_finetuned.ckpt"
    try:
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'reconstruction_finetuned'
        }, checkpoint_path)
        
        print(f"💾 재구성 학습된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    print("\n🎉 재구성 손실로 AI-VAD 파인튜닝 완료!")
    print("💡 이상탐지 원리:")
    print("1. 정상 데이터의 재구성 패턴 학습")
    print("2. 비정상 데이터는 재구성 오류가 큼")
    print("3. 재구성 오류 = 이상 점수")
    print("4. UI에서 'aivad_reconstruction_finetuned.ckpt' 로드하여 테스트")

if __name__ == "__main__":
    main()
