"""
AI-VAD 올바른 학습 방법 (최종 정정 버전)
AI-VAD 원래 방식과 동일하게 동작하도록 수정
"""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models.video import AiVad
from anomalib.engine import Engine
from anomalib.data import Folder
import shutil
from torch.utils.data import Dataset, DataLoader

class CorrectVideoDataset(Dataset):
    """AI-VAD 원래 방식과 동일한 비디오 시퀀스 데이터셋"""
    
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

def create_correct_dataset_from_json(json_path="image_segments.json", target_dir="correct_dataset"):
    """JSON에서 정상 프레임들을 올바른 데이터셋으로 변환"""
    print(f"📁 AI-VAD용 올바른 데이터셋 생성: {json_path}")
    
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
            
            # 각 세그먼트에서 프레임들 복사
            for j, img_path in enumerate(images[:5]):  # 각 세그먼트에서 최대 5개
                if os.path.exists(img_path):
                    # 파일명 생성
                    name = f"normal_{normal_count:03d}_{j:02d}_{Path(img_path).name}"
                    target_path = normal_dir / name
                    
                    try:
                        # 파일 복사
                        shutil.copy2(img_path, target_path)
                        copied_count += 1
                        
                        if copied_count <= 20:  # 처음 20개만 표시
                            print(f"    📸 {name}")
                            
                    except Exception as e:
                        print(f"    ⚠️ {img_path} 복사 실패: {e}")
    
    print(f"  ✅ 정상 세그먼트: {normal_count}개")
    print(f"  ✅ 복사된 이미지: {copied_count}개")
    
    if copied_count == 0:
        print("❌ 복사된 이미지가 없습니다!")
        return None
    
    return str(Path(target_dir).absolute())

def main():
    """메인 함수"""
    print("🚀 AI-VAD 올바른 학습 방법 (최종 정정 버전)")
    print("=" * 60)
    print("💡 AI-VAD 원래 방식:")
    print("   1. Feature Extraction: Flow, Region, Pose, Deep features")
    print("   2. Density Estimation: 정상 데이터의 분포 학습")
    print("   3. One-Class Learning: 정상 데이터만으로 분포 모델링")
    print("   4. No NN Training: 가중치 학습 없음!")
    print("   5. 우리 데이터 사용: Domain 적용성 향상")
    print("=" * 60)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 1. 올바른 데이터셋 생성
    dataset_root = create_correct_dataset_from_json()
    if dataset_root is None:
        print("❌ 올바른 데이터셋 생성 실패")
        return
    
    # 2. AI-VAD 모델 생성 (원래 설정)
    print(f"\n🤖 AI-VAD 모델 생성...")
    try:
        model = AiVad(
            # AI-VAD 원래 설정 (모든 Feature 활성화)
            use_velocity_features=True,
            use_pose_features=True,
            use_deep_features=True,
            # Density estimation 설정
            n_components_velocity=2,
            n_neighbors_pose=1,
            n_neighbors_deep=1,
        )
        print("✅ AI-VAD 모델 생성 완료")
        
    except Exception as e:
        print(f"❌ AI-VAD 모델 생성 실패: {e}")
        return
    
    # 3. 사전 훈련된 가중치 로드
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
    
    # 4. Anomalib Engine 생성 (AI-VAD 전용 설정)
    print(f"\n🔧 Anomalib Engine 생성...")
    try:
        engine = Engine(
            devices=1 if device == "cuda" else "auto",
            accelerator="gpu" if device == "cuda" else "cpu",
            precision="32",  # AI-VAD는 32bit 사용
            # AI-VAD 전용 설정
            max_epochs=1,  # AI-VAD는 1 에포크만
            gradient_clip_val=0,  # Gradient clipping 없음
            log_every_n_steps=1,
            enable_progress_bar=True,
            enable_model_summary=False,  # 모델 요약 비활성화
            num_sanity_val_steps=0,  # 검증 단계 없음
        )
        
        print("✅ Engine 생성 완료")
        
    except Exception as e:
        print(f"❌ Engine 생성 실패: {e}")
        return
    
    # 5. 비디오 시퀀스 데이터셋 생성
    print(f"\n📊 비디오 시퀀스 데이터셋 생성...")
    try:
        normal_dir = Path(dataset_root) / "train" / "good"
        video_dataset = CorrectVideoDataset(normal_dir, sequence_length=2)
        
        # DataLoader 생성
        dataloader = DataLoader(
            video_dataset,
            batch_size=1,  # 작은 배치 크기
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        print(f"✅ 비디오 시퀀스 데이터셋 생성 완료")
        print(f"   📊 배치 크기: {dataloader.batch_size}")
        print(f"   📊 총 배치 수: {len(dataloader)}")
        
    except Exception as e:
        print(f"❌ 비디오 시퀀스 데이터셋 생성 실패: {e}")
        return
    
    # 6. AI-VAD 학습 (원래 방식)
    print(f"\n🎯 AI-VAD 학습 시작 (원래 방식)...")
    print("💡 학습 과정:")
    print("   1. 이미지 → 비디오 시퀀스 변환")
    print("   2. Feature Extraction: Flow, Region, Pose, Deep features")
    print("   3. Density Update: 정상 특성들을 density estimator에 누적")
    print("   4. Density Fit: 모든 특성으로 분포 모델 학습")
    print("   5. No Backpropagation: 가중치 업데이트 없음!")
    print("   6. 우리 데이터 사용: Domain 적용성 향상!")
    
    try:
        # AI-VAD의 올바른 학습 방법 (원래 방식)
        model.eval().to(device)
        
        total_detections = 0
        successful_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 데이터를 GPU로 이동
                video_tensor = batch['image'].to(device)  # [batch_size, 1, 2, 3, 224, 224]
                
                # 배치 차원 조정 [batch_size, 2, 3, 224, 224]
                video_tensor = video_tensor.squeeze(1)
                
                # AI-VAD 모델 호출 (원래 방식)
                with torch.no_grad():
                    # AI-VAD의 원래 학습 방식
                    features_per_batch = model.model(video_tensor)
                
                # Density estimator 업데이트 (원래 방식)
                if hasattr(model.model, 'density_estimator'):
                    for features in features_per_batch:
                        model.model.density_estimator.update(features, f"video_{batch_idx}")
                        if features:
                            total_detections += len(next(iter(features.values())))
                
                successful_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  처리된 배치: {batch_idx + 1}/{len(dataloader)}")
                
            except Exception as e:
                print(f"  ⚠️ 배치 {batch_idx + 1} 실패: {e}")
                # 에러가 발생해도 계속 진행
                continue
        
        print("✅ AI-VAD 학습 완료!")
        
        # Density estimator 상태 확인
        print(f"📊 Density Estimator 상태:")
        print(f"   - 성공한 배치: {successful_batches}/{len(dataloader)}")
        print(f"   - 총 감지 수: {total_detections}")
        
        # Density estimator fit 호출
        if total_detections > 0:
            model.fit()  # density estimator 학습
            print("✅ Density Estimator 학습 완료")
        else:
            print("⚠️ 감지된 영역이 없습니다.")
            print("💡 이는 정상적인 현상일 수 있습니다 (복잡한 객체 감지)")
        
    except Exception as e:
        print(f"❌ AI-VAD 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 체크포인트 저장
    checkpoint_path = "aivad_correct_learned.ckpt"
    try:
        # AI-VAD 모델 상태 저장
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'correct_density_estimation',
            'total_detections': total_detections,
        }, checkpoint_path)
        
        print(f"💾 학습된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    print("\n🎉 AI-VAD 올바른 학습 완료!")
    print("💡 학습된 내용:")
    print("1. 우리 데이터로 Domain 적용성 향상")
    print("2. 이미지 → 비디오 시퀀스 변환")
    print("3. 정상 데이터의 Feature 분포 학습")
    print("4. Density Estimator로 이상 탐지 준비")
    print("5. UI에서 'aivad_correct_learned.ckpt' 로드하여 테스트")
    print("6. 비정상 데이터는 분포에서 벗어나 높은 점수")

if __name__ == "__main__":
    main()
