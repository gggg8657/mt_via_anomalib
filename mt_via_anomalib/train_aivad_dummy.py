"""
AI-VAD 더미 데이터로 학습 테스트
실제 이미지 없이 더미 이미지로 AI-VAD 학습 과정 테스트
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models.video import AiVad
from anomalib.engine import Engine
from anomalib.data import Folder

def create_dummy_dataset(target_dir="dummy_dataset"):
    """더미 이미지 데이터셋 생성"""
    print(f"📁 더미 이미지 데이터셋 생성: {target_dir}")
    
    # 폴더 생성
    normal_dir = Path(target_dir) / "train" / "good"
    normal_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📂 더미 이미지 폴더: {normal_dir}")
    
    # 더미 이미지 생성
    dummy_count = 50  # 50개 더미 이미지
    
    for i in range(dummy_count):
        # 다양한 패턴의 더미 이미지 생성
        if i % 4 == 0:
            # 패턴 1: 단색 배경
            img = np.ones((224, 224, 3), dtype=np.uint8) * (i * 5 % 255)
        elif i % 4 == 1:
            # 패턴 2: 그라데이션
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            for y in range(224):
                img[y, :, 0] = int(255 * y / 224)  # 빨간색 그라데이션
        elif i % 4 == 2:
            # 패턴 3: 원형 패턴
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            center = (112, 112)
            cv2.circle(img, center, 50 + i, (0, 255, 0), -1)
        else:
            # 패턴 4: 직선 패턴
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            for x in range(0, 224, 20):
                cv2.line(img, (x, 0), (x, 224), (0, 0, 255), 2)
        
        # 파일명 생성
        img_name = f"dummy_normal_{i:03d}.jpg"
        img_path = normal_dir / img_name
        
        # 이미지 저장
        cv2.imwrite(str(img_path), img)
    
    print(f"  ✅ 생성된 더미 이미지: {dummy_count}개")
    
    # 처음 몇 개 이미지 표시
    for i in range(min(5, dummy_count)):
        print(f"    📸 dummy_normal_{i:03d}.jpg")
    
    return str(Path(target_dir).absolute())

def main():
    """메인 함수"""
    print("🚀 AI-VAD 더미 데이터 학습 테스트")
    print("=" * 50)
    print("💡 목적:")
    print("   1. AI-VAD 학습 과정 테스트")
    print("   2. Density Estimation 동작 확인")
    print("   3. Feature Extraction 테스트")
    print("   4. 실제 데이터 없이 학습 과정 검증")
    print("=" * 50)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 1. 더미 데이터셋 생성
    dataset_root = create_dummy_dataset()
    
    # 2. AI-VAD 모델 생성
    print(f"\n🤖 AI-VAD 모델 생성...")
    try:
        model = AiVad(
            # Feature 추출 설정
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
    
    # 3. 사전 훈련된 가중치 로드 (선택사항)
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
    
    # 5. Folder 데이터 모듈 생성
    print(f"\n📊 Folder 데이터 모듈 생성...")
    try:
        datamodule = Folder(
            name="dummy_frames",
            root=dataset_root,
            normal_dir="train/good",
            train_batch_size=1,  # 작은 배치 크기
            eval_batch_size=1,
            num_workers=0,
        )
        
        print("✅ Folder 데이터 모듈 생성 완료")
        
    except Exception as e:
        print(f"❌ Folder 데이터 모듈 생성 실패: {e}")
        return
    
    # 6. AI-VAD 학습 (더미 데이터로 테스트)
    print(f"\n🎯 AI-VAD 학습 시작 (더미 데이터)...")
    print("💡 학습 과정:")
    print("   1. Feature Extraction: Flow, Region, Pose, Deep features")
    print("   2. Density Update: 더미 특성들을 density estimator에 누적")
    print("   3. Density Fit: 모든 특성으로 분포 모델 학습")
    print("   4. No Backpropagation: 가중치 업데이트 없음!")
    
    try:
        # AI-VAD의 올바른 학습 방법
        engine.fit(model=model, datamodule=datamodule)
        
        print("✅ AI-VAD 학습 완료!")
        
        # Density estimator 상태 확인
        if hasattr(model.model, 'density_estimator'):
            print(f"📊 Density Estimator 상태:")
            print(f"   - 총 감지 수: {model.total_detections}")
            
            # Density estimator fit 호출
            if model.total_detections > 0:
                model.fit()  # density estimator 학습
                print("✅ Density Estimator 학습 완료")
            else:
                print("⚠️ 감지된 영역이 없습니다.")
        
    except Exception as e:
        print(f"❌ AI-VAD 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 체크포인트 저장
    checkpoint_path = "aivad_dummy_learned.ckpt"
    try:
        # AI-VAD 모델 상태 저장
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'dummy_density_estimation',
            'total_detections': model.total_detections,
        }, checkpoint_path)
        
        print(f"💾 학습된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    print("\n🎉 AI-VAD 더미 데이터 학습 완료!")
    print("💡 학습된 내용:")
    print("1. 더미 데이터의 Feature 분포 학습")
    print("2. Density Estimator 동작 확인")
    print("3. AI-VAD 학습 과정 검증 완료")
    print("4. UI에서 'aivad_dummy_learned.ckpt' 로드하여 테스트")

if __name__ == "__main__":
    main()
