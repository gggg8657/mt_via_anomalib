"""
AI-VAD의 올바른 학습 방법 (최종 버전)
image_segments.json의 이미지들을 사용하여 AI-VAD 학습
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

def create_final_dataset_from_json(json_path="image_segments.json", target_dir="final_dataset"):
    """JSON에서 정상 프레임들을 최종 데이터셋으로 변환"""
    print(f"📁 AI-VAD용 최종 데이터셋 생성: {json_path}")
    
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
            for j in range(min(3, len(images) - 1)):  # 각 세그먼트에서 최대 3개 시퀀스
                img_path = images[j]
                
                if os.path.exists(img_path):
                    # 파일명 생성 (더 간단하게)
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
    print("🚀 AI-VAD 올바른 학습 방법 (최종 버전)")
    print("=" * 50)
    print("💡 핵심 원리:")
    print("   1. Feature Extraction: Flow, Region, Pose, Deep features")
    print("   2. Density Estimation: 정상 데이터의 분포 학습")
    print("   3. One-Class Learning: 정상 데이터만으로 분포 모델링")
    print("   4. No NN Training: 가중치 학습 없음!")
    print("=" * 50)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 1. 최종 데이터셋 생성
    dataset_root = create_final_dataset_from_json()
    if dataset_root is None:
        print("❌ 최종 데이터셋 생성 실패")
        return
    
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
            name="final_frames",
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
    
    # 6. AI-VAD 학습 (올바른 방법)
    print(f"\n🎯 AI-VAD 학습 시작...")
    print("💡 학습 과정:")
    print("   1. Feature Extraction: Flow, Region, Pose, Deep features")
    print("   2. Density Update: 정상 특성들을 density estimator에 누적")
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
    checkpoint_path = "aivad_final_learned.ckpt"
    try:
        # AI-VAD 모델 상태 저장
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'final_density_estimation',
            'total_detections': model.total_detections,
        }, checkpoint_path)
        
        print(f"💾 학습된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    print("\n🎉 AI-VAD 올바른 학습 완료!")
    print("💡 학습된 내용:")
    print("1. 정상 데이터의 Feature 분포 학습")
    print("2. Density Estimator로 이상 탐지 준비")
    print("3. UI에서 'aivad_final_learned.ckpt' 로드하여 테스트")
    print("4. 비정상 데이터는 분포에서 벗어나 높은 점수")

if __name__ == "__main__":
    main()
