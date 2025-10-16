"""
AI-VAD의 올바른 학습 방법 (간단 버전)
기존 정상 비디오 파일들을 직접 사용
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models.video import AiVad
from anomalib.engine import Engine
from anomalib.data import Folder
import glob

def create_simple_dataset_from_videos():
    """기존 정상 비디오 파일들로 간단한 데이터셋 생성"""
    print("📁 기존 정상 비디오 파일들 검색...")
    
    # 정상 비디오 파일들 찾기
    video_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv']
    normal_videos = []
    
    for ext in video_extensions:
        videos = glob.glob(f"normal_*{ext}")
        normal_videos.extend(videos)
    
    print(f"📊 발견된 정상 비디오: {len(normal_videos)}개")
    
    if len(normal_videos) == 0:
        print("❌ 정상 비디오 파일을 찾을 수 없습니다!")
        return None
    
    # 처음 몇 개만 표시
    for i, video in enumerate(normal_videos[:5]):
        print(f"  🎬 {video}")
    
    if len(normal_videos) > 5:
        print(f"  ... 외 {len(normal_videos) - 5}개")
    
    return normal_videos

def extract_frames_from_videos(video_files, output_dir="extracted_frames"):
    """비디오에서 프레임 추출"""
    print(f"\n📸 비디오에서 프레임 추출: {output_dir}")
    
    # 출력 디렉토리 생성
    frame_dir = Path(output_dir) / "train" / "good"
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    
    for i, video_file in enumerate(video_files[:10]):  # 처음 10개만 처리
        try:
            cap = cv2.VideoCapture(video_file)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 5프레임마다 추출
                if frame_count % 5 == 0:
                    # 224x224로 리사이즈
                    frame_resized = cv2.resize(frame, (224, 224))
                    
                    # 파일명 생성
                    frame_name = f"video_{i:03d}_frame_{frame_count:04d}.jpg"
                    frame_path = frame_dir / frame_name
                    
                    # 프레임 저장
                    cv2.imwrite(str(frame_path), frame_resized)
                    extracted_count += 1
                    
                    # 최대 20프레임만 추출
                    if extracted_count >= 20:
                        break
                
                frame_count += 1
            
            cap.release()
            
            if extracted_count >= 20:
                break
                
        except Exception as e:
            print(f"  ⚠️ {video_file} 처리 실패: {e}")
    
    print(f"✅ 추출된 프레임: {extracted_count}개")
    return str(Path(output_dir).absolute())

def main():
    """메인 함수"""
    print("🚀 AI-VAD 올바른 학습 방법 (간단 버전)")
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
    
    # 1. 기존 정상 비디오 파일들 검색
    normal_videos = create_simple_dataset_from_videos()
    if normal_videos is None:
        print("❌ 정상 비디오 파일을 찾을 수 없습니다!")
        print("💡 normal_*.avi, normal_*.mp4 파일이 있는지 확인하세요.")
        return
    
    # 2. 비디오에서 프레임 추출
    dataset_root = extract_frames_from_videos(normal_videos)
    if dataset_root is None:
        print("❌ 프레임 추출 실패")
        return
    
    # 3. AI-VAD 모델 생성
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
    
    # 4. 사전 훈련된 가중치 로드 (선택사항)
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
    
    # 5. Anomalib Engine 생성 (AI-VAD 전용 설정)
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
    
    # 6. Folder 데이터 모듈 생성 (이미지 기반)
    print(f"\n📊 Folder 데이터 모듈 생성...")
    try:
        datamodule = Folder(
            name="simple_frames",
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
    
    # 7. AI-VAD 학습 (올바른 방법)
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
    
    # 8. 체크포인트 저장
    checkpoint_path = "aivad_simple_learned.ckpt"
    try:
        # AI-VAD 모델 상태 저장
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'simple_density_estimation',
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
    print("3. UI에서 'aivad_simple_learned.ckpt' 로드하여 테스트")
    print("4. 비정상 데이터는 분포에서 벗어나 높은 점수")

if __name__ == "__main__":
    main()
