"""
AI-VAD의 올바른 학습 방법
Density Estimation 기반 One-Class Learning
"""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models.video import AiVad
from anomalib.engine import Engine
from anomalib.data import Avenue
from anomalib.data.utils import VideoTargetFrame
import shutil

def create_proper_video_dataset_from_json(json_path="image_segments.json", target_dir="proper_video_dataset"):
    """JSON에서 정상 프레임들을 비디오 시퀀스로 변환"""
    print(f"📁 AI-VAD용 비디오 시퀀스 생성: {json_path}")
    
    # Avenue 데이터셋 구조 생성
    train_dir = Path(target_dir) / "train" / "normal"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📂 정상 비디오 폴더: {train_dir}")
    
    # JSON 파일 로드
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        print(f"  📊 JSON 로드 완료: {len(segments)}개 세그먼트")
    except Exception as e:
        print(f"❌ JSON 로드 실패: {e}")
        return None
    
    # 정상 프레임들을 비디오로 변환
    video_count = 0
    
    for i, segment in enumerate(segments):
        if segment.get('category') == 'normal' and 'images' in segment:
            images = segment['images']
            
            # 각 세그먼트에서 연속된 프레임들로 비디오 생성
            if len(images) >= 2:
                video_count += 1
                
                # 비디오 파일명 생성
                video_name = f"normal_{video_count:03d}.mp4"
                video_path = train_dir / video_name
                
                try:
                    # 첫 번째 프레임으로 비디오 정보 파악
                    first_frame = cv2.imread(images[0])
                    if first_frame is not None:
                        height, width = first_frame.shape[:2]
                        
                        # 비디오 라이터 생성
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))
                        
                        # 프레임들을 비디오로 추가
                        for img_path in images[:10]:  # 최대 10프레임
                            frame = cv2.imread(img_path)
                            if frame is not None:
                                out.write(frame)
                        
                        out.release()
                        
                        if video_count <= 5:  # 처음 5개만 표시
                            print(f"    🎬 {video_name} ({len(images)}프레임)")
                        
                except Exception as e:
                    print(f"    ⚠️ {video_name} 생성 실패: {e}")
                    if os.path.exists(video_path):
                        os.remove(video_path)
    
    print(f"  ✅ 생성된 비디오: {video_count}개")
    
    if video_count == 0:
        print("❌ 생성된 비디오가 없습니다!")
        return None
    
    return str(Path(target_dir).absolute())

def main():
    """메인 함수"""
    print("🚀 AI-VAD 올바른 학습 방법")
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
    
    # 1. 비디오 데이터셋 생성
    dataset_root = create_proper_video_dataset_from_json()
    if dataset_root is None:
        print("❌ 비디오 데이터셋 생성 실패")
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
    
    # 5. 커스텀 데이터로더 생성 (비디오 파일 직접 처리)
    print(f"\n📊 커스텀 비디오 데이터로더 생성...")
    try:
        # Avenue 데이터 모듈 사용 (AI-VAD 표준)
        datamodule = Avenue(
            root=dataset_root,
            clip_length_in_frames=2,  # AI-VAD 표준
            frames_between_clips=1,
            target_frame=VideoTargetFrame.LAST,
            train_batch_size=1,  # 작은 배치 크기
            eval_batch_size=1,
            num_workers=0,
        )
        
        print("✅ 커스텀 데이터로더 생성 완료")
        
    except Exception as e:
        print(f"❌ 커스텀 데이터로더 생성 실패: {e}")
        print("💡 Avenue 데이터셋 구조가 필요합니다.")
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
    checkpoint_path = "aivad_proper_learned.ckpt"
    try:
        # AI-VAD 모델 상태 저장
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'proper_density_estimation',
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
    print("3. UI에서 'aivad_proper_learned.ckpt' 로드하여 테스트")
    print("4. 비정상 데이터는 분포에서 벗어나 높은 점수")

if __name__ == "__main__":
    main()
