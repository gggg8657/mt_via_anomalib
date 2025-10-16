"""
AI-VAD 실제 비디오 데이터로 학습
video_files_list.py의 실제 비디오 파일들을 사용
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models.video import AiVad
from anomalib.engine import Engine
from anomalib.data import Avenue
import shutil

# video_files_list.py에서 비디오 파일 경로 가져오기
def get_video_files():
    """video_files_list.py에서 비디오 파일 경로 가져오기"""
    # video_files_list.py의 내용을 직접 가져오기
    video_files = [
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_0.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_1.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_2.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_3.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_4.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_5.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_6.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_7.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_8.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_9.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_10.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_11.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_12.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_13.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_14.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_15.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_16.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_17.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_18.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_19.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_20.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_21.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_22.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_0.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_1.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_2.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_3.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_4.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_5.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_6.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_7.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_8.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_9.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_10.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_11.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_12.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_13.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_14.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_15.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_16.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_17.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_18.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_19.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_20.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_21.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_22.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_23.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_24.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_25.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_26.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_27.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_28.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_29.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_30.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_31.avi",
        "C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_32.avi",
    ]
    return video_files

def create_real_video_dataset(video_files, target_dir="real_video_dataset"):
    """실제 비디오 파일들을 AI-VAD용 데이터셋으로 변환"""
    print(f"📁 실제 비디오 데이터셋 생성: {target_dir}")
    
    # Avenue 데이터셋 구조 생성
    train_dir = Path(target_dir) / "train" / "normal"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📂 정상 비디오 폴더: {train_dir}")
    
    # 실제 비디오 파일들 복사
    copied_count = 0
    
    for i, video_file in enumerate(video_files):
        if os.path.exists(video_file):
            # 비디오 파일명 생성
            video_name = f"normal_{i:03d}.avi"
            target_path = train_dir / video_name
            
            try:
                # 파일 복사
                shutil.copy2(video_file, target_path)
                copied_count += 1
                
                if copied_count <= 10:  # 처음 10개만 표시
                    print(f"    🎬 {video_name}")
                
            except Exception as e:
                print(f"    ⚠️ {video_file} 복사 실패: {e}")
        else:
            print(f"    ⚠️ 파일 없음: {video_file}")
    
    print(f"  ✅ 복사된 비디오: {copied_count}개")
    
    if copied_count == 0:
        print("❌ 복사된 비디오가 없습니다!")
        return None
    
    return str(Path(target_dir).absolute())

def main():
    """메인 함수"""
    print("🚀 AI-VAD 실제 비디오 데이터로 학습")
    print("=" * 60)
    print("💡 실제 비디오 데이터 사용:")
    print("   1. 실제 비디오 파일들 (56개)")
    print("   2. AI-VAD 원래 방식 (Density Estimation)")
    print("   3. 우리 환경의 실제 객체들")
    print("   4. 높은 Domain 적용성")
    print("=" * 60)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 1. 실제 비디오 파일들 가져오기
    video_files = get_video_files()
    print(f"\n📊 실제 비디오 파일: {len(video_files)}개")
    
    # 2. 실제 비디오 데이터셋 생성
    dataset_root = create_real_video_dataset(video_files)
    if dataset_root is None:
        print("❌ 실제 비디오 데이터셋 생성 실패")
        return
    
    # 3. AI-VAD 모델 생성 (원래 설정)
    print(f"\n🤖 AI-VAD 모델 생성...")
    try:
        model = AiVad(
            # AI-VAD 원래 설정
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
    
    # 6. Avenue 데이터 모듈 생성 (실제 비디오용)
    print(f"\n📊 Avenue 데이터 모듈 생성...")
    try:
        # Avenue 데이터 모듈 사용 (실제 비디오)
        datamodule = Avenue(
            root=dataset_root,
            clip_length_in_frames=2,  # AI-VAD 표준
            frames_between_clips=1,
            train_batch_size=1,  # 작은 배치 크기
            eval_batch_size=1,
            num_workers=0,
        )
        
        print("✅ Avenue 데이터 모듈 생성 완료")
        
    except Exception as e:
        print(f"❌ Avenue 데이터 모듈 생성 실패: {e}")
        print("💡 Avenue 데이터셋 구조가 필요합니다.")
        return
    
    # 7. AI-VAD 학습 (실제 비디오로)
    print(f"\n🎯 AI-VAD 학습 시작 (실제 비디오)...")
    print("💡 학습 과정:")
    print("   1. 실제 비디오 파일들 사용")
    print("   2. Feature Extraction: Flow, Region, Pose, Deep features")
    print("   3. Density Update: 정상 특성들을 density estimator에 누적")
    print("   4. Density Fit: 모든 특성으로 분포 모델 학습")
    print("   5. No Backpropagation: 가중치 업데이트 없음!")
    print("   6. 우리 환경의 실제 객체들 학습!")
    
    try:
        # AI-VAD의 올바른 학습 방법 (실제 비디오)
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
    checkpoint_path = "aivad_real_videos_learned.ckpt"
    try:
        # AI-VAD 모델 상태 저장
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'real_videos_density_estimation',
            'total_detections': model.total_detections,
        }, checkpoint_path)
        
        print(f"💾 학습된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    print("\n🎉 AI-VAD 실제 비디오 학습 완료!")
    print("💡 학습된 내용:")
    print("1. 실제 비디오 파일들로 학습")
    print("2. 우리 환경의 실제 객체들")
    print("3. 높은 Domain 적용성")
    print("4. 정상 데이터의 Feature 분포 학습")
    print("5. Density Estimator로 이상 탐지 준비")
    print("6. UI에서 'aivad_real_videos_learned.ckpt' 로드하여 테스트")
    print("7. 비정상 데이터는 분포에서 벗어나 높은 점수")

if __name__ == "__main__":
    main()
