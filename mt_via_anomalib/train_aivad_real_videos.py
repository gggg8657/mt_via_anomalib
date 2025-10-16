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
    
    # 3. AI-VAD 모델 생성 (객체 감지 개선 설정)
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
            # 객체 감지 개선 설정
            box_score_thresh=0.3,  # 낮춤 (0.7 -> 0.3)
            min_bbox_area=50,      # 낮춤 (100 -> 50)
            max_bbox_overlap=0.8,  # 높임 (0.65 -> 0.8)
            foreground_binary_threshold=10,  # 낮춤 (18 -> 10)
        )
        
        # 모델을 GPU로 이동
        if device == "cuda":
            model = model.to(device)
            print(f"✅ AI-VAD 모델 생성 완료 (GPU: {device})")
        else:
            print("✅ AI-VAD 모델 생성 완료 (CPU)")
        
    except Exception as e:
        print(f"❌ AI-VAD 모델 생성 실패: {e}")
        return
    
    # 4. 사전 훈련된 가중치 로드
    checkpoint_path = "aivad_proper_checkpoint.ckpt"
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 사전 훈련된 가중치 로드: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                # 가중치 로드 후에도 모델을 GPU로 이동
                if device == "cuda":
                    model = model.to(device)
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
    
    # 6. 직접적인 비디오 처리 방식 (Avenue 대신)
    print(f"\n📊 직접적인 비디오 처리 방식 사용...")
    print("💡 Avenue 데이터 모듈 대신 직접 비디오 처리")
    
    # 비디오 파일들 직접 처리
    video_files_available = []
    train_dir = Path(dataset_root) / "train" / "normal"
    
    for video_file in train_dir.glob("*.avi"):
        if video_file.exists():
            video_files_available.append(str(video_file))
    
    print(f"✅ 사용 가능한 비디오: {len(video_files_available)}개")
    
    if len(video_files_available) == 0:
        print("❌ 사용 가능한 비디오가 없습니다!")
        return
    
    # 7. AI-VAD 직접 학습 (비디오 파일들 직접 처리)
    print(f"\n🎯 AI-VAD 직접 학습 시작...")
    print("💡 학습 과정:")
    print("   1. 실제 비디오 파일들 직접 로드")
    print("   2. Feature Extraction: Flow, Region, Pose, Deep features")
    print("   3. Density Update: 정상 특성들을 density estimator에 누적")
    print("   4. Density Fit: 모든 특성으로 분포 모델 학습")
    print("   5. No Backpropagation: 가중치 업데이트 없음!")
    print("   6. 우리 환경의 실제 객체들 학습!")
    
    try:
        # 직접적인 비디오 처리 및 AI-VAD 학습
        model.model.eval()  # 평가 모드로 설정
        
        # 모델이 GPU에 있는지 확인
        print(f"🔍 모델 디바이스 확인:")
        print(f"   - 모델 디바이스: {next(model.model.parameters()).device}")
        print(f"   - 타겟 디바이스: {device}")
        
        if device == "cuda" and next(model.model.parameters()).device.type != "cuda":
            print("⚠️ 모델을 GPU로 이동 중...")
            model.model = model.model.to(device)
        
        total_clips_processed = 0
        total_detections = 0
        
        for i, video_path in enumerate(video_files_available[:10]):  # 처음 10개만 처리
            print(f"\n📹 비디오 처리 중: {i+1}/{min(10, len(video_files_available))}")
            print(f"   파일: {Path(video_path).name}")
            
            try:
                # 비디오 로드
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"   ⚠️ 비디오 열기 실패: {video_path}")
                    continue
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"   📊 프레임 수: {frame_count}, FPS: {fps:.1f}")
                
                # 2프레임씩 클립으로 처리
                clip_count = 0
                frame_buffer = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 프레임 전처리
                    frame_resized = cv2.resize(frame, (224, 224))
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
                    
                    frame_buffer.append(frame_tensor)
                    
                    # 2프레임이 모이면 클립 처리
                    if len(frame_buffer) == 2:
                        try:
                            # 비디오 클립 생성 [2, 3, 224, 224]
                            video_clip = torch.stack(frame_buffer).unsqueeze(0)  # [1, 2, 3, 224, 224]
                            
                            # 디바이스 확인 및 이동
                            if device == "cuda":
                                video_clip = video_clip.to(device)
                            
                            # AI-VAD 추론 (학습 모드)
                            with torch.no_grad():
                                try:
                                    output = model.model(video_clip)
                                    
                                    # 출력 구조 확인
                                    if isinstance(output, list) and len(output) > 0:
                                        # 특성 추출 및 density estimator 업데이트
                                        if hasattr(model.model, 'density_estimator'):
                                            # AI-VAD의 내부 특성들을 density estimator에 추가
                                            model.model.density_estimator.update(output)
                                            total_detections += 1
                                    else:
                                        # 출력이 비어있거나 예상과 다름
                                        if clip_count == 0:  # 첫 번째 클립에서만 출력
                                            print(f"   ⚠️ AI-VAD 출력이 비어있거나 예상과 다름: {type(output)}")
                                        
                                except Exception as e:
                                    if "index 0 is out of bounds" in str(e):
                                        # Region Extractor에서 객체 감지 실패
                                        if clip_count == 0:  # 첫 번째 클립에서만 출력
                                            print(f"   ⚠️ 객체 감지 실패: Region Extractor가 객체를 찾지 못함")
                                    else:
                                        print(f"   ⚠️ AI-VAD 추론 실패: {e}")
                                    continue
                            
                            clip_count += 1
                            total_clips_processed += 1
                            
                            if clip_count % 10 == 0:
                                print(f"   ✅ 처리된 클립: {clip_count}")
                            
                        except Exception as e:
                            # 첫 번째 클립에서만 상세 에러 출력
                            if clip_count == 0:
                                print(f"   ⚠️ 클립 처리 실패: {e}")
                        
                        # 버퍼에서 첫 번째 프레임 제거
                        frame_buffer.pop(0)
                
                cap.release()
                print(f"   ✅ 완료: {clip_count}개 클립 처리")
                
            except Exception as e:
                print(f"   ❌ 비디오 처리 실패: {e}")
                continue
        
        print(f"\n📊 전체 처리 결과:")
        print(f"   - 처리된 비디오: {min(10, len(video_files_available))}개")
        print(f"   - 처리된 클립: {total_clips_processed}개")
        print(f"   - 총 감지 수: {total_detections}개")
        
        # Density estimator 최종 학습
        if hasattr(model.model, 'density_estimator') and total_detections > 0:
            print(f"\n🔧 Density Estimator 최종 학습...")
            model.model.density_estimator.fit()
            print("✅ Density Estimator 학습 완료")
        else:
            print("⚠️ 감지된 특성이 없어 density estimator 학습을 건너뜁니다.")
            print("💡 해결 방법:")
            print("   1. 비디오에 움직이는 객체가 있는지 확인")
            print("   2. 조명이 충분한지 확인")
            print("   3. 객체가 충분히 큰지 확인 (최소 50x50 픽셀)")
            print("   4. AI-VAD 파라미터를 더 낮춤 (box_score_thresh=0.1)")
        
        print("✅ AI-VAD 직접 학습 완료!")
        
    except Exception as e:
        print(f"❌ AI-VAD 직접 학습 실패: {e}")
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
            'training_type': 'real_videos_direct_processing',
            'total_clips_processed': total_clips_processed,
            'total_detections': total_detections,
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
