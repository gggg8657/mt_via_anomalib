"""
AI-VAD 디렉토리 기반 학습
효율적인 비디오 파일 로딩
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models.video import AiVad

def analyze_video_content(video_path):
    """비디오 내용 분석"""
    print(f"🔍 비디오 분석: {Path(video_path).name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ❌ 비디오 열기 실패")
        return False, 0
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   📊 해상도: {width}x{height}, 프레임: {frame_count}, FPS: {fps:.1f}")
    
    # 움직임 감지
    prev_frame = None
    motion_scores = []
    
    for i in range(min(10, frame_count)):  # 처음 10프레임만 분석
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 전처리
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # 움직임 감지
            diff = cv2.absdiff(prev_frame, frame_gray)
            motion_score = np.mean(diff)
            motion_scores.append(motion_score)
        
        prev_frame = frame_gray.copy()
    
    cap.release()
    
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    motion_detected = avg_motion > 5
    print(f"   🏃 평균 움직임 점수: {avg_motion:.2f}")
    print(f"   🎯 움직임 감지: {'✅' if motion_detected else '❌'}")
    
    return motion_detected, avg_motion

def get_video_files_from_directory(video_dir):
    """디렉토리에서 비디오 파일들 가져오기"""
    print(f"📁 비디오 디렉토리 스캔: {video_dir}")
    
    if not os.path.exists(video_dir):
        print(f"❌ 디렉토리가 존재하지 않습니다: {video_dir}")
        return []
    
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    
    # 디렉토리에서 모든 비디오 파일 찾기
    for file_path in Path(video_dir).rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))
    
    print(f"✅ 발견된 비디오 파일: {len(video_files)}개")
    
    # 비디오 파일들 정렬
    video_files.sort()
    
    return video_files

def main():
    """메인 함수"""
    print("🚀 AI-VAD 디렉토리 기반 학습")
    print("=" * 60)
    print("💡 효율적인 디렉토리 기반 처리:")
    print("   1. 디렉토리에서 자동으로 비디오 파일 스캔")
    print("   2. 파일 복사 없이 직접 처리")
    print("   3. 극단적으로 민감한 객체 감지")
    print("   4. 비디오 내용 분석 및 움직임 감지")
    print("=" * 60)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 1. 비디오 디렉토리 설정
    video_directory = input("\n📁 비디오 디렉토리 경로를 입력하세요 (예: C:\\Users\\User\\Videos): ").strip()
    
    if not video_directory:
        # 기본 디렉토리 설정
        video_directory = r"C:\Users\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\training_videos"
        print(f"기본 디렉토리 사용: {video_directory}")
    
    # 2. 디렉토리에서 비디오 파일들 가져오기
    video_files = get_video_files_from_directory(video_directory)
    
    if len(video_files) == 0:
        print("❌ 비디오 파일을 찾을 수 없습니다!")
        print("💡 지원되는 형식: .avi, .mp4, .mov, .mkv, .flv, .wmv")
        return
    
    # 3. 비디오 내용 분석
    print(f"\n🔍 비디오 내용 분석 시작...")
    motion_videos = []
    
    for video_file in video_files[:10]:  # 처음 10개만 분석
        motion_detected, motion_score = analyze_video_content(video_file)
        if motion_detected:
            motion_videos.append((video_file, motion_score))
    
    print(f"\n📊 분석 결과:")
    print(f"   - 전체 비디오: {len(video_files)}개")
    print(f"   - 움직임 감지된 비디오: {len(motion_videos)}개")
    
    # 4. AI-VAD 모델 생성 (극단적으로 민감한 설정)
    print(f"\n🤖 AI-VAD 모델 생성 (극단적으로 민감한 설정)...")
    try:
        model = AiVad(
            # AI-VAD 기본 설정
            use_velocity_features=True,
            use_pose_features=True,
            use_deep_features=True,
            # Density estimation 설정
            n_components_velocity=2,
            n_neighbors_pose=1,
            n_neighbors_deep=1,
            # 극단적으로 민감한 객체 감지 설정
            box_score_thresh=0.1,  # 극단적으로 낮춤
            min_bbox_area=25,      # 극단적으로 작게
            max_bbox_overlap=0.9,  # 극단적으로 높게
            foreground_binary_threshold=5,  # 극단적으로 민감
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
    
    # 5. 사전 훈련된 가중치 로드
    checkpoint_path = "aivad_proper_checkpoint.ckpt"
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 사전 훈련된 가중치 로드: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                if device == "cuda":
                    model = model.to(device)
                print("✅ 사전 훈련된 가중치 로드 완료")
            else:
                print("⚠️ state_dict가 없습니다.")
        except Exception as e:
            print(f"⚠️ 가중치 로드 실패: {e}")
    
    # 6. AI-VAD 직접 학습 (디렉토리 기반)
    print(f"\n🎯 AI-VAD 디렉토리 기반 학습 시작...")
    print("💡 학습 과정:")
    print("   1. 디렉토리에서 비디오 파일 직접 로드")
    print("   2. 극단적으로 민감한 객체 감지")
    print("   3. Density Update: 정상 특성들을 density estimator에 누적")
    print("   4. Density Fit: 모든 특성으로 분포 모델 학습")
    
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
        
        # 움직임이 감지된 비디오들만 처리 (또는 모든 비디오 처리)
        videos_to_process = motion_videos if motion_videos else [(vf, 0) for vf in video_files[:5]]
        
        total_clips_processed = 0
        total_detections = 0
        
        for i, (video_path, motion_score) in enumerate(videos_to_process[:5]):  # 처음 5개만 처리
            print(f"\n📹 비디오 처리 중: {i+1}/{min(5, len(videos_to_process))}")
            print(f"   파일: {Path(video_path).name}")
            print(f"   움직임 점수: {motion_score:.2f}")
            
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
                            
                            # AI-VAD 추론 (극단적으로 민감한 설정)
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
                                            
                                        clip_count += 1
                                        total_clips_processed += 1
                                        
                                        if clip_count == 1:  # 첫 번째 성공한 클립
                                            print(f"   🎉 첫 번째 객체 감지 성공!")
                                        
                                    else:
                                        # 출력이 비어있거나 예상과 다름
                                        if clip_count == 0:  # 첫 번째 클립에서만 출력
                                            print(f"   ⚠️ AI-VAD 출력이 비어있거나 예상과 다름: {type(output)}")
                                        
                                except Exception as e:
                                    if "index 0 is out of bounds" in str(e):
                                        # Region Extractor에서 객체 감지 실패
                                        if clip_count == 0:  # 첫 번째 클립에서만 출력
                                            print(f"   ⚠️ 객체 감지 실패: Region Extractor가 객체를 찾지 못함")
                                    elif "amax(): Expected reduction dim 0" in str(e):
                                        # 빈 텐서 문제
                                        if clip_count == 0:
                                            print(f"   ⚠️ 빈 텐서 문제: amax() 에러")
                                    else:
                                        if clip_count == 0:
                                            print(f"   ⚠️ AI-VAD 추론 실패: {e}")
                                    continue
                            
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
        print(f"   - 처리된 비디오: {min(5, len(videos_to_process))}개")
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
            print("   3. 객체가 충분히 큰지 확인 (최소 25x25 픽셀)")
            print("   4. 다른 비디오 디렉토리 시도")
        
        print("✅ AI-VAD 디렉토리 기반 학습 완료!")
        
    except Exception as e:
        print(f"❌ AI-VAD 디렉토리 기반 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 체크포인트 저장
    checkpoint_path = "aivad_directory_learned.ckpt"
    try:
        # AI-VAD 모델 상태 저장
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'directory_based_density_estimation',
            'total_clips_processed': total_clips_processed,
            'total_detections': total_detections,
            'video_directory': video_directory,
        }, checkpoint_path)
        
        print(f"💾 학습된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    print("\n🎉 AI-VAD 디렉토리 기반 학습 완료!")
    print("💡 학습된 내용:")
    print("1. 디렉토리에서 자동으로 비디오 파일 스캔")
    print("2. 파일 복사 없이 직접 처리")
    print("3. 극단적으로 민감한 객체 감지")
    print("4. 움직임 감지된 비디오 우선 처리")
    print("5. Density Estimator로 이상 탐지 준비")
    print("6. UI에서 'aivad_directory_learned.ckpt' 로드하여 테스트")

if __name__ == "__main__":
    main()
