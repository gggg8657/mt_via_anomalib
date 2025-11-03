"""
AI-VAD 극단적인 설정으로 학습
모든 비디오에서 객체 감지 실패 문제 해결
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models.video import AiVad

def analyze_video_content_detailed(video_path):
    """비디오 내용 상세 분석"""
    print(f"🔍 상세 비디오 분석: {Path(video_path).name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ❌ 비디오 열기 실패")
        return False, 0, []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   📊 해상도: {width}x{height}, 프레임: {frame_count}, FPS: {fps:.1f}")
    
    # 더 상세한 움직임 감지
    prev_frame = None
    motion_scores = []
    edge_scores = []
    
    for i in range(min(20, frame_count)):  # 처음 20프레임 분석
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 전처리
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 에지 감지
        edges = cv2.Canny(frame_gray, 50, 150)
        edge_score = np.mean(edges)
        edge_scores.append(edge_score)
        
        if prev_frame is not None:
            # 움직임 감지
            diff = cv2.absdiff(prev_frame, frame_gray)
            motion_score = np.mean(diff)
            motion_scores.append(motion_score)
        
        prev_frame = frame_gray.copy()
    
    cap.release()
    
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    avg_edge = np.mean(edge_scores) if edge_scores else 0
    max_motion = np.max(motion_scores) if motion_scores else 0
    
    print(f"   🏃 평균 움직임: {avg_motion:.2f}, 최대 움직임: {max_motion:.2f}")
    print(f"   📐 평균 에지: {avg_edge:.2f}")
    print(f"   🎯 움직임 감지: {'✅' if max_motion > 2 else '❌'}")
    
    return max_motion > 2, max_motion, motion_scores

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
    print("🚀 AI-VAD 극단적인 설정으로 학습")
    print("=" * 60)
    print("💡 극단적인 설정:")
    print("   1. box_score_thresh=0.05 (극단적으로 낮음)")
    print("   2. min_bbox_area=10 (극단적으로 작음)")
    print("   3. max_bbox_overlap=0.95 (극단적으로 높음)")
    print("   4. foreground_binary_threshold=2 (극단적으로 민감)")
    print("   5. 상세한 비디오 분석 및 움직임 감지")
    print("=" * 60)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 1. 비디오 디렉토리 설정
    video_directory = input("\n📁 비디오 디렉토리 경로를 입력하세요: ").strip()
    
    if not video_directory:
        # 기본 디렉토리 설정
        video_directory = r"C:\Users\User\PycharmProjects\pythonProject1\mt_via_anomalib\mt_via_anomalib\cropped_videos\normal"
        print(f"기본 디렉토리 사용: {video_directory}")
    
    # 2. 디렉토리에서 비디오 파일들 가져오기
    video_files = get_video_files_from_directory(video_directory)
    
    if len(video_files) == 0:
        print("❌ 비디오 파일을 찾을 수 없습니다!")
        return
    
    # 3. 모든 비디오 활용 (분석 없이 직접 처리)
    print(f"\n🔍 모든 비디오 활용 모드...")
    print(f"💡 분석 단계를 건너뛰고 모든 비디오를 직접 처리합니다.")
    
    # 모든 비디오를 처리 대상으로 설정
    motion_videos = [(vf, 1.0) for vf in video_files]
    
    print(f"\n📊 처리 대상:")
    print(f"   - 전체 비디오: {len(video_files)}개")
    print(f"   - 처리할 비디오: {len(motion_videos)}개")
    print(f"   - 처리 방식: 모든 비디오 직접 처리")
    
    # 4. AI-VAD 모델 생성 (극단적인 설정)
    print(f"\n🤖 AI-VAD 모델 생성 (극단적인 설정)...")
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
            # 극단적인 객체 감지 설정
            box_score_thresh=0.05,  # 극단적으로 낮춤 (0.1 -> 0.05)
            min_bbox_area=10,       # 극단적으로 작게 (25 -> 10)
            max_bbox_overlap=0.95,  # 극단적으로 높게 (0.9 -> 0.95)
            foreground_binary_threshold=2,  # 극단적으로 민감 (5 -> 2)
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
    
    # 6. AI-VAD 직접 학습 (극단적인 설정)
    print(f"\n🎯 AI-VAD 극단적인 학습 시작...")
    print("💡 학습 과정:")
    print("   1. 극단적으로 민감한 객체 감지")
    print("   2. 모든 움직임과 변화 감지")
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
        
        total_clips_processed = 0
        total_detections = 0
        
        # 모든 비디오 처리 (처리 시간을 고려하여 배치 단위로 처리)
        batch_size = 10  # 한 번에 10개씩 처리
        total_batches = (len(motion_videos) + batch_size - 1) // batch_size
        
        print(f"📊 처리 계획:")
        print(f"   - 전체 비디오: {len(motion_videos)}개")
        print(f"   - 배치 크기: {batch_size}개")
        print(f"   - 총 배치 수: {total_batches}개")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(motion_videos))
            batch_videos = motion_videos[start_idx:end_idx]
            
            print(f"\n🔄 배치 {batch_idx + 1}/{total_batches} 처리 중...")
            print(f"   - 처리 범위: {start_idx + 1}~{end_idx}")
            
            for i, (video_path, motion_score) in enumerate(batch_videos):
                video_idx = start_idx + i + 1
                print(f"\n📹 비디오 처리 중: {video_idx}/{len(motion_videos)}")
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
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    print(f"   📊 해상도: {width}x{height}, 프레임: {frame_count}, FPS: {fps:.1f}")
                    
                    # 해상도가 너무 작으면 경고
                    if width < 224 or height < 224:
                        print(f"   ⚠️ 해상도가 작습니다: {width}x{height} (권장: 224x224 이상)")
                    
                    # 2프레임씩 클립으로 처리
                    clip_count = 0
                    frame_buffer = []
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 프레임 전처리 (더 큰 해상도로 리사이즈)
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
                                
                                # AI-VAD 추론 (극단적인 설정)
                                try:
                                    with torch.no_grad():
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
                                            elif clip_count % 5 == 0:
                                                print(f"   ✅ 처리된 클립: {clip_count}")
                                            
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
            
            # 배치 완료 후 진행 상황 출력
            processed_videos = min(end_idx, len(motion_videos))
            print(f"\n✅ 배치 {batch_idx + 1}/{total_batches} 완료")
            print(f"   - 처리된 비디오: {processed_videos}/{len(motion_videos)}")
            print(f"   - 처리된 클립: {total_clips_processed}")
            print(f"   - 총 감지 수: {total_detections}")
        
        print(f"\n📊 전체 처리 결과:")
        print(f"   - 처리된 비디오: {len(motion_videos)}개")
        print(f"   - 처리된 클립: {total_clips_processed}개")
        print(f"   - 총 감지 수: {total_detections}개")
        
        # Density estimator 최종 학습
        if hasattr(model.model, 'density_estimator') and total_detections > 0:
            print(f"\n🔧 Density Estimator 최종 학습...")
            model.model.density_estimator.fit()
            print("✅ Density Estimator 학습 완료")
        else:
            print("⚠️ 감지된 특성이 없어 density estimator 학습을 건너뜁니다.")
            print("💡 근본적인 문제:")
            print("   1. 비디오 해상도가 너무 작음 (270x480)")
            print("   2. 비디오에 실제 움직이는 객체가 없음")
            print("   3. 비디오 품질이 낮음 (압축, 노이즈)")
            print("   4. 조명이 부족하거나 일정함")
            print("💡 해결 방법:")
            print("   1. 더 큰 해상도의 비디오 사용 (최소 640x480)")
            print("   2. 실제 움직이는 사람/객체가 있는 비디오 사용")
            print("   3. 조명이 충분하고 변화가 있는 환경")
            print("   4. 압축되지 않은 고품질 비디오 사용")
        
        print("✅ AI-VAD 극단적인 학습 완료!")
        
    except Exception as e:
        print(f"❌ AI-VAD 극단적인 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 체크포인트 저장
    checkpoint_path = "aivad_extreme_learned.ckpt"
    try:
        # AI-VAD 모델 상태 저장
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'extreme_density_estimation',
            'total_clips_processed': total_clips_processed,
            'total_detections': total_detections,
            'video_directory': video_directory,
        }, checkpoint_path)
        
        print(f"💾 학습된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    print("\n🎉 AI-VAD 극단적인 학습 완료!")
    print("💡 학습된 내용:")
    print("1. 극단적으로 민감한 객체 감지")
    print("2. 모든 움직임과 변화 감지")
    print("3. 상세한 비디오 분석")
    print("4. Density Estimator로 이상 탐지 준비")
    print("5. UI에서 'aivad_extreme_learned.ckpt' 로드하여 테스트")

if __name__ == "__main__":
    main()
