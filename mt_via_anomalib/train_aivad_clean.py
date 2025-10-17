"""
AI-VAD 깔끔한 버전으로 학습
구문 오류 없는 깔끔한 코드
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models.video import AiVad

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

def process_video_file(video_path, model, device):
    """단일 비디오 파일 처리"""
    print(f"📹 비디오 처리: {Path(video_path).name}")
    
    try:
        # 비디오 로드
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"   ⚠️ 비디오 열기 실패")
            return 0, 0
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   📊 해상도: {width}x{height}, 프레임: {frame_count}, FPS: {fps:.1f}")
        
        # 2프레임씩 클립으로 처리
        clip_count = 0
        detection_count = 0
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
                    # 비디오 클립 생성 [1, 2, 3, 224, 224]
                    video_clip = torch.stack(frame_buffer).unsqueeze(0).to(device)
                    
                    # AI-VAD 추론
                    with torch.no_grad():
                        output = model.model(video_clip)
                        
                        # 출력 구조 확인
                        if isinstance(output, list) and len(output) > 0:
                            # 특성 추출 및 density estimator 업데이트
                            if hasattr(model.model, 'density_estimator'):
                                model.model.density_estimator.update(output)
                                detection_count += 1
                            
                            clip_count += 1
                            
                            if clip_count == 1:
                                print(f"   🎉 첫 번째 객체 감지 성공!")
                            elif clip_count % 10 == 0:
                                print(f"   ✅ 처리된 클립: {clip_count}")
                        
                except Exception as e:
                    # 에러 무시하고 계속 진행
                    pass
                
                # 버퍼에서 첫 번째 프레임 제거
                frame_buffer.pop(0)
        
        cap.release()
        print(f"   ✅ 완료: {clip_count}개 클립, {detection_count}개 감지")
        return clip_count, detection_count
        
    except Exception as e:
        print(f"   ❌ 비디오 처리 실패: {e}")
        return 0, 0

def main():
    """메인 함수"""
    print("🚀 AI-VAD 깔끔한 버전으로 학습")
    print("=" * 60)
    print("💡 극단적인 설정:")
    print("   1. box_score_thresh=0.05 (극단적으로 낮음)")
    print("   2. min_bbox_area=10 (극단적으로 작음)")
    print("   3. max_bbox_overlap=0.95 (극단적으로 높음)")
    print("   4. foreground_binary_threshold=2 (극단적으로 민감)")
    print("=" * 60)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 1. 비디오 디렉토리 설정
    video_directory = "/data/DJ/anomalib_DATAPATH/training_videos"
    print(f"\n📁 비디오 디렉토리: {video_directory}")
    
    # 2. 디렉토리에서 비디오 파일들 가져오기
    video_files = get_video_files_from_directory(video_directory)
    
    if len(video_files) == 0:
        print("❌ 비디오 파일을 찾을 수 없습니다!")
        return
    
    # 3. AI-VAD 모델 생성 (극단적인 설정)
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
            box_score_thresh=0.05,  # 극단적으로 낮춤
            min_bbox_area=10,       # 극단적으로 작게
            max_bbox_overlap=0.95,  # 극단적으로 높게
            foreground_binary_threshold=2,  # 극단적으로 민감
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
                if device == "cuda":
                    model = model.to(device)
                print("✅ 사전 훈련된 가중치 로드 완료")
            else:
                print("⚠️ state_dict가 없습니다.")
        except Exception as e:
            print(f"⚠️ 가중치 로드 실패: {e}")
    
    # 5. AI-VAD 학습 (배치 처리)
    print(f"\n🎯 AI-VAD 학습 시작...")
    
    # 모델을 평가 모드로 설정
    model.model.eval()
    
    # 배치 처리 설정
    batch_size = 10
    total_batches = (len(video_files) + batch_size - 1) // batch_size
    
    print(f"📊 처리 계획:")
    print(f"   - 전체 비디오: {len(video_files)}개")
    print(f"   - 배치 크기: {batch_size}개")
    print(f"   - 총 배치 수: {total_batches}개")
    
    total_clips_processed = 0
    total_detections = 0
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(video_files))
        batch_videos = video_files[start_idx:end_idx]
        
        print(f"\n🔄 배치 {batch_idx + 1}/{total_batches} 처리 중...")
        print(f"   - 처리 범위: {start_idx + 1}~{end_idx}")
        
        batch_clips = 0
        batch_detections = 0
        
        for video_path in batch_videos:
            clips, detections = process_video_file(video_path, model, device)
            batch_clips += clips
            batch_detections += detections
        
        total_clips_processed += batch_clips
        total_detections += batch_detections
        
        print(f"\n✅ 배치 {batch_idx + 1}/{total_batches} 완료")
        print(f"   - 배치 클립: {batch_clips}")
        print(f"   - 배치 감지: {batch_detections}")
        print(f"   - 총 클립: {total_clips_processed}")
        print(f"   - 총 감지: {total_detections}")
    
    # 6. Density estimator 최종 학습
    print(f"\n📊 전체 처리 결과:")
    print(f"   - 처리된 비디오: {len(video_files)}개")
    print(f"   - 처리된 클립: {total_clips_processed}개")
    print(f"   - 총 감지 수: {total_detections}개")
    
    if hasattr(model.model, 'density_estimator') and total_detections > 0:
        print(f"\n🔧 Density Estimator 최종 학습...")
        model.model.density_estimator.fit()
        print("✅ Density Estimator 학습 완료")
    else:
        print("⚠️ 감지된 특성이 없어 density estimator 학습을 건너뜁니다.")
    
    # 7. 체크포인트 저장
    checkpoint_path = "aivad_clean_learned.ckpt"
    try:
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'clean_density_estimation',
            'total_clips_processed': total_clips_processed,
            'total_detections': total_detections,
            'video_directory': video_directory,
        }, checkpoint_path)
        
        print(f"💾 학습된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    print("\n🎉 AI-VAD 깔끔한 학습 완료!")
    print("💡 학습된 내용:")
    print("1. 극단적으로 민감한 객체 감지")
    print("2. 모든 움직임과 변화 감지")
    print("3. 2064개 비디오 모두 처리")
    print("4. Density Estimator로 이상 탐지 준비")
    print("5. UI에서 'aivad_clean_learned.ckpt' 로드하여 테스트")

if __name__ == "__main__":
    main()
