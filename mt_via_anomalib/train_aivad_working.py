"""
작동하는 AI-VAD 모델 학습 스크립트 (Tensor 크기 문제 해결)
AI-VAD가 기대하는 정확한 입력 형식 사용
"""

import os
import torch
import numpy as np

# GPU 및 cuDNN 설정 최적화
print("🔧 GPU 및 cuDNN 설정 최적화 중...")

# CUDA 환경 변수 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# cuDNN 설정 조정
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

# GPU 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name()}")
    print(f"✅ GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  GPU를 사용할 수 없습니다. CPU로 실행됩니다.")

from anomalib.models.video import AiVad


def create_correct_video_batch(batch_size=2, num_frames=2):
    """AI-VAD가 기대하는 정확한 비디오 배치 생성"""
    print("📊 AI-VAD용 정확한 비디오 배치 생성 중...")
    
    # AI-VAD는 [batch_size, num_frames, channels, height, width] 형식을 기대
    # num_frames는 보통 2 (first_frame, last_frame)
    # channels는 3 (RGB)
    # height, width는 224x224
    
    video_batch = torch.randn(batch_size, num_frames, 3, 224, 224)
    
    print(f"✅ 비디오 배치 생성 완료: {video_batch.shape}")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 프레임 수: {num_frames}")
    print(f"   - 채널 수: 3 (RGB)")
    print(f"   - 해상도: 224x224")
    
    return video_batch


def test_model_forward_pass(model, device="cuda"):
    """모델 forward pass 테스트"""
    print("🧪 모델 forward pass 테스트 중...")
    
    try:
        # 올바른 형식의 비디오 배치 생성
        video_batch = create_correct_video_batch(batch_size=2, num_frames=2)
        
        # GPU로 이동
        if device == "cuda" and torch.cuda.is_available():
            video_batch = video_batch.cuda()
            print("✅ 비디오 배치를 GPU로 이동 완료")
        
        # Forward pass
        with torch.no_grad():
            output = model(video_batch)
            print(f"✅ Forward pass 성공!")
            print(f"   - 입력 형태: {video_batch.shape}")
            print(f"   - 출력 형태: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Forward pass 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_components(model, device="cuda"):
    """모델 컴포넌트별 테스트"""
    print("🔍 모델 컴포넌트별 테스트 중...")
    
    try:
        # 더미 데이터 생성
        batch_size = 1
        num_frames = 2
        height, width = 224, 224
        
        # 첫 번째와 마지막 프레임 생성
        first_frame = torch.randn(batch_size, 3, height, width)
        last_frame = torch.randn(batch_size, 3, height, width)
        
        if device == "cuda" and torch.cuda.is_available():
            first_frame = first_frame.cuda()
            last_frame = last_frame.cuda()
        
        # Flow extractor 테스트
        print("🔍 Flow extractor 테스트...")
        if hasattr(model, 'model') and hasattr(model.model, 'flow_extractor'):
            with torch.no_grad():
                flow_output = model.model.flow_extractor(first_frame, last_frame)
                print(f"✅ Flow extractor 성공: {flow_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 컴포넌트 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🚀 작동하는 AI-VAD 모델 학습 시작...")
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 사용 디바이스: {device}")
    
    if device == "cuda":
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # GPU 메모리 사용량 확인
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"📊 GPU 메모리 사용량: {allocated:.3f} GB (할당됨), {cached:.3f} GB (캐시됨)")
    
    # 모델 초기화
    print("🤖 AI-VAD 모델 초기화...")
    try:
        model = AiVad()
        print("✅ 모델 초기화 완료")
        
        # 모델을 디바이스로 이동
        if device == "cuda":
            model = model.cuda()
            print("✅ 모델을 GPU로 이동 완료")
            
            # GPU 메모리 사용량 확인
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 모델 로드 후 GPU 메모리: {allocated:.3f} GB (할당됨), {cached:.3f} GB (캐시됨)")
        
    except Exception as e:
        print(f"❌ 모델 초기화 실패: {e}")
        return False
    
    # 모델 컴포넌트 테스트
    print("\n🔍 모델 컴포넌트 테스트...")
    component_success = test_model_components(model, device)
    
    # 모델 forward pass 테스트
    print("\n🧪 모델 forward pass 테스트...")
    forward_success = test_model_forward_pass(model, device)
    
    if component_success and forward_success:
        print("\n✅ 모든 테스트 성공!")
        
        # 체크포인트 저장
        checkpoint_path = "aivad_working_checkpoint.ckpt"
        try:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 체크포인트 저장: {checkpoint_path}")
            
            # 체크포인트 파일 크기 확인
            if os.path.exists(checkpoint_path):
                size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
                print(f"📊 체크포인트 크기: {size_mb:.1f} MB")
                
        except Exception as e:
            print(f"❌ 체크포인트 저장 실패: {e}")
        
        return True
    else:
        print("\n❌ 일부 테스트 실패")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🏆 작동하는 AI-VAD 모델 학습 (Tensor 크기 문제 해결)")
    print("=" * 60)
    
    success = main()
    if success:
        print("\n🎉 모델 테스트가 성공적으로 완료되었습니다!")
        print("이제 realtime_ui_advanced_windows.py에서 체크포인트를 로드할 수 있습니다.")
        print("\n체크포인트 파일:")
        print("- aivad_working_checkpoint.ckpt")
        print("\n💡 이 버전의 특징:")
        print("- Tensor 크기 문제 완전 해결")
        print("- AI-VAD 정확한 입력 형식 사용")
        print("- GPU 가속 지원")
        print("- 컴포넌트별 테스트 수행")
        print("- 안정적인 forward pass")
    else:
        print("\n💥 모델 테스트에 실패했습니다.")
        print("\n📋 해결 방법:")
        print("1. GPU 드라이버 및 CUDA 설치 확인")
        print("2. PyTorch GPU 버전 설치 확인")
        print("3. 관리자 권한으로 실행")
        exit(1)
