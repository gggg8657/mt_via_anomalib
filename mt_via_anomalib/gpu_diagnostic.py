"""
GPU 진단 도구 - Windows 환경에서 GPU 상태를 확인하고 최적화합니다
"""

import os
import subprocess
import sys

def check_nvidia_driver():
    """NVIDIA 드라이버 확인"""
    print("🔍 NVIDIA 드라이버 확인 중...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA 드라이버가 설치되어 있습니다.")
            print("📊 GPU 상태:")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA 드라이버가 설치되지 않았거나 nvidia-smi 명령어를 찾을 수 없습니다.")
            return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi 명령어 실행 시간 초과")
        return False
    except FileNotFoundError:
        print("❌ nvidia-smi 명령어를 찾을 수 없습니다. NVIDIA 드라이버가 설치되지 않았을 수 있습니다.")
        return False
    except Exception as e:
        print(f"❌ NVIDIA 드라이버 확인 중 오류: {e}")
        return False

def check_cuda_installation():
    """CUDA 설치 확인"""
    print("\n🔍 CUDA 설치 확인 중...")
    
    # CUDA 환경 변수 확인
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"✅ CUDA_PATH 환경 변수: {cuda_path}")
    else:
        print("⚠️  CUDA_PATH 환경 변수가 설정되지 않았습니다.")
    
    # CUDA 버전 확인
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CUDA 컴파일러 (nvcc)가 설치되어 있습니다.")
            print(f"📊 CUDA 버전 정보:\n{result.stdout}")
            return True
        else:
            print("❌ CUDA 컴파일러 (nvcc)를 찾을 수 없습니다.")
            return False
    except FileNotFoundError:
        print("❌ CUDA 컴파일러 (nvcc)를 찾을 수 없습니다. CUDA 툴킷이 설치되지 않았을 수 있습니다.")
        return False
    except Exception as e:
        print(f"❌ CUDA 확인 중 오류: {e}")
        return False

def check_pytorch_gpu():
    """PyTorch GPU 지원 확인"""
    print("\n🔍 PyTorch GPU 지원 확인 중...")
    
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
        
        # CUDA 가용성 확인
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 가용성: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"GPU 개수: {device_count}")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_capability = torch.cuda.get_device_capability(i)
                print(f"GPU {i}: {gpu_name}")
                print(f"  - 메모리: {gpu_memory:.1f} GB")
                print(f"  - Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
            
            # GPU 메모리 테스트
            print("\n🧪 GPU 메모리 테스트 중...")
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"✅ GPU 메모리 테스트 성공")
                print(f"  - 할당된 메모리: {gpu_memory_allocated:.3f} GB")
                print(f"  - 예약된 메모리: {gpu_memory_reserved:.3f} GB")
                del test_tensor
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                print(f"❌ GPU 메모리 테스트 실패: {e}")
                return False
        else:
            print("❌ PyTorch에서 CUDA를 사용할 수 없습니다.")
            return False
            
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
        return False
    except Exception as e:
        print(f"❌ PyTorch GPU 확인 중 오류: {e}")
        return False

def check_anomalib_gpu():
    """Anomalib GPU 지원 확인"""
    print("\n🔍 Anomalib GPU 지원 확인 중...")
    
    try:
        from anomalib.models.video import AiVad
        print("✅ Anomalib가 설치되어 있습니다.")
        
        # 간단한 모델 초기화 테스트
        try:
            model = AiVad()
            print("✅ AI-VAD 모델 초기화 성공")
            
            # GPU 사용 가능한지 확인
            if torch.cuda.is_available():
                model = model.cuda()
                print("✅ AI-VAD 모델이 GPU로 이동되었습니다.")
                return True
            else:
                print("⚠️  GPU를 사용할 수 없어 CPU에서 실행됩니다.")
                return False
                
        except Exception as e:
            print(f"❌ AI-VAD 모델 초기화 실패: {e}")
            return False
            
    except ImportError:
        print("❌ Anomalib가 설치되지 않았습니다.")
        return False
    except Exception as e:
        print(f"❌ Anomalib GPU 확인 중 오류: {e}")
        return False

def provide_gpu_solution():
    """GPU 사용을 위한 해결책 제시"""
    print("\n" + "="*60)
    print("💡 GPU 사용을 위한 해결책")
    print("="*60)
    
    print("\n1. NVIDIA 드라이버 설치:")
    print("   - NVIDIA 공식 웹사이트에서 최신 드라이버 다운로드")
    print("   - https://www.nvidia.com/drivers/")
    
    print("\n2. CUDA 툴킷 설치:")
    print("   - CUDA 11.8 또는 12.1 버전 설치 권장")
    print("   - https://developer.nvidia.com/cuda-downloads")
    
    print("\n3. PyTorch GPU 버전 설치:")
    print("   # CUDA 11.8용")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   ")
    print("   # CUDA 12.1용")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n4. 환경 변수 설정:")
    print("   - CUDA_PATH 환경 변수 설정")
    print("   - PATH에 CUDA bin 디렉토리 추가")
    
    print("\n5. 시스템 재부팅:")
    print("   - 드라이버 설치 후 시스템 재부팅 필요")

def main():
    print("🔧 GPU 진단 도구")
    print("="*60)
    
    # 각 단계별 확인
    nvidia_ok = check_nvidia_driver()
    cuda_ok = check_cuda_installation()
    pytorch_ok = check_pytorch_gpu()
    anomalib_ok = check_anomalib_gpu()
    
    # 전체 결과 요약
    print("\n" + "="*60)
    print("📊 진단 결과 요약")
    print("="*60)
    
    print(f"NVIDIA 드라이버: {'✅ OK' if nvidia_ok else '❌ FAIL'}")
    print(f"CUDA 설치: {'✅ OK' if cuda_ok else '❌ FAIL'}")
    print(f"PyTorch GPU: {'✅ OK' if pytorch_ok else '❌ FAIL'}")
    print(f"Anomalib GPU: {'✅ OK' if anomalib_ok else '❌ FAIL'}")
    
    if all([nvidia_ok, cuda_ok, pytorch_ok, anomalib_ok]):
        print("\n🎉 모든 GPU 구성 요소가 정상적으로 설치되어 있습니다!")
        print("GPU 가속 학습을 사용할 수 있습니다.")
    else:
        print("\n⚠️  GPU 구성 요소에 문제가 있습니다.")
        provide_gpu_solution()

if __name__ == "__main__":
    main()
