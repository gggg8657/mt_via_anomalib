"""
GPU 감지 문제 해결 및 최적화 스크립트
"""

import os
import torch
import subprocess
import sys

def check_nvidia_smi():
    """nvidia-smi로 GPU 정보 확인"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("=== nvidia-smi 출력 ===")
        print(result.stdout)
        return True
    except FileNotFoundError:
        print("❌ nvidia-smi를 찾을 수 없습니다.")
        return False

def check_pytorch_gpu():
    """PyTorch GPU 감지 확인"""
    print("\n=== PyTorch GPU 정보 ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  - 메모리: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - 멀티프로세서: {props.multi_processor_count}개")

def fix_gpu_detection():
    """GPU 감지 문제 해결"""
    print("\n=== GPU 감지 문제 해결 ===")
    
    # 1. 환경 변수 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # 모든 GPU 사용 가능하게 설정
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # 2. PyTorch 재초기화
    if 'torch.cuda' in sys.modules:
        del sys.modules['torch.cuda']
        import importlib
        importlib.reload(torch.cuda)
    
    # 3. GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("✅ GPU 환경 설정 완료")

def test_gpu_usage():
    """GPU 사용 테스트"""
    print("\n=== GPU 사용 테스트 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다.")
        return False
    
    try:
        # 간단한 텐서 연산으로 GPU 테스트
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        
        print(f"✅ GPU 0 테스트 성공: {z.shape}")
        print(f"GPU 0 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        # 여러 GPU 테스트
        for i in range(min(4, torch.cuda.device_count())):
            try:
                device = torch.device(f"cuda:{i}")
                x = torch.randn(100, 100, device=device)
                y = torch.randn(100, 100, device=device)
                z = torch.mm(x, y)
                print(f"✅ GPU {i} 테스트 성공: {z.shape}")
            except Exception as e:
                print(f"❌ GPU {i} 테스트 실패: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU 테스트 실패: {e}")
        return False

def optimize_for_h100():
    """H100 GPU 최적화 설정"""
    print("\n=== H100 GPU 최적화 ===")
    
    if not torch.cuda.is_available():
        return
    
    # H100 특화 설정
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Mixed precision 설정
    torch.set_float32_matmul_precision('medium')
    
    # 메모리 풀 최적화
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    print("✅ H100 최적화 설정 완료")

def create_gpu_test_script():
    """GPU 테스트 스크립트 생성"""
    script_content = '''
import torch
import time

def benchmark_gpu():
    """GPU 벤치마크 테스트"""
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다.")
        return
    
    device = torch.device("cuda:0")
    print(f"GPU 벤치마크 시작: {torch.cuda.get_device_name(0)}")
    
    # 메모리 사용량 측정
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 큰 행렬 곱셈 테스트
    size = 4096
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)
    
    # 워밍업
    for _ in range(3):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    
    # 실제 벤치마크
    start_time = time.time()
    for _ in range(10):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"평균 실행 시간: {avg_time*1000:.2f} ms")
    print(f"최대 메모리 사용량: {peak_memory:.1f} MB")
    print(f"메모리 효율성: {peak_memory/(size*size*4*2/1024**2):.1f}x")

if __name__ == "__main__":
    benchmark_gpu()
'''
    
    with open("gpu_benchmark.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ GPU 벤치마크 스크립트 생성: gpu_benchmark.py")

def main():
    print("🔧 GPU 감지 및 최적화 도구")
    print("=" * 50)
    
    # 1. nvidia-smi 확인
    check_nvidia_smi()
    
    # 2. PyTorch GPU 정보
    check_pytorch_gpu()
    
    # 3. GPU 감지 문제 해결
    fix_gpu_detection()
    
    # 4. H100 최적화
    optimize_for_h100()
    
    # 5. GPU 사용 테스트
    test_success = test_gpu_usage()
    
    # 6. 벤치마크 스크립트 생성
    create_gpu_test_script()
    
    print("\n" + "=" * 50)
    print("📋 요약:")
    
    if test_success:
        print("✅ GPU가 정상적으로 작동합니다!")
        print("이제 AI-VAD 학습을 시작할 수 있습니다.")
        print("\n다음 명령으로 학습을 시작하세요:")
        print("python train_aivad.py")
    else:
        print("❌ GPU 사용에 문제가 있습니다.")
        print("다음을 확인해주세요:")
        print("1. NVIDIA 드라이버 설치")
        print("2. CUDA 설치")
        print("3. PyTorch CUDA 버전 호환성")
    
    print("\n💡 GPU 벤치마크 테스트:")
    print("python gpu_benchmark.py")

if __name__ == "__main__":
    main()
