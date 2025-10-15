# 🚀 GPU 문제 해결 가이드

## 🚨 GPU가 인식되지 않는 문제

### 현재 상황:
```
GPU available: False, used: False
```

## 🔍 문제 진단

### 1. GPU 진단 도구 실행
```cmd
python gpu_diagnostic.py
```

이 도구는 다음을 확인합니다:
- NVIDIA 드라이버 설치 상태
- CUDA 툴킷 설치 상태  
- PyTorch GPU 지원 여부
- Anomalib GPU 지원 여부

## ✅ 해결 방법

### 1단계: NVIDIA 드라이버 확인 및 설치

#### 현재 드라이버 확인:
```cmd
nvidia-smi
```

#### 드라이버 설치:
1. **NVIDIA 공식 웹사이트** 방문: https://www.nvidia.com/drivers/
2. **GPU 모델에 맞는 최신 드라이버** 다운로드
3. **드라이버 설치** 후 시스템 재부팅

### 2단계: CUDA 툴킷 설치

#### CUDA 11.8 설치 (권장):
1. **CUDA 11.8 다운로드**: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. **Windows x86_64** 버전 선택
3. **exe (local)** 설치 파일 다운로드
4. **관리자 권한으로 설치**

#### CUDA 12.1 설치 (최신):
1. **CUDA 12.1 다운로드**: https://developer.nvidia.com/cuda-12-1-0-download-archive
2. **Windows x86_64** 버전 선택
3. **exe (local)** 설치 파일 다운로드
4. **관리자 권한으로 설치**

### 3단계: PyTorch GPU 버전 설치

#### CUDA 11.8용 PyTorch:
```cmd
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1용 PyTorch:
```cmd
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4단계: 환경 변수 설정

#### 시스템 환경 변수 추가:
1. **시스템 속성** → **고급** → **환경 변수**
2. **시스템 변수**에서 **새로 만들기**:
   - 변수 이름: `CUDA_PATH`
   - 변수 값: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8` (CUDA 버전에 따라 조정)
3. **PATH** 변수에 추가:
   - `%CUDA_PATH%\bin`
   - `%CUDA_PATH%\libnvvp`

### 5단계: 설치 확인

#### CUDA 설치 확인:
```cmd
nvcc --version
nvidia-smi
```

#### PyTorch GPU 확인:
```cmd
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🔧 자동 해결 스크립트

### GPU 설정 스크립트:
```cmd
# GPU 진단 실행
python gpu_diagnostic.py

# 완벽한 학습 스크립트 실행 (GPU 자동 감지)
python train_custom_perfect.py

# 또는 원클릭 실행
run_windows_perfect.bat
```

## 📊 GPU 성능 최적화

### 학습 시 GPU 사용 확인:
```python
# train_custom_perfect.py에서 자동으로 다음을 수행:
1. GPU 가용성 자동 감지
2. GPU 메모리 최적화
3. 16비트 혼합 정밀도 사용 (GPU)
4. 32비트 정밀도 사용 (CPU)
5. cuDNN 최적화 설정
```

### GPU 메모리 관리:
```python
# 자동으로 수행되는 최적화:
- GPU 메모리 캐시 정리
- 배치 크기 자동 조정
- 메모리 사용량 모니터링
```

## 🆘 문제가 지속될 경우

### 1. 시스템 요구사항 확인:
- **NVIDIA GPU** (GTX 1060 이상 권장)
- **8GB RAM** 이상
- **Windows 10/11** 64비트
- **DirectX 12** 지원

### 2. 충돌하는 소프트웨어 확인:
- **바이러스 백신** 실시간 보호 일시 중지
- **Windows Defender** 예외 설정
- **다른 GPU 소프트웨어** 종료

### 3. 완전 재설치:
```cmd
# 1. 모든 CUDA 관련 소프트웨어 제거
# 2. NVIDIA 드라이버 완전 제거
# 3. 시스템 재부팅
# 4. 최신 드라이버 설치
# 5. CUDA 툴킷 설치
# 6. PyTorch GPU 버전 설치
```

## 💡 성능 향상 팁

### GPU 사용 시:
- **16비트 혼합 정밀도** 사용으로 메모리 절약
- **배치 크기** 늘리기 (GPU 메모리에 따라)
- **cuDNN 벤치마킹** 활성화

### CPU 사용 시:
- **배치 크기** 줄이기
- **워커 수** 조정
- **32비트 정밀도** 사용

## 📈 성능 비교

| 설정 | 학습 시간 | 메모리 사용량 | 정확도 |
|------|-----------|---------------|--------|
| GPU (16-bit) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GPU (32-bit) | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CPU (32-bit) | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎯 권장 설정

### 최적의 GPU 설정:
```python
# train_custom_perfect.py에서 자동 설정됨
precision="16-mixed"  # GPU에서 메모리 효율성
strategy="auto"       # 자동 최적화
gradient_clip_val=1.0 # 안정적인 학습
```

### 최적의 CPU 설정:
```python
# train_custom_perfect.py에서 자동 설정됨
precision=32          # CPU에서 안정성
num_workers=0         # Windows 호환성
batch_size=4          # 메모리 효율성
```

---

## 🏆 결론

`train_custom_perfect.py`는 GPU와 CPU를 자동으로 감지하고 최적화된 설정으로 학습을 진행합니다. GPU가 사용 가능하면 자동으로 GPU 가속을 사용하고, 그렇지 않으면 CPU에서 안정적으로 실행됩니다.

**GPU 사용을 위해서는 위의 단계를 순서대로 따라하시면 됩니다!** 🚀
