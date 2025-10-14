# AI-VAD Realtime UI Windows 설치 및 실행 가이드

## 🚀 빠른 시작

### 1. 시스템 요구사항
- **운영체제**: Windows 10/11 (64-bit)
- **Python**: 3.8 이상
- **GPU**: NVIDIA GPU (CUDA 지원, 선택사항)
- **RAM**: 최소 8GB (16GB 권장)
- **저장공간**: 최소 10GB 여유공간

### 2. 필수 소프트웨어 설치

#### Python 설치
```bash
# Python 3.9 이상 다운로드 및 설치
# https://www.python.org/downloads/windows/
# 설치 시 "Add Python to PATH" 체크박스 선택
```

#### CUDA 설치 (GPU 사용시)
```bash
# NVIDIA CUDA Toolkit 11.8 또는 12.0 설치
# https://developer.nvidia.com/cuda-downloads

# cuDNN 설치
# https://developer.nvidia.com/cudnn
```

### 3. 프로젝트 설정

#### 가상환경 생성
```bash
# 명령 프롬프트 또는 PowerShell에서 실행
cd C:\your\project\path
python -m venv anomalib_env
anomalib_env\Scripts\activate
```

#### 패키지 설치
```bash
# 필수 패키지 설치
pip install -r requirements_windows.txt

# 또는 개별 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install anomalib opencv-python PySide6 numpy scipy
```

### 4. 실행 방법

#### 기본 실행
```bash
python realtime_ui_advanced_windows.py
```

#### GPU 사용 실행
```bash
# GPU 사용 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# GPU 모드로 실행
python realtime_ui_advanced_windows.py
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. CUDA 오류
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# 버전 불일치시 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. OpenCV 카메라 오류
```bash
# DirectShow 백엔드 사용 확인
python -c "import cv2; print(cv2.getBuildInformation())"

# 카메라 테스트
python -c "import cv2; cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); print(cap.isOpened())"
```

#### 3. PySide6 설치 오류
```bash
# Qt 라이브러리 설치
pip install --upgrade PySide6

# 또는 conda 사용
conda install -c conda-forge pyside6
```

#### 4. 메모리 부족 오류
```bash
# 배치 크기 줄이기
# 코드에서 fps_limit을 낮춤 (예: 15 FPS)

# GPU 메모리 정리
python -c "import torch; torch.cuda.empty_cache()"
```

### 성능 최적화

#### GPU 메모리 최적화
```python
# realtime_ui_advanced_windows.py에서 설정
torch.backends.cudnn.benchmark = True  # 고정 입력 크기시
torch.backends.cudnn.deterministic = False  # 성능 우선시
```

#### CPU 모드 실행
```bash
# CPU만 사용하는 경우
export CUDA_VISIBLE_DEVICES=""
python realtime_ui_advanced_windows.py
```

## 📁 파일 구조

```
mt_via_anomalib/
├── realtime_ui_advanced_windows.py  # 윈도우즈 최적화 버전
├── realtime_ui_advanced.py          # 원본 버전
├── requirements_windows.txt         # 윈도우즈 의존성
├── WINDOWS_SETUP_GUIDE.md          # 이 가이드
└── checkpoints/                     # 모델 체크포인트 (선택사항)
    └── aivad_model.ckpt
```

## 🎯 사용법

### 1. 영상 파일 분석
1. 프로그램 실행
2. "영상 파일" 버튼 클릭
3. 분석할 비디오 파일 선택
4. "재생" 버튼 클릭

### 2. 웹캠 실시간 분석
1. 프로그램 실행
2. "웹캠" 버튼 클릭
3. 사용할 카메라 선택
4. "재생" 버튼 클릭

### 3. 고급 설정
- **임계치 조정**: 이상 탐지 민감도 설정
- **적응 임계치**: 자동으로 임계치 조정
- **시각화 옵션**: 박스, 마스크, 히트맵 표시
- **FPS 제한**: 성능과 정확도 균형

## 📊 성능 벤치마크

### 권장 하드웨어별 설정

#### 고성능 GPU (RTX 3080/4080 이상)
- FPS: 30-60
- 해상도: 1280x720
- 모든 시각화 옵션 활성화

#### 중급 GPU (GTX 1660/RTX 3060)
- FPS: 15-30
- 해상도: 854x480
- 히트맵만 활성화

#### CPU만 사용
- FPS: 5-10
- 해상도: 640x360
- 최소 시각화

## 🆘 지원 및 문의

### 로그 확인
```bash
# 실행 로그 확인
python realtime_ui_advanced_windows.py > log.txt 2>&1

# 오류 발생시 로그 파일 확인
type log.txt
```

### 시스템 정보 수집
```bash
# 시스템 정보 확인
python -c "
import torch
import cv2
import platform
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'OpenCV: {cv2.__version__}')
print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')
"
```

## 🔄 업데이트

### 패키지 업데이트
```bash
# 가상환경 활성화
anomalib_env\Scripts\activate

# 패키지 업데이트
pip install --upgrade -r requirements_windows.txt

# anomalib 업데이트
pip install --upgrade anomalib
```

### 코드 업데이트
```bash
# Git을 사용하는 경우
git pull origin main

# 또는 새 버전 다운로드 후 교체
```
