# 커스텀 비디오 데이터셋으로 AI-VAD 학습 가이드

이 가이드는 Avenue 데이터셋 대신 자신의 비디오 데이터로 AI-VAD 모델을 학습시키는 방법을 설명합니다.

## 📋 목차

1. [준비사항](#준비사항)
2. [데이터셋 구조](#데이터셋-구조)
3. [학습 방법](#학습-방법)
4. [고급 설정](#고급-설정)
5. [문제 해결](#문제-해결)

## 🎯 준비사항

### 필요한 것들
- Python 3.8-3.10
- anomalib 2.1.0 이상
- PyTorch (CUDA 지원 권장)
- 비디오 파일들 (정상 상황)

### 지원되는 비디오 형식
- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.flv`
- `.wmv`

## 📁 데이터셋 구조

### 간단한 방법 (권장)
```
your_videos/
├── normal_video1.mp4
├── normal_video2.avi
├── normal_video3.mov
└── ...
```

### 고급 방법
```
custom_dataset/
├── train/
│   ├── normal/          # 정상 비디오들
│   │   ├── video1.mp4
│   │   └── video2.avi
│   └── abnormal/        # 이상 비디오들 (선택사항)
│       └── anomaly.mp4
├── val/                 # 검증 데이터 (선택사항)
│   ├── normal/
│   └── abnormal/
└── test/                # 테스트 데이터 (선택사항)
    ├── normal/
    └── abnormal/
```

## 🚀 학습 방법

### 1단계: 스크립트 수정

`train_custom_simple.py` 파일을 열어서 다음 부분을 수정하세요:

```python
# ===== 여기를 수정하세요 =====
# 1. 비디오 파일 경로들을 여기에 추가하세요
video_files = [
    "/path/to/your/video1.mp4",
    "/path/to/your/video2.avi",
    "/path/to/your/video3.mov",
    # 더 많은 비디오 파일들...
]

# 2. 데이터셋이 저장될 경로
dataset_path = "./custom_video_dataset"

# 3. 학습 설정
max_epochs = 10        # 에포크 수 (더 많이 학습하려면 증가)
batch_size = 8         # 배치 크기 (GPU 메모리에 따라 조정)
# =============================
```

### 2단계: 학습 실행

```bash
# mt_p310 환경 활성화
conda activate mt_p310

# 학습 실행
python train_custom_simple.py
```

### 3단계: 체크포인트 사용

학습이 완료되면 `aivad_custom_checkpoint.ckpt` 파일이 생성됩니다. 이 파일을 `realtime_ui_advanced_windows.py`에서 로드하여 사용할 수 있습니다.

## ⚙️ 고급 설정

### 학습 파라미터 조정

```python
# train_custom_simple.py에서 수정 가능한 파라미터들

# 데이터 관련
clip_length_in_frames = 2        # 클립 길이 (프레임 수)
frames_between_clips = 1         # 클립 간 프레임 수
target_frame = VideoTargetFrame.LAST  # 타겟 프레임

# 학습 관련
max_epochs = 20                  # 에포크 수
batch_size = 8                   # 배치 크기
gradient_clip_val = 1.0          # 그래디언트 클리핑
num_workers = 4                  # 데이터 로더 워커 수

# 정밀도
precision = 32                   # 32: 정밀도, 16: 혼합 정밀도 (GPU 메모리 절약)
```

### GPU 메모리 최적화

```python
# GPU 메모리가 부족한 경우
batch_size = 4                   # 배치 크기 줄이기
precision = 16                   # 혼합 정밀도 사용
limit_train_batches = 50         # 훈련 배치 수 제한
```

### 더 많은 에포크로 학습

```python
# 더 정확한 모델을 원하는 경우
max_epochs = 50                  # 에포크 수 증가
limit_train_batches = None       # 배치 제한 해제
limit_val_batches = None         # 검증 배치 제한 해제
```

## 🔧 문제 해결

### 일반적인 오류들

#### 1. "비디오 파일을 찾을 수 없습니다"
```bash
# 해결 방법: 파일 경로가 올바른지 확인
ls -la /path/to/your/video.mp4
```

#### 2. "CUDA out of memory"
```python
# 해결 방법: 배치 크기 줄이기
batch_size = 2  # 또는 1

# 또는 혼합 정밀도 사용
precision = 16
```

#### 3. "데이터셋 로드 실패"
```python
# 해결 방법: 데이터셋 경로 확인
dataset_path = "/absolute/path/to/your/dataset"
```

#### 4. "비디오 형식이 지원되지 않음"
```bash
# 해결 방법: FFmpeg로 변환
ffmpeg -i input_video.xxx output_video.mp4
```

### 성능 최적화 팁

1. **비디오 전처리**
   - 해상도: 720p 이하 권장
   - 길이: 10-30초 클립으로 분할
   - 프레임율: 15-30 FPS

2. **데이터 증강**
   ```python
   # 필요시 데이터 증강 추가
   from torchvision import transforms
   
   train_augmentations = transforms.Compose([
       transforms.RandomHorizontalFlip(0.5),
       transforms.RandomRotation(5),
   ])
   ```

3. **검증 데이터 분리**
   - 훈련 데이터의 20%를 검증용으로 분리
   - 검증 데이터로 과적합 방지

## 📊 학습 모니터링

### 로그 확인
학습 중 다음 정보들이 출력됩니다:
- GPU 사용률
- 배치당 처리 시간
- 손실값 (Loss)
- 검증 정확도

### 체크포인트 저장
```python
# 추가 체크포인트 저장 (에포크마다)
checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints',
    filename='aivad-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss',
)
```

## 🎯 사용 예시

### 예시 1: 보안 카메라 데이터
```python
video_files = [
    "/data/security/normal_hallway_001.mp4",
    "/data/security/normal_hallway_002.mp4",
    "/data/security/normal_entrance_001.mp4",
    "/data/security/normal_entrance_002.mp4",
]
```

### 예시 2: 제조업체 품질 검사
```python
video_files = [
    "/data/quality/normal_product_001.mp4",
    "/data/quality/normal_product_002.mp4",
    "/data/quality/normal_assembly_001.mp4",
]
```

### 예시 3: 교통 모니터링
```python
video_files = [
    "/data/traffic/normal_intersection_001.mp4",
    "/data/traffic/normal_highway_001.mp4",
    "/data/traffic/normal_parking_001.mp4",
]
```

## 📞 추가 도움

문제가 발생하면 다음을 확인하세요:

1. **로그 확인**: 오류 메시지를 자세히 읽어보세요
2. **환경 확인**: Python, PyTorch, anomalib 버전 확인
3. **데이터 확인**: 비디오 파일이 손상되지 않았는지 확인
4. **리소스 확인**: GPU 메모리, 디스크 공간 확인

더 자세한 도움이 필요하면 anomalib 공식 문서를 참조하세요: https://anomalib.readthedocs.io/
