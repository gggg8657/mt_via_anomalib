# AI-VAD 학습 오류 해결 가이드

## 🚨 발생한 오류들

### 1. Windows 권한 오류
```
[WinError 1314] 클라이언트가 필요한 권한을 가지고 있지 않습니다: 
'C:\\Users\\User\\...\\results\\AiVad\\Avenue\\v0' -> 
'C:\\Users\\User\\...\\results\\AiVad\\Avenue\\latest'
```

### 2. CustomVideoDataModule 오류
```
AttributeError: 'CustomVideoDataModule' object has no attribute 'train_data'
```

### 3. pandas DataFrame 오류
```
ValueError: cannot set a frame with no defined index and a scalar
```

### 4. 변수 스코프 오류
```
name 'j' is not defined
```

### 5. 경로 처리 오류
```
unsupported operand type(s) for /: 'str' and 'str'
```

## ✅ 해결된 방법

### 🔧 수정된 스크립트 사용

**`train_custom_ultimate.py`** - 궁극의 안정 버전 ⭐
- 모든 알려진 오류 해결 (Windows 권한, CustomVideoDataModule, pandas DataFrame, 변수 스코프, 경로 처리)
- Avenue 데이터셋의 실제 구조 완벽 모방
- training_videos, testing_videos, ground_truth_demo 완전 구현
- Windows 경로 처리 최적화
- 변수 스코프 오류 해결
- 최고 수준의 안정성과 호환성

**`train_custom_final.py`** - 이전 안정적인 버전
- Avenue 데이터셋의 완전한 구조 모방
- pandas DataFrame 오류 해결
- training_videos, testing_videos, ground_truth_demo 폴더 생성
- 더미 메타데이터 파일 생성
- 최고 수준의 안정성 확보

**`train_custom_fixed.py`** - 이전 안정적인 버전
- CustomVideoDataModule 대신 기존 Avenue 데이터 모듈 사용
- train_data 속성 오류 해결
- Windows 권한 문제 해결

### 🚀 사용 방법

#### 1단계: 비디오 파일 준비
```python
# train_custom_ultimate.py에서 수정
video_files = [
    "C:\\Users\\YourName\\Videos\\normal_video1.mp4",
    "D:\\SecurityCameras\\normal_footage.mp4",
    "E:\\MyVideos\\normal_clip.avi",
]
```

#### 2단계: 실행
```cmd
# 관리자 권한으로 명령 프롬프트 실행 후
python train_custom_ultimate.py

# 또는 원클릭 실행
run_windows_ultimate.bat
```

## 📁 사용 가능한 스크립트들

### 1. `train_custom_ultimate.py` ⭐ (궁극의 권장)
- **장점**: 궁극의 안정성, 모든 오류 해결됨
- **사용법**: 비디오 파일 경로만 추가하면 됨
- **결과**: `aivad_custom_ultimate_checkpoint.ckpt`
- **특징**: 모든 알려진 오류 해결, Avenue 데이터셋 완벽 모방

### 2. `train_custom_final.py` (이전 최고 버전)
- **장점**: 최고 수준의 안정성, 대부분 오류 해결됨
- **사용법**: 비디오 파일 경로만 추가하면 됨
- **결과**: `aivad_custom_final_checkpoint.ckpt`
- **특징**: Avenue 데이터셋 완전 모방, pandas 오류 해결

### 3. `train_custom_fixed.py` (이전 버전)
- **장점**: 안정적, 기본 오류 해결됨
- **사용법**: 비디오 파일 경로만 추가하면 됨
- **결과**: `aivad_custom_fixed_checkpoint.ckpt`

### 4. `train_custom_windows_fix.py`
- **장점**: Windows 권한 문제 해결
- **단점**: CustomVideoDataModule 오류 가능성
- **사용법**: Windows 특화 설정

### 5. `train_custom_simple.py`
- **장점**: 간단한 구조
- **단점**: 일부 오류 가능성
- **사용법**: 기본적인 커스텀 데이터셋

### 6. `train_aivad_simple.py`
- **장점**: Avenue 데이터셋 사용, 매우 안정적
- **단점**: Avenue 데이터셋에만 제한
- **사용법**: 기본 AI-VAD 학습

## 🎯 권장 사용 순서

1. **첫 번째 시도**: `train_custom_ultimate.py` (궁극의 안정적) ⭐
2. **두 번째 시도**: `train_custom_final.py` (최고 안정적)
3. **세 번째 시도**: `train_custom_fixed.py` (안정적)
4. **네 번째 시도**: `train_aivad_simple.py` (Avenue 데이터셋으로 테스트)
5. **다섯 번째 시도**: `train_custom_windows_fix.py` (Windows 특화)

## 📋 체크리스트

### 학습 전 확인사항
- [ ] 관리자 권한으로 실행
- [ ] 비디오 파일 경로가 올바름
- [ ] 비디오 파일이 존재함
- [ ] 지원되는 형식 (.mp4, .avi, .mov, .mkv, .flv, .wmv)
- [ ] 충분한 디스크 공간 (최소 5GB)
- [ ] Python 환경 활성화 (mt_p310)

### 오류 발생 시 확인사항
- [ ] 오류 메시지 전체 확인
- [ ] 파일 경로에 한글이나 특수문자 없는지 확인
- [ ] 바이러스 백신이 파일을 차단하지 않는지 확인
- [ ] 비디오 파일이 손상되지 않았는지 확인

## 🔍 문제 진단

### 권한 문제 진단
```cmd
# 관리자 권한 확인
net session

# 폴더 권한 확인
icacls .

# 기존 결과 폴더 정리
rmdir /s /q results
rmdir /s /q custom_results
```

### 파일 문제 진단
```cmd
# 비디오 파일 존재 확인
dir "C:\path\to\your\video.mp4"

# 파일 크기 확인
dir "C:\path\to\your\video.mp4" | findstr "bytes"
```

### 환경 문제 진단
```cmd
# Python 버전 확인
python --version

# Conda 환경 확인
conda env list

# 패키지 설치 확인
python -c "import anomalib, torch"
```

## 💡 추가 팁

### 성공적인 학습을 위한 팁
1. **비디오 품질**: 720p 이하, 10-30초 길이 권장
2. **파일 형식**: .mp4 형식이 가장 안정적
3. **파일 경로**: 공백이나 특수문자 없는 경로 사용
4. **디스크 공간**: 충분한 여유 공간 확보
5. **백그라운드 프로그램**: 불필요한 프로그램 종료

### Windows 특화 팁
1. **관리자 권한**: 반드시 관리자로 실행
2. **바이러스 백신**: 프로젝트 폴더 예외 설정
3. **Windows Defender**: 실시간 보호 일시 중지
4. **파일 경로**: 짧은 경로 사용 (C:\temp 등)

## 🆘 최종 해결책

모든 방법이 실패할 경우:

1. **새로운 환경에서 실행**:
   ```cmd
   # 새 폴더에서 실행
   mkdir C:\temp\aivad_test
   cd C:\temp\aivad_test
   # 여기에 파일들 복사 후 실행
   ```

2. **WSL 사용**:
   ```bash
   # Windows Subsystem for Linux에서 실행
   wsl
   cd /mnt/c/path/to/project
   python train_custom_fixed.py
   ```

3. **Docker 사용**:
   ```bash
   # Docker 컨테이너에서 실행
   docker run -it --gpus all anomalib:latest
   ```

## 📞 지원

문제가 지속되면 다음 정보와 함께 문의하세요:

1. **운영체제**: Windows 버전
2. **Python 버전**: `python --version`
3. **오류 메시지**: 전체 스택 트레이스
4. **비디오 파일 정보**: 형식, 크기, 경로
5. **시도한 방법들**: 어떤 스크립트들을 시도했는지

성공적인 학습을 위해 `train_custom_fixed.py`를 우선적으로 사용하시기 바랍니다! 🎯
