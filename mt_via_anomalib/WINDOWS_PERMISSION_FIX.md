# Windows 권한 문제 해결 가이드

이 가이드는 Windows에서 anomalib 학습 시 발생하는 권한 오류를 해결하는 방법을 설명합니다.

## 🚨 발생하는 오류

```
[WinError 1314] 클라이언트가 필요한 권한을 가지고 있지 않습니다: 
'C:\\Users\\User\\...\\results\\AiVad\\Avenue\\v0' -> 
'C:\\Users\\User\\...\\results\\AiVad\\Avenue\\latest'
```

## 🔧 해결 방법

### 방법 1: 자동 해결 (권장)

1. **관리자 권한으로 명령 프롬프트 실행**
   - `Win + R` → `cmd` → `Ctrl + Shift + Enter`

2. **권한 문제 해결 스크립트 실행**
   ```cmd
   cd C:\path\to\your\project\mt_via_anomalib
   fix_windows_permissions.bat
   ```

3. **원클릭 학습 실행**
   ```cmd
   run_windows_custom.bat
   ```

### 방법 2: 수동 해결

#### 1단계: 관리자 권한으로 실행
- 명령 프롬프트를 관리자로 실행
- PowerShell을 관리자로 실행

#### 2단계: 기존 폴더 삭제
```cmd
# 프로젝트 디렉토리로 이동
cd C:\path\to\your\project\mt_via_anomalib

# 기존 결과 폴더 삭제
rmdir /s /q results
rmdir /s /q custom_results
```

#### 3단계: 폴더 권한 설정
```cmd
# 현재 폴더 권한을 전체 제어로 설정
icacls . /grant Everyone:F /T /Q
```

#### 4단계: 학습 실행
```cmd
conda activate mt_p310
python train_custom_windows_fix.py
```

### 방법 3: 환경 변수 설정

시스템 환경 변수를 설정하여 권한 문제를 우회:

```cmd
# 환경 변수 설정
set ANOMALIB_LOGGER=false
set ANOMALIB_RESULTS_PATH=C:\temp\anomalib_results

# 학습 실행
python train_custom_windows_fix.py
```

### 방법 4: 다른 드라이브 사용

C: 드라이브 대신 D: 드라이브를 사용:

```cmd
# D: 드라이브로 이동
D:
cd D:\Projects\mt_via_anomalib

# 학습 실행
python train_custom_windows_fix.py
```

## 🛡️ 예방 방법

### 1. 바이러스 백신 예외 설정
- Windows Defender 또는 설치된 백신 프로그램에서 프로젝트 폴더를 예외 목록에 추가
- 실시간 보호 일시 중지 (학습 중에만)

### 2. 폴더 권한 미리 설정
```cmd
# 프로젝트 폴더 권한을 미리 설정
icacls "C:\path\to\your\project" /grant Everyone:F /T /Q
```

### 3. 임시 폴더 사용
```python
# train_custom_windows_fix.py에서 결과 경로 변경
import tempfile
temp_dir = tempfile.mkdtemp()
os.environ["ANOMALIB_RESULTS_PATH"] = temp_dir
```

## 🔍 문제 진단

### 권한 확인
```cmd
# 현재 폴더의 권한 확인
icacls .

# 특정 사용자의 권한 확인
icacls . | findstr "YourUsername"
```

### 프로세스 확인
```cmd
# anomalib 관련 프로세스 확인
tasklist | findstr python
```

### 디스크 공간 확인
```cmd
# 디스크 공간 확인
dir C:\ | findstr "bytes free"
```

## 📋 체크리스트

학습 전에 다음 사항들을 확인하세요:

- [ ] 명령 프롬프트를 관리자 권한으로 실행
- [ ] 프로젝트 폴더의 권한이 전체 제어로 설정됨
- [ ] 바이러스 백신이 프로젝트 폴더를 예외 처리
- [ ] 충분한 디스크 공간 (최소 10GB)
- [ ] 비디오 파일 경로가 올바름
- [ ] Python 환경이 활성화됨 (mt_p310)

## 🆘 추가 도움

### 로그 확인
학습 중 오류가 발생하면 다음 로그를 확인하세요:

1. **콘솔 출력**: 터미널에 표시되는 오류 메시지
2. **Windows 이벤트 뷰어**: `eventvwr.msc` 실행
3. **Python 로그**: `custom_results` 폴더의 로그 파일

### 일반적인 오류와 해결책

#### "Access Denied" 오류
```cmd
# 해결: 관리자 권한으로 실행
runas /user:Administrator cmd
```

#### "File in use" 오류
```cmd
# 해결: 관련 프로세스 종료
taskkill /f /im python.exe
```

#### "Disk space" 오류
```cmd
# 해결: 디스크 정리
cleanmgr /sagerun:1
```

#### "Path too long" 오류
```cmd
# 해결: 짧은 경로 사용
subst X: C:\very\long\path\to\project
cd X:
```

## 📞 지원

문제가 지속되면 다음을 시도하세요:

1. **다른 폴더에서 실행**: 바탕화면이나 C:\temp에서 실행
2. **다른 사용자로 실행**: 새 Windows 사용자 계정 생성
3. **가상머신 사용**: VMware나 VirtualBox에서 실행
4. **WSL 사용**: Windows Subsystem for Linux에서 실행

## 🎯 성공 확인

학습이 성공적으로 완료되면 다음 파일들이 생성됩니다:

- `aivad_custom_windows_checkpoint.ckpt` (학습된 모델)
- `custom_results/` 폴더 (학습 결과)
- `custom_video_dataset/` 폴더 (처리된 데이터셋)

이제 `realtime_ui_advanced_windows.py`에서 체크포인트를 로드하여 실시간 이상 탐지를 사용할 수 있습니다!
