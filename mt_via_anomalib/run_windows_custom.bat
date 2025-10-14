@echo off
chcp 65001 > nul
echo 🚀 Windows용 커스텀 비디오 데이터셋 AI-VAD 학습
echo ==================================================

REM 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ 관리자 권한으로 실행 중
) else (
    echo ⚠️  관리자 권한이 필요합니다.
    echo 이 스크립트를 관리자로 실행하세요.
    echo.
    echo 해결 방법:
    echo 1. 명령 프롬프트를 관리자로 실행
    echo 2. 이 스크립트를 다시 실행
    pause
    exit /b 1
)

REM 권한 문제 해결
echo.
echo 🔧 Windows 권한 문제를 해결합니다...
call fix_windows_permissions.bat

REM Python 환경 활성화
echo.
echo 🐍 Python 환경을 활성화합니다...
call conda activate mt_p310
if %errorLevel% NEQ 0 (
    echo ❌ mt_p310 환경 활성화 실패
    echo 다음 명령어로 환경을 생성하세요:
    echo conda create -n mt_p310 python=3.10
    pause
    exit /b 1
)

REM 학습 실행
echo.
echo 🎯 AI-VAD 모델 학습을 시작합니다...
python train_custom_windows_fix.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==================================================
    echo ❌ 학습 중 오류가 발생했습니다.
    echo ==================================================
    echo.
    echo 💡 해결 방법:
    echo 1. train_custom_windows_fix.py 파일에서 video_files 리스트에 비디오 파일 경로를 추가하세요
    echo 2. Windows 경로 예시:
    echo    "C:\\Users\\YourName\\Videos\\normal_video1.mp4"
    echo    "D:\\SecurityCameras\\normal_footage.mp4"
    echo 3. 비디오 파일이 존재하는지 확인하세요
    echo 4. 지원되는 형식: .mp4, .avi, .mov, .mkv, .flv, .wmv
    echo.
    echo 📚 더 자세한 정보는 CUSTOM_DATASET_GUIDE.md 파일을 참조하세요.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo 🎉 학습이 성공적으로 완료되었습니다!
echo ==================================================
echo.
echo 📁 생성된 파일:
echo - aivad_custom_windows_checkpoint.ckpt (학습된 모델)
echo.
echo 🚀 다음 단계:
echo 1. realtime_ui_advanced_windows.py 실행
echo 2. "체크포인트 로드" 버튼 클릭
echo 3. aivad_custom_windows_checkpoint.ckpt 파일 선택
echo 4. 웹캠이나 영상 파일로 실시간 이상 탐지 테스트
echo.
echo 📖 사용법은 WINDOWS_SETUP_GUIDE.md 파일을 참조하세요.
echo.
pause
