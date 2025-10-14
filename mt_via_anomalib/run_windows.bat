@echo off
REM AI-VAD Realtime UI Windows 실행 배치 파일
REM 이 파일을 더블클릭하여 프로그램을 실행할 수 있습니다.

echo ========================================
echo AI-VAD Realtime Anomaly Detection
echo Windows 실행 스크립트
echo ========================================
echo.

REM 현재 디렉토리로 이동
cd /d "%~dp0"

REM Python 설치 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되지 않았거나 PATH에 추가되지 않았습니다.
    echo Python 3.8 이상을 설치하고 PATH에 추가해주세요.
    echo https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

REM 가상환경 확인 및 활성화
if exist "anomalib_env\Scripts\activate.bat" (
    echo 가상환경 활성화 중...
    call anomalib_env\Scripts\activate.bat
) else (
    echo 가상환경을 찾을 수 없습니다. 새로 생성합니다...
    python -m venv anomalib_env
    call anomalib_env\Scripts\activate.bat
    echo 필수 패키지 설치 중... (시간이 걸릴 수 있습니다)
    pip install -r requirements_windows.txt
)

REM 시스템 정보 출력
echo.
echo 시스템 정보:
python -c "import torch, platform; print(f'OS: {platform.system()} {platform.release()}'); print(f'Python: {platform.python_version()}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo.

REM GPU 확인
python -c "import torch; print(f'GPU 사용 가능: {torch.cuda.is_available()}'); print(f'GPU 이름: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>nul
if errorlevel 1 (
    echo GPU 정보를 확인할 수 없습니다. CPU 모드로 실행됩니다.
)

echo.
echo 프로그램을 시작합니다...
echo.

REM 프로그램 실행
python realtime_ui_advanced_windows.py

REM 오류 발생시 메시지
if errorlevel 1 (
    echo.
    echo [오류] 프로그램 실행 중 오류가 발생했습니다.
    echo.
    echo 가능한 해결 방법:
    echo 1. requirements_windows.txt의 패키지들을 설치했는지 확인
    echo 2. GPU 드라이버가 최신인지 확인
    echo 3. WINDOWS_SETUP_GUIDE.md 파일을 참고하세요
    echo.
    pause
)

echo.
echo 프로그램이 종료되었습니다.
pause
