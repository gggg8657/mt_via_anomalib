@echo off
chcp 65001 > nul
echo 🧪 간단한 AI-VAD UI 테스트
echo ==========================================================

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

REM 간단한 UI 테스트 실행
echo.
echo 🧪 간단한 AI-VAD UI 테스트를 시작합니다...
python simple_ui_test.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================================
    echo ❌ UI 테스트 중 오류가 발생했습니다.
    echo ==========================================================
    echo.
    echo 💡 해결 방법:
    echo 1. PySide6 설치 확인: pip install PySide6
    echo 2. OpenCV 설치 확인: pip install opencv-python
    echo 3. Matplotlib 설치 확인: pip install matplotlib
    echo 4. 관리자 권한으로 실행
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo 🎉 간단한 UI 테스트 완료!
echo ==========================================================
echo.
echo 💡 이 테스트의 특징:
echo - 복잡한 AI-VAD 추론 없이 기본 UI 동작 확인
echo - 더미 점수 생성으로 시각화 테스트
echo - 모델 로드 없이도 기본 기능 확인
echo - 오류 발생 시에도 안정적으로 동작
echo.
echo 🚀 다음 단계:
echo 1. 이 간단한 UI가 정상 동작하는지 확인
echo 2. 영상 파일이나 웹캠 선택 테스트
echo 3. 모델 로드 기능 테스트
echo 4. 정상 동작 확인 후 고급 UI 사용
echo.
pause
