@echo off
chcp 65001 > nul
echo 🏆 간단한 UI 호환 체크포인트 생성 (PyTorch 2.6 호환)
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

REM 간단한 UI 호환 체크포인트 생성
echo.
echo 🎯 간단한 UI 호환 체크포인트를 생성합니다...
python create_simple_checkpoint.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================================
    echo ❌ 체크포인트 생성 중 오류가 발생했습니다.
    echo ==========================================================
    echo.
    echo 💡 해결 방법:
    echo 1. GPU 드라이버 및 CUDA 설치 확인
    echo 2. PyTorch 설치 확인
    echo 3. 관리자 권한으로 실행
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo 🎉 간단한 UI 호환 체크포인트 생성 완료!
echo ==========================================================
echo.
echo 📁 생성된 파일:
echo - aivad_simple_checkpoint.ckpt
echo.
echo 🚀 다음 단계:
echo 1. python realtime_ui_advanced_windows.py 실행
echo 2. "체크포인트 로드" 버튼 클릭
echo 3. aivad_simple_checkpoint.ckpt 파일 선택
echo 4. 실시간 이상 탐지 테스트
echo.
echo 💡 이 체크포인트의 특징:
echo - PyTorch 2.6 완전 호환
echo - realtime_ui_advanced_windows.py와 완벽 호환
echo - 간단한 체크포인트 구조
echo - 복잡한 Lightning 구조 제거
echo - 안정적인 로드/저장
echo - GPU/CPU 모두 지원
echo.
pause
