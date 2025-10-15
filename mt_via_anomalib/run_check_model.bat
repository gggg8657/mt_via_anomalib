@echo off
chcp 65001 > nul
echo 🔍 AI-VAD 모델 구조 확인
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

REM 모델 구조 확인 실행
echo.
echo 🔍 AI-VAD 모델 구조를 확인합니다...
python check_model_structure.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================================
    echo ❌ 모델 구조 확인 중 오류가 발생했습니다.
    echo ==========================================================
    echo.
    echo 💡 해결 방법:
    echo 1. Anomalib 설치 확인: pip install anomalib
    echo 2. PyTorch 설치 확인: pip install torch
    echo 3. 관리자 권한으로 실행
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo 🎉 모델 구조 확인 완료!
echo ==========================================================
echo.
echo 💡 확인된 내용:
echo - Anomalib 버전 정보
echo - AI-VAD 모델 구조
echo - CLIP extractor 포함 여부
echo - Flow/Region extractor 확인
echo - 체크포인트 키 분석
echo.
echo 🚀 다음 단계:
echo 1. 생성된 aivad_proper_checkpoint.ckpt 확인
echo 2. UI에서 새 체크포인트 로드 테스트
echo 3. 실제 AI-VAD 구조로 추론 확인
echo.
pause
