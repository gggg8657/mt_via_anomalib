@echo off
echo ========================================
echo AI-VAD 디렉토리 기반 학습
echo (효율적인 비디오 파일 로딩)
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM AI-VAD 디렉토리 기반 학습
echo 🚀 AI-VAD 디렉토리 기반 학습 시작...
echo 💡 효율적인 디렉토리 기반 처리:
echo    1. 디렉토리에서 자동으로 비디오 파일 스캔
echo    2. 파일 복사 없이 직접 처리
echo    3. 극단적으로 민감한 객체 감지
echo    4. 비디오 내용 분석 및 움직임 감지
echo.
echo 📁 비디오 디렉토리 경로를 입력하세요.
echo 예시: C:\Users\User\Videos
echo.
python train_aivad_directory.py

if errorlevel 1 (
    echo ❌ 학습 실패
    pause
    exit /b 1
)

echo ✅ 학습 완료
echo 💾 생성된 파일: aivad_directory_learned.ckpt
echo 💡 다음 단계: UI에서 로드하여 이상탐지 성능 테스트
pause
