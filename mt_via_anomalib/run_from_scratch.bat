@echo off
echo ========================================
echo AI-VAD 모델 처음부터 학습 실행
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM 처음부터 학습 스크립트 실행
echo 🚀 AI-VAD 처음부터 학습 시작...
python train_aivad_from_scratch.py

if errorlevel 1 (
    echo ❌ 학습 실패
    pause
    exit /b 1
)

echo ✅ 학습 완료
echo 💾 생성된 파일: aivad_from_scratch.ckpt
pause
