@echo off
echo ========================================
echo 간단한 AI-VAD 파인튜닝 실행
echo (Anomalib Engine 우회)
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM 간단한 파인튜닝 스크립트 실행
echo 🚀 간단한 AI-VAD 파인튜닝 시작...
python train_simple_finetune.py

if errorlevel 1 (
    echo ❌ 파인튜닝 실패
    pause
    exit /b 1
)

echo ✅ 파인튜닝 완료
echo 💾 생성된 파일: aivad_simple_finetuned.ckpt
pause
