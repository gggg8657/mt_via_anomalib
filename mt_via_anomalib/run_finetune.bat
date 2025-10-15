@echo off
echo ========================================
echo AI-VAD 모델 파인튜닝 실행
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM 파인튜닝 스크립트 실행
echo 🚀 AI-VAD 파인튜닝 시작...
python train_aivad_finetune.py

if errorlevel 1 (
    echo ❌ 파인튜닝 실패
    pause
    exit /b 1
)

echo ✅ 파인튜닝 완료
echo 💾 생성된 파일: aivad_finetuned.ckpt
pause
