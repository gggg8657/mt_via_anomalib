@echo off
echo ========================================
echo 재구성 손실로 AI-VAD 파인튜닝
echo (이상탐지의 올바른 방법)
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM 재구성 학습으로 파인튜닝
echo 🚀 재구성 손실로 AI-VAD 파인튜닝 시작...
echo 💡 이상탐지 원리: 정상 데이터 재구성 능력 학습
python train_aivad_reconstruction.py

if errorlevel 1 (
    echo ❌ 파인튜닝 실패
    pause
    exit /b 1
)

echo ✅ 파인튜닝 완료
echo 💾 생성된 파일: aivad_reconstruction_finetuned.ckpt
echo 💡 다음 단계: UI에서 로드하여 이상탐지 성능 테스트
pause
