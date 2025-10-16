@echo off
echo ========================================
echo Anomalib 공식 방법으로 AI-VAD 파인튜닝
echo (Folder 데이터 모듈 사용)
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM Anomalib 공식 방법으로 파인튜닝
echo 🚀 Anomalib 공식 방법으로 AI-VAD 파인튜닝 시작...
python train_aivad_official.py

if errorlevel 1 (
    echo ❌ 파인튜닝 실패
    pause
    exit /b 1
)

echo ✅ 파인튜닝 완료
echo 💾 생성된 파일: aivad_official_finetuned.ckpt
pause
