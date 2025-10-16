@echo off
echo ========================================
echo 비디오 시퀀스로 AI-VAD 파인튜닝
echo (이미지 프레임들을 비디오 시퀀스로 변환)
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM 비디오 시퀀스로 파인튜닝
echo 🚀 비디오 시퀀스로 AI-VAD 파인튜닝 시작...
python train_aivad_video_sequence.py

if errorlevel 1 (
    echo ❌ 파인튜닝 실패
    pause
    exit /b 1
)

echo ✅ 파인튜닝 완료
echo 💾 생성된 파일: aivad_video_sequence_finetuned.ckpt
pause
