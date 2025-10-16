@echo off
echo ========================================
echo 이미지 프레임으로 AI-VAD 파인튜닝
echo (image_segments.json 활용)
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM 이미지 프레임으로 파인튜닝
echo 🚀 이미지 프레임으로 AI-VAD 파인튜닝 시작...
python train_aivad_with_frames.py

if errorlevel 1 (
    echo ❌ 파인튜닝 실패
    pause
    exit /b 1
)

echo ✅ 파인튜닝 완료
echo 💾 생성된 파일: aivad_frame_finetuned.ckpt
pause
