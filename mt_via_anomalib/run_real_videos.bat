@echo off
echo ========================================
echo AI-VAD 실제 비디오 데이터로 학습
echo (video_files_list.py 활용)
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM AI-VAD 실제 비디오로 학습
echo 🚀 AI-VAD 실제 비디오 학습 시작...
echo 💡 실제 비디오 데이터 사용:
echo    1. 실제 비디오 파일들 (56개)
echo    2. AI-VAD 원래 방식 (Density Estimation)
echo    3. 우리 환경의 실제 객체들
echo    4. 높은 Domain 적용성
python train_aivad_real_videos.py

if errorlevel 1 (
    echo ❌ 학습 실패
    pause
    exit /b 1
)

echo ✅ 학습 완료
echo 💾 생성된 파일: aivad_real_videos_learned.ckpt
echo 💡 다음 단계: UI에서 로드하여 이상탐지 성능 테스트
pause
