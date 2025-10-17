@echo off
echo ========================================
echo AI-VAD 극단적인 설정으로 학습
echo (모든 비디오에서 객체 감지 실패 문제 해결)
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM AI-VAD 극단적인 설정으로 학습
echo 🚀 AI-VAD 극단적인 학습 시작...
echo 💡 극단적인 설정:
echo    1. box_score_thresh=0.05 (극단적으로 낮음)
echo    2. min_bbox_area=10 (극단적으로 작음)
echo    3. max_bbox_overlap=0.95 (극단적으로 높음)
echo    4. foreground_binary_threshold=2 (극단적으로 민감)
echo    5. 상세한 비디오 분석 및 움직임 감지
echo.
python train_aivad_extreme.py

if errorlevel 1 (
    echo ❌ 학습 실패
    pause
    exit /b 1
)

echo ✅ 학습 완료
echo 💾 생성된 파일: aivad_extreme_learned.ckpt
echo 💡 다음 단계: UI에서 로드하여 이상탐지 성능 테스트
pause
