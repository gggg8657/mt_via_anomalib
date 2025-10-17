@echo off
echo ========================================
echo AI-VAD 극단적으로 민감한 설정으로 학습
echo (객체 감지 실패 문제 해결)
echo ========================================

REM conda 환경 활성화
call conda activate mt_p310
if errorlevel 1 (
    echo ❌ conda 환경 활성화 실패
    pause
    exit /b 1
)

echo ✅ conda 환경 활성화 완료

REM AI-VAD 극단적으로 민감한 설정으로 학습
echo 🚀 AI-VAD 극단적으로 민감한 학습 시작...
echo 💡 극단적으로 민감한 설정:
echo    1. box_score_thresh=0.1 (매우 낮음)
echo    2. min_bbox_area=25 (매우 작음)
echo    3. max_bbox_overlap=0.9 (매우 높음)
echo    4. foreground_binary_threshold=5 (매우 민감)
echo    5. 비디오 내용 분석 및 움직임 감지
python train_aivad_ultra_sensitive.py

if errorlevel 1 (
    echo ❌ 학습 실패
    pause
    exit /b 1
)

echo ✅ 학습 완료
echo 💾 생성된 파일: aivad_ultra_sensitive_learned.ckpt
echo 💡 다음 단계: UI에서 로드하여 이상탐지 성능 테스트
pause
