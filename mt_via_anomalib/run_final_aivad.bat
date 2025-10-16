@echo off
echo ========================================
echo AI-VAD 올바른 학습 방법 (최종 버전)
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

REM AI-VAD 최종 학습
echo 🚀 AI-VAD 최종 학습 시작...
echo 💡 핵심 원리:
echo    1. Feature Extraction: Flow, Region, Pose, Deep features
echo    2. Density Estimation: 정상 데이터의 분포 학습
echo    3. One-Class Learning: 정상 데이터만으로 분포 모델링
echo    4. No NN Training: 가중치 학습 없음!
python train_aivad_final.py

if errorlevel 1 (
    echo ❌ 학습 실패
    pause
    exit /b 1
)

echo ✅ 학습 완료
echo 💾 생성된 파일: aivad_final_learned.ckpt
echo 💡 다음 단계: UI에서 로드하여 이상탐지 성능 테스트
pause
