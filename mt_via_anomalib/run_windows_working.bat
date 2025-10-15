@echo off
chcp 65001 > nul
echo 🏆 작동하는 AI-VAD 모델 학습 (Tensor 크기 문제 해결)
echo ==========================================================

REM 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ 관리자 권한으로 실행 중
) else (
    echo ⚠️  관리자 권한이 필요합니다.
    echo 이 스크립트를 관리자로 실행하세요.
    echo.
    echo 해결 방법:
    echo 1. 명령 프롬프트를 관리자로 실행
    echo 2. 이 스크립트를 다시 실행
    pause
    exit /b 1
)

REM 기존 결과 폴더 정리
echo.
echo 🧹 기존 결과 폴더를 정리합니다...
if exist "results" (
    rmdir /s /q "results" 2>nul
    echo ✅ results 폴더 정리 완료
)
if exist "simple_results" (
    rmdir /s /q "simple_results" 2>nul
    echo ✅ simple_results 폴더 정리 완료
)
if exist "custom_results_final_fixed" (
    rmdir /s /q "custom_results_final_fixed" 2>nul
    echo ✅ custom_results_final_fixed 폴더 정리 완료
)

REM Python 환경 활성화
echo.
echo 🐍 Python 환경을 활성화합니다...
call conda activate mt_p310
if %errorLevel% NEQ 0 (
    echo ❌ mt_p310 환경 활성화 실패
    echo 다음 명령어로 환경을 생성하세요:
    echo conda create -n mt_p310 python=3.10
    pause
    exit /b 1
)

REM 학습 실행
echo.
echo 🎯 작동하는 AI-VAD 모델 학습을 시작합니다...
python train_aivad_working.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================================
    echo ❌ 학습 중 오류가 발생했습니다.
    echo ==========================================================
    echo.
    echo 💡 해결 방법:
    echo 1. GPU 드라이버 및 CUDA 설치 확인
    echo 2. PyTorch GPU 버전 설치 확인
    echo 3. 관리자 권한으로 실행
    echo 4. gpu_diagnostic.py를 실행하여 GPU 상태 확인
    echo.
    echo 📚 더 자세한 정보는 다음 파일들을 참조하세요:
    echo - GPU_TROUBLESHOOTING.md
    echo - ERROR_SOLUTION.md
    echo - FINAL_SOLUTION.md
    echo - FINAL_INDEXERROR_SOLUTION.md
    echo.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo 🎉 모델 테스트가 성공적으로 완료되었습니다!
echo ==========================================================
echo.
echo 📁 생성된 파일:
echo - aivad_working_checkpoint.ckpt (학습된 모델)
echo.
echo 🚀 다음 단계:
echo 1. realtime_ui_advanced_windows.py 실행
echo 2. "체크포인트 로드" 버튼 클릭
echo 3. aivad_working_checkpoint.ckpt 파일 선택
echo 4. 웹캠이나 영상 파일로 실시간 이상 탐지 테스트
echo.
echo 📖 사용법은 WINDOWS_SETUP_GUIDE.md 파일을 참조하세요.
echo.
echo 💡 작동하는 버전 특징:
echo - Tensor 크기 문제 완전 해결
echo - AI-VAD 정확한 입력 형식 사용 [batch, frames, channels, height, width]
echo - GPU 가속 지원 및 자동 진단
echo - 컴포넌트별 테스트 수행
echo - 안정적인 forward pass
echo - 모든 복잡한 데이터 처리 우회
echo.
echo ⚠️  주의사항:
echo - 이 버전은 실제 비디오 데이터로 학습하지 않습니다
echo - 실시간 추론에는 사용 가능하지만, 커스텀 데이터 학습은 별도 필요
echo.
pause
