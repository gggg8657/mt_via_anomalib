@echo off
chcp 65001 > nul
echo 🪟 Windows 권한 문제 해결 스크립트
echo ========================================

echo.
echo 1. 프로젝트 폴더 권한 설정 중...

REM 현재 디렉토리의 권한을 전체 제어로 설정
echo 현재 디렉토리: %CD%
echo 권한을 설정하고 있습니다...

REM 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ 관리자 권한으로 실행 중
) else (
    echo ⚠️  관리자 권한이 필요합니다. 이 스크립트를 관리자로 실행하세요.
    echo.
    echo 해결 방법:
    echo 1. 명령 프롬프트를 관리자로 실행
    echo 2. 이 스크립트를 다시 실행
    pause
    exit /b 1
)

REM 기존 results 폴더 삭제 (권한 문제 방지)
if exist "results" (
    echo 기존 results 폴더를 삭제합니다...
    rmdir /s /q "results" 2>nul
    if exist "results" (
        echo ❌ results 폴더 삭제 실패 - 수동으로 삭제하세요
    ) else (
        echo ✅ results 폴더 삭제 완료
    )
)

REM 기존 custom_results 폴더 삭제
if exist "custom_results" (
    echo 기존 custom_results 폴더를 삭제합니다...
    rmdir /s /q "custom_results" 2>nul
    if exist "custom_results" (
        echo ❌ custom_results 폴더 삭제 실패 - 수동으로 삭제하세요
    ) else (
        echo ✅ custom_results 폴더 삭제 완료
    )
)

REM 폴더 권한 설정
echo.
echo 2. 폴더 권한을 설정합니다...
icacls "%CD%" /grant Everyone:F /T /Q 2>nul
if %errorLevel% == 0 (
    echo ✅ 폴더 권한 설정 완료
) else (
    echo ⚠️  폴더 권한 설정 실패 (정상일 수 있음)
)

echo.
echo 3. Python 환경 확인 중...
python --version >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Python이 설치되어 있습니다
    python --version
) else (
    echo ❌ Python이 설치되어 있지 않습니다
    echo Python을 설치하고 PATH를 설정하세요
    pause
    exit /b 1
)

echo.
echo 4. conda 환경 확인 중...
conda --version >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Conda가 설치되어 있습니다
    echo mt_p310 환경을 활성화합니다...
    call conda activate mt_p310
    if %errorLevel% == 0 (
        echo ✅ mt_p310 환경 활성화 완료
    ) else (
        echo ❌ mt_p310 환경을 찾을 수 없습니다
        echo 다음 명령어로 환경을 생성하세요:
        echo conda create -n mt_p310 python=3.10
        pause
        exit /b 1
    )
) else (
    echo ⚠️  Conda가 설치되어 있지 않습니다
    echo Python 가상환경을 사용하거나 Conda를 설치하세요
)

echo.
echo 5. 필요한 패키지 확인 중...
python -c "import anomalib, torch" 2>nul
if %errorLevel% == 0 (
    echo ✅ 필요한 패키지가 설치되어 있습니다
) else (
    echo ❌ 필요한 패키지가 설치되어 있지 않습니다
    echo 다음 명령어로 설치하세요:
    echo pip install anomalib torch torchvision
    pause
    exit /b 1
)

echo.
echo ========================================
echo 🎉 Windows 권한 문제 해결 완료!
echo ========================================
echo.
echo 이제 다음 명령어로 학습을 시작할 수 있습니다:
echo python train_custom_windows_fix.py
echo.
echo 또는 원클릭 실행:
echo python run_windows_custom.bat
echo.
pause
