"""
Windows 환경 최적화 유틸리티
AI-VAD Realtime UI 실행 전 시스템 환경을 최적화합니다.
"""

import os
import sys
import platform
import subprocess
import torch
import cv2
from typing import Dict, List, Tuple


class WindowsOptimizer:
    """윈도우즈 환경 최적화 클래스"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.optimizations_applied = []
        
    def _get_system_info(self) -> Dict:
        """시스템 정보 수집"""
        info = {
            'os': platform.system(),
            'os_version': platform.release(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'opencv_version': cv2.__version__
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return info
    
    def print_system_info(self):
        """시스템 정보 출력"""
        print("=" * 50)
        print("시스템 정보")
        print("=" * 50)
        for key, value in self.system_info.items():
            print(f"{key}: {value}")
        print("=" * 50)
    
    def optimize_environment(self) -> List[str]:
        """환경 최적화 적용"""
        optimizations = []
        
        # 1. CUDA 환경 설정
        if self.system_info['cuda_available']:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 성능 우선
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            optimizations.append("CUDA 환경 설정 완료")
        
        # 2. PyTorch 백엔드 최적화
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        optimizations.append("PyTorch 백엔드 최적화 완료")
        
        # 3. 메모리 최적화
        torch.set_float32_matmul_precision('medium')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        optimizations.append("메모리 최적화 완료")
        
        # 4. OpenCV 최적화
        cv2.setNumThreads(4)  # 스레드 수 제한
        optimizations.append("OpenCV 최적화 완료")
        
        # 5. 윈도우즈 특화 설정
        if platform.system() == "Windows":
            os.environ['OMP_NUM_THREADS'] = '4'
            optimizations.append("윈도우즈 스레드 최적화 완료")
        
        self.optimizations_applied = optimizations
        return optimizations
    
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """의존성 패키지 확인"""
        required_packages = [
            'torch', 'torchvision', 'torchaudio',
            'anomalib', 'opencv-python', 'PySide6',
            'numpy', 'scipy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'opencv-python':
                    import cv2
                elif package == 'PySide6':
                    from PySide6 import QtCore
                elif package == 'torchvision':
                    import torchvision
                elif package == 'torchaudio':
                    import torchaudio
                else:
                    __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        return len(missing_packages) == 0, missing_packages
    
    def check_camera_availability(self) -> List[str]:
        """사용 가능한 카메라 확인"""
        available_cameras = []
        
        for i in range(5):  # 0-4번 카메라 확인
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_cameras.append(f"카메라 {i}")
                cap.release()
        
        return available_cameras
    
    def get_performance_recommendations(self) -> Dict[str, str]:
        """성능 권장사항 제공"""
        recommendations = {}
        
        # GPU 메모리에 따른 권장사항
        if self.system_info['cuda_available']:
            gpu_memory = self.system_info.get('gpu_memory', 0)
            
            if gpu_memory >= 8:
                recommendations['fps'] = "30-60 FPS"
                recommendations['resolution'] = "1280x720"
                recommendations['visualization'] = "모든 옵션 활성화"
            elif gpu_memory >= 4:
                recommendations['fps'] = "15-30 FPS"
                recommendations['resolution'] = "854x480"
                recommendations['visualization'] = "히트맵만 활성화"
            else:
                recommendations['fps'] = "10-15 FPS"
                recommendations['resolution'] = "640x360"
                recommendations['visualization'] = "최소 시각화"
        else:
            recommendations['fps'] = "5-10 FPS"
            recommendations['resolution'] = "640x360"
            recommendations['visualization'] = "최소 시각화"
            recommendations['note'] = "GPU 사용 권장"
        
        return recommendations
    
    def run_diagnostics(self):
        """전체 진단 실행"""
        print("\n🔍 시스템 진단 시작...\n")
        
        # 1. 시스템 정보 출력
        self.print_system_info()
        
        # 2. 의존성 확인
        print("\n📦 의존성 패키지 확인...")
        deps_ok, missing = self.check_dependencies()
        if deps_ok:
            print("✅ 모든 필수 패키지가 설치되어 있습니다.")
        else:
            print("❌ 누락된 패키지:")
            for pkg in missing:
                print(f"   - {pkg}")
            print("\n다음 명령으로 설치하세요:")
            print("pip install " + " ".join(missing))
        
        # 3. 카메라 확인
        print("\n📹 카메라 확인...")
        cameras = self.check_camera_availability()
        if cameras:
            print("✅ 사용 가능한 카메라:")
            for cam in cameras:
                print(f"   - {cam}")
        else:
            print("⚠️  사용 가능한 카메라가 없습니다.")
        
        # 4. 성능 권장사항
        print("\n⚡ 성능 권장사항:")
        recommendations = self.get_performance_recommendations()
        for key, value in recommendations.items():
            print(f"   {key}: {value}")
        
        # 5. 환경 최적화
        print("\n🔧 환경 최적화 적용...")
        optimizations = self.optimize_environment()
        for opt in optimizations:
            print(f"   ✅ {opt}")
        
        print("\n🎯 진단 완료!")
        return deps_ok, cameras, recommendations


def main():
    """메인 실행 함수"""
    print("AI-VAD Windows 환경 최적화 도구")
    print("=" * 40)
    
    optimizer = WindowsOptimizer()
    
    # 진단 실행
    deps_ok, cameras, recommendations = optimizer.run_diagnostics()
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("실행 권장사항")
    print("=" * 50)
    
    if deps_ok:
        print("✅ 모든 준비가 완료되었습니다!")
        print("다음 명령으로 프로그램을 실행하세요:")
        print("python realtime_ui_advanced_windows.py")
    else:
        print("❌ 먼저 누락된 패키지를 설치해주세요.")
        print("설치 후 다시 이 스크립트를 실행하세요.")
    
    print("\n💡 성능 팁:")
    for key, value in recommendations.items():
        print(f"   - {key}: {value}")
    
    if not cameras:
        print("\n⚠️  카메라가 감지되지 않았습니다.")
        print("   영상 파일로 테스트해보세요.")


if __name__ == "__main__":
    main()
