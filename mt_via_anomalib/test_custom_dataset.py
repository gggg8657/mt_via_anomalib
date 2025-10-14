"""
커스텀 데이터셋 테스트 스크립트
"""

import os
import pathlib
from train_custom_simple import prepare_custom_dataset


def test_custom_dataset():
    """커스텀 데이터셋 테스트"""
    
    print("🧪 커스텀 데이터셋 테스트 시작...")
    
    # 테스트용 가상 비디오 파일들 (실제로는 존재하지 않음)
    test_video_files = [
        "/path/to/test/video1.mp4",
        "/path/to/test/video2.avi",
        "/path/to/test/video3.mov",
    ]
    
    dataset_path = "./test_custom_dataset"
    
    try:
        # 데이터셋 준비 테스트
        prepare_custom_dataset(dataset_path, test_video_files)
        
        # 결과 확인
        dataset_path = pathlib.Path(dataset_path)
        train_path = dataset_path / "train"
        
        if train_path.exists():
            files = list(train_path.glob("*"))
            print(f"✅ 테스트 완료: {len(files)}개 파일 처리됨")
            print(f"📁 생성된 경로: {train_path}")
        else:
            print("❌ 테스트 실패: train 디렉토리가 생성되지 않음")
            
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")


def show_usage_examples():
    """사용 예시 표시"""
    
    print("\n" + "="*60)
    print("📋 커스텀 데이터셋 사용 예시")
    print("="*60)
    
    print("\n1️⃣  기본 사용법:")
    print("""
# train_custom_simple.py 파일에서 수정할 부분:

video_files = [
    "/home/user/videos/security_normal_001.mp4",
    "/home/user/videos/security_normal_002.mp4",
    "/home/user/videos/security_normal_003.mp4",
]

dataset_path = "./my_custom_dataset"
max_epochs = 10
batch_size = 8
""")
    
    print("\n2️⃣  Windows 경로 예시:")
    print("""
video_files = [
    "C:\\Users\\YourName\\Videos\\normal_video1.mp4",
    "C:\\Users\\YourName\\Videos\\normal_video2.avi",
    "D:\\SecurityCameras\\normal_footage.mp4",
]
""")
    
    print("\n3️⃣  상대 경로 예시:")
    print("""
video_files = [
    "./videos/normal_001.mp4",
    "./videos/normal_002.mp4",
    "../data/normal_videos/video1.mov",
]
""")
    
    print("\n4️⃣  대량 파일 처리 예시:")
    print("""
import glob

# 특정 디렉토리의 모든 비디오 파일 자동 추가
video_files = glob.glob("/path/to/videos/*.mp4") + glob.glob("/path/to/videos/*.avi")

# 또는
video_files = []
video_dirs = ["/path/to/videos1", "/path/to/videos2", "/path/to/videos3"]
for video_dir in video_dirs:
    video_files.extend(glob.glob(os.path.join(video_dir, "*.mp4")))
    video_files.extend(glob.glob(os.path.join(video_dir, "*.avi")))
""")
    
    print("\n5️⃣  실행 명령어:")
    print("""
# Linux/Mac
conda activate mt_p310
python train_custom_simple.py

# Windows
conda activate mt_p310
python train_custom_simple.py
""")


def main():
    print("🎬 커스텀 비디오 데이터셋 도구")
    print("="*40)
    
    # 사용 예시 표시
    show_usage_examples()
    
    # 테스트 실행
    test_custom_dataset()
    
    print("\n✅ 테스트 완료!")
    print("\n📚 더 자세한 정보는 CUSTOM_DATASET_GUIDE.md 파일을 참조하세요.")


if __name__ == "__main__":
    main()
