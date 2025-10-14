# 수정된 커스텀 비디오 데이터셋 학습

## 사용 방법

1. train_custom_fixed.py 파일을 편집하여 video_files 리스트에 비디오 파일 경로를 추가하세요:

```python
video_files = [
    "C:\\Users\\YourName\\Videos\\normal_video1.mp4",
    "D:\\SecurityCameras\\normal_footage.mp4",
    "E:\\MyVideos\\normal_clip.avi",
]
```

2. 스크립트를 실행하세요:

```bash
python train_custom_fixed.py
```

## 수정 사항
- CustomVideoDataModule 대신 기존 Avenue 데이터 모듈 사용
- train_data 속성 오류 해결
- 더 안정적인 학습 환경 제공

## 지원되는 비디오 형식
- .mp4 (권장)
- .avi  
- .mov
- .mkv
- .flv
- .wmv

## 주의사항
- 비디오 파일들이 정상적인 상황을 보여주는 것이 좋습니다
- 이상 상황이 포함된 비디오가 있다면 별도로 관리하세요
