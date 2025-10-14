# 커스텀 비디오 데이터셋

## 사용 방법

1. train_custom_simple.py 파일을 편집하여 video_files 리스트에 비디오 파일 경로를 추가하세요:

```python
video_files = [
    "/path/to/your/video1.mp4",
    "/path/to/your/video2.avi",
    "/path/to/your/video3.mov",
]
```

2. 스크립트를 실행하세요:

```bash
python train_custom_simple.py
```

## 지원되는 비디오 형식
- .mp4
- .avi  
- .mov
- .mkv
- .flv
- .wmv

## 주의사항
- 비디오 파일들이 정상적인 상황(이상이 없는 상황)을 보여주는 것이 좋습니다
- 이상 상황이 포함된 비디오가 있다면 별도로 관리하세요
