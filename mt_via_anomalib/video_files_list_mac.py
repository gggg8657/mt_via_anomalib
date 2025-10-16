"""
비디오 파일 경로 리스트
이 파일을 수정하여 사용할 비디오 파일들을 지정하세요
"""

# 비디오 파일 경로 리스트
video_files = [
    # 사용자 지정 경로 (Windows):
    #"C:\\Users\\User\\Documents\\repos\\VAD\\CV_module_test_tmp\\STEAD\\videos\\normal_video.mp4",
    #"C:\\Users\\User\\Documents\\repos\\VAD\\CV_module_test_tmp\\STEAD\\videos\\unknown_video.mp4",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_0.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_1.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_2.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_3.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_4.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_5.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_6.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_7.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_8.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_9.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_10.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_11.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_12.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_13.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_14.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_15.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_16.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_17.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_18.avi",
    "C:\\Users\\User\PycharmProjects\pythonProject1\Accurate-Interpretable-VAD\data\my_camera\\training_videos\\normal_19.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_20.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_21.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\normal_22.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_0.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_1.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_2.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_3.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_4.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_5.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_6.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_7.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_8.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_9.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_10.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_11.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_12.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_13.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_14.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_15.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_16.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_17.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_18.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_19.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_20.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_21.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_22.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_23.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_24.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_25.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_26.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_27.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_28.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_29.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_30.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_31.avi",
"C:\\Users\\User\\PycharmProjects\\pythonProject1\\Accurate-Interpretable-VAD\\data\\my_camera\\training_videos\\rest_32.avi"


    # 추가 비디오 파일들을 여기에 추가하세요:
    # "C:\\Users\\YourName\\Videos\\normal_video1.mp4",
    # "C:\\Users\\YourName\\Videos\\normal_video2.avi",
    # "D:\\SecurityCameras\\normal_footage.mp4",
    # "E:\\MyVideos\\normal_clip.mov",
]

# 사용 예시:
# video_files = [
#     "C:\\path\\to\\your\\video1.mp4",
#     "C:\\path\\to\\your\\video2.avi",
#     "D:\\another\\path\\video3.mov",
# ]