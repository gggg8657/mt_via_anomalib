"""
간단한 AI-VAD UI 테스트
복잡한 추론 과정 없이 기본적인 UI 동작 확인
"""

import sys
import os
import time
import threading
from typing import Optional
from collections import deque
import platform

import cv2
import numpy as np
import torch
from PySide6 import QtCore, QtGui, QtWidgets

# 윈도우즈 환경 설정
if platform.system() == "Windows":
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    torch.set_float32_matmul_precision('medium')

# GPU 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


class SimpleVideoReader(QtCore.QThread):
    frameReady = QtCore.Signal(np.ndarray)
    finished = QtCore.Signal()

    def __init__(self, video_path: str, fps_limit: Optional[float] = None, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self._stop = False
        self._pause = False
        self.fps_limit = fps_limit
        self._cap: Optional[cv2.VideoCapture] = None
        self.is_camera = video_path.isdigit() or video_path.startswith(('rtsp://', 'http://', 'https://'))

    def run(self) -> None:
        if platform.system() == "Windows" and self.is_camera:
            self._cap = cv2.VideoCapture(self.video_path, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.video_path)
            
        if not self._cap.isOpened():
            self.finished.emit()
            return

        if self.is_camera:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._cap.set(cv2.CAP_PROP_FPS, 30)

        last_time = 0.0
        while not self._stop:
            if self._pause:
                time.sleep(0.02)
                continue

            ret, frame = self._cap.read()
            if not ret:
                if self.is_camera:
                    time.sleep(0.1)
                    continue
                else:
                    break

            if self.fps_limit and self.fps_limit > 0:
                now = time.time()
                min_interval = 1.0 / self.fps_limit
                elapsed = now - last_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_time = time.time()

            self.frameReady.emit(frame)

        if self._cap is not None:
            self._cap.release()
        self.finished.emit()

    def stop(self) -> None:
        self._stop = True

    def pause(self, pause: bool) -> None:
        self._pause = pause


class SimpleMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("간단한 AI-VAD 테스트")
        
        # 화면 크기 설정
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        window_width = min(800, int(screen_geometry.width() * 0.6))
        window_height = min(600, int(screen_geometry.height() * 0.6))
        self.resize(window_width, window_height)
        
        # 화면 중앙에 위치
        x = (screen_geometry.width() - window_width) // 2
        y = (screen_geometry.height() - window_height) // 2
        self.move(x, y)
        
        self.video_path = None
        self.reader: Optional[SimpleVideoReader] = None
        self.model_loaded = False
        
        self._setup_ui()
        
    def _setup_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # 상단 컨트롤
        controls = QtWidgets.QHBoxLayout()
        
        self.btn_open_video = QtWidgets.QPushButton("영상 파일")
        self.btn_open_camera = QtWidgets.QPushButton("웹캠")
        self.btn_load_model = QtWidgets.QPushButton("모델 로드")
        self.btn_play = QtWidgets.QPushButton("재생")
        self.btn_pause = QtWidgets.QPushButton("일시정지")
        self.btn_stop = QtWidgets.QPushButton("정지")
        
        controls.addWidget(self.btn_open_video)
        controls.addWidget(self.btn_open_camera)
        controls.addWidget(self.btn_load_model)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_stop)
        
        layout.addLayout(controls)
        
        # 비디오 표시
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #000000;")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("비디오를 선택하세요")
        layout.addWidget(self.video_label)
        
        # 상태 표시
        status_layout = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("상태: 대기")
        self.lbl_score = QtWidgets.QLabel("Score: 0.000")
        self.lbl_fps = QtWidgets.QLabel("FPS: 0")
        
        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.lbl_score)
        status_layout.addWidget(self.lbl_fps)
        
        layout.addLayout(status_layout)
        
        # 이벤트 연결
        self.btn_open_video.clicked.connect(self.on_open_video)
        self.btn_open_camera.clicked.connect(self.on_open_camera)
        self.btn_load_model.clicked.connect(self.on_load_model)
        self.btn_play.clicked.connect(self.on_play)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_stop.clicked.connect(self.on_stop)
        
        # FPS 타이머
        self.fps_timer = QtCore.QTimer()
        self.fps_timer.timeout.connect(self._update_fps)
        self.fps_counter = 0
        self.fps_timer.start(1000)
        
    def on_open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "영상 파일 선택", os.path.expanduser("~\\Downloads"), 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if path:
            self.video_path = path
            self.status_message(f"영상 선택: {os.path.basename(path)}")
            
    def on_open_camera(self):
        self.video_path = "0"
        self.status_message("카메라 연결: 0")
        
    def on_load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "체크포인트 선택", os.path.expanduser("~\\Downloads"), 
            "Checkpoint (*.ckpt *.pt *.pth);;All Files (*.*)"
        )
        if path:
            try:
                print(f"🔄 체크포인트 로드 시도: {path}")
                
                # 체크포인트 로드
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                print(f"체크포인트 키들: {list(checkpoint.keys())}")
                
                # 모델 로드 시뮬레이션 (실제로는 로드하지 않음)
                self.model_loaded = True
                self.status_message(f"모델 로드 완료: {os.path.basename(path)}")
                
            except Exception as e:
                print(f"❌ 체크포인트 로드 실패: {e}")
                QtWidgets.QMessageBox.critical(self, "모델 로드 실패", str(e))
                
    def on_play(self):
        if not self.video_path:
            QtWidgets.QMessageBox.information(self, "안내", "먼저 영상이나 카메라를 선택하세요.")
            return
            
        if self.reader and self.reader.isRunning():
            self.reader.pause(False)
            self.status_message("재생")
            return
            
        self.reader = SimpleVideoReader(self.video_path, fps_limit=30)
        self.reader.frameReady.connect(self.on_frame)
        self.reader.finished.connect(self.on_finished)
        self.reader.start()
        
        self.status_message("재생 시작")
        
    def on_pause(self):
        if self.reader:
            self.reader.pause(True)
            self.status_message("일시정지")
            
    def on_stop(self):
        if self.reader:
            self.reader.stop()
            self.reader.wait(1000)
            self.reader = None
        self.status_message("정지")
        
    def on_finished(self):
        self.status_message("영상 종료")
        
    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame_bgr: np.ndarray) -> None:
        try:
            # 간단한 더미 추론
            if self.model_loaded:
                # 랜덤 점수 생성 (0.0 ~ 1.0)
                score = np.random.random()
                is_anomaly = score > 0.7
                
                # 더미 오버레이 생성
                overlay = frame_bgr.copy()
                if is_anomaly:
                    # 빨간색 테두리
                    cv2.rectangle(overlay, (10, 10), (overlay.shape[1]-10, overlay.shape[0]-10), (0, 0, 255), 3)
                    cv2.putText(overlay, "ANOMALY DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # 초록색 테두리
                    cv2.rectangle(overlay, (10, 10), (overlay.shape[1]-10, overlay.shape[0]-10), (0, 255, 0), 2)
                
                self.lbl_score.setText(f"Score: {score:.3f}")
                self.lbl_score.setStyleSheet(
                    "color: #ff4444; font-weight: bold;" if is_anomaly else "color: #00ff00; font-weight: bold;"
                )
            else:
                overlay = frame_bgr
                self.lbl_score.setText("Score: N/A")
                
            self._display_frame(overlay)
            self.fps_counter += 1
            
        except Exception as e:
            print(f"❌ 프레임 처리 오류: {e}")
            self._display_frame(frame_bgr)
            
    def _display_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        
        # 비디오 크기 제한
        max_width = min(640, self.video_label.width())
        max_height = min(480, self.video_label.height())
        
        pixmap = pixmap.scaled(
            max_width, max_height,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)
        
    def _update_fps(self):
        self.lbl_fps.setText(f"FPS: {self.fps_counter}")
        self.fps_counter = 0
        
    def status_message(self, msg: str) -> None:
        self.lbl_status.setText(f"상태: {msg}")
        
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.reader:
            self.reader.stop()
            self.reader.wait(1000)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # 고해상도 지원
    if platform.system() == "Windows":
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    window = SimpleMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
