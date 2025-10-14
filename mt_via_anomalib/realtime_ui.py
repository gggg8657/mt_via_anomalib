import sys
import os
import time
from typing import Optional

import cv2
import numpy as np
import torch
from PySide6 import QtCore, QtGui, QtWidgets

# 환경 설정 (필요시 사용)
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")


class VideoReaderThread(QtCore.QThread):
    frameReady = QtCore.Signal(np.ndarray)
    finished = QtCore.Signal()

    def __init__(self, video_path: str, fps_limit: Optional[float] = None, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self._stop = False
        self._pause = False
        self.fps_limit = fps_limit
        self._cap: Optional[cv2.VideoCapture] = None

    def run(self) -> None:
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            self.finished.emit()
            return

        last_time = 0.0
        while not self._stop:
            if self._pause:
                time.sleep(0.02)
                continue

            ret, frame = self._cap.read()
            if not ret:
                break

            # FPS 제한
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

    def pause(self, value: bool) -> None:
        self._pause = value


class AiVadInferencer:
    def __init__(self, device: str = "cuda") -> None:
        from anomalib.models.video import AiVad

        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        # 기본 구성: 속도/포즈/딥 특징 모두 사용 (체크포인트가 있다면 거기에 맞춰야 함)
        self.model = AiVad()
        self.model.eval().to(self.device)

        # 내부 torch 모델에 직접 접근하여 2프레임 클립 추론
        # Lightning 모듈 내부의 self.model이 AiVadModel
        self.core = self.model.model
        self.core.eval().to(self.device)

        self.prev_frame_bgr: Optional[np.ndarray] = None

    def load_checkpoint(self, ckpt_path: str) -> None:
        # LightningModule의 load_from_checkpoint 사용
        from anomalib.models.video import AiVad

        loaded = AiVad.load_from_checkpoint(ckpt_path, map_location=self.device)
        loaded.eval().to(self.device)
        self.model = loaded
        self.core = self.model.model
        self.core.eval().to(self.device)

    @staticmethod
    def _bgr_to_chw_float_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
        chw = np.transpose(frame_rgb, (2, 0, 1))  # (C,H,W)
        return torch.from_numpy(chw)

    def infer_on_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        # 이전 프레임이 없으면 버퍼에 저장만 하고 반환
        if self.prev_frame_bgr is None:
            self.prev_frame_bgr = frame_bgr
            return frame_bgr, 0.0

        # 2프레임 클립 구성: (1, 2, 3, H, W)
        t0 = self._bgr_to_chw_float_tensor(self.prev_frame_bgr)
        t1 = self._bgr_to_chw_float_tensor(frame_bgr)
        batch = torch.stack([t0, t1], dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.core(batch)

        # output.pred_score: (N,) -> 단일 배치, anomaly_map: (N,H,W)
        score = float(output.pred_score[0].detach().cpu().item())
        anomaly_map = output.anomaly_map[0].detach().cpu().numpy()  # (H,W)

        # 히트맵 오버레이 생성
        heatmap = self._create_heatmap_overlay(frame_bgr, anomaly_map)

        # 다음 스텝 대비 버퍼 갱신
        self.prev_frame_bgr = frame_bgr
        return heatmap, score

    @staticmethod
    def _create_heatmap_overlay(frame_bgr: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        amap = anomaly_map
        # 정규화 안전장치
        min_v, max_v = float(np.min(amap)), float(np.max(amap))
        if max_v - min_v < 1e-6:
            norm = np.zeros_like(amap, dtype=np.float32)
        else:
            norm = (amap - min_v) / (max_v - min_v)
        norm_resized = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap((norm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap, alpha, 0)
        return overlay


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI-VAD Realtime Anomaly Detection")
        self.resize(1200, 800)

        # 상태
        self.video_path: Optional[str] = None
        self.ckpt_path: Optional[str] = None
        self.threshold: float = 0.5

        # 모델 초기화
        self.inferencer = AiVadInferencer(device="cuda")

        # UI 구성
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # 상단 컨트롤 바
        controls = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("영상 열기")
        self.btn_ckpt = QtWidgets.QPushButton("체크포인트 열기")
        self.btn_play = QtWidgets.QPushButton("재생")
        self.btn_pause = QtWidgets.QPushButton("일시정지")
        self.btn_stop = QtWidgets.QPushButton("정지")
        controls.addWidget(self.btn_open)
        controls.addWidget(self.btn_ckpt)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_stop)

        # 임계치 슬라이더/표시
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.lbl_threshold = QtWidgets.QLabel(f"Threshold: {self.threshold:.2f}")
        controls.addWidget(self.lbl_threshold)
        controls.addWidget(self.threshold_slider)

        layout.addLayout(controls)

        # 비디오 표시 라벨
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111111; color: white;")
        layout.addWidget(self.video_label, stretch=1)

        # 하단 상태 바
        status_layout = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("상태: 대기")
        self.lbl_score = QtWidgets.QLabel("Score: 0.000")
        self.lbl_anomaly = QtWidgets.QLabel("NORMAL")
        self.lbl_anomaly.setStyleSheet("color: #00ff00; font-weight: bold;")
        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.lbl_score)
        status_layout.addWidget(self.lbl_anomaly)
        layout.addLayout(status_layout)

        # 이벤트 연결
        self.btn_open.clicked.connect(self.on_open_video)
        self.btn_ckpt.clicked.connect(self.on_open_ckpt)
        self.btn_play.clicked.connect(self.on_play)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_stop.clicked.connect(self.on_stop)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)

        # 스레드
        self.reader: Optional[VideoReaderThread] = None

    def on_threshold_changed(self, value: int) -> None:
        self.threshold = float(value) / 100.0
        self.lbl_threshold.setText(f"Threshold: {self.threshold:.2f}")

    def on_open_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "영상 파일 선택", os.getcwd(), "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        self.video_path = path
        self.status_message(f"영상 선택: {os.path.basename(path)}")
        # 새 영상 열면 이전 프레임 버퍼 초기화
        self.inferencer.prev_frame_bgr = None

    def on_open_ckpt(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "체크포인트 선택", os.getcwd(), "Checkpoint (*.ckpt *.pt)")
        if not path:
            return
        self.ckpt_path = path
        try:
            self.inferencer.load_checkpoint(path)
            self.status_message(f"체크포인트 로드 완료: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "체크포인트 로드 실패", str(e))

    def on_play(self) -> None:
        if not self.video_path:
            QtWidgets.QMessageBox.information(self, "안내", "먼저 영상을 선택하세요.")
            return
        if self.reader and self.reader.isRunning():
            self.reader.pause(False)
            self.status_message("재생")
            return

        self.reader = VideoReaderThread(self.video_path, fps_limit=30.0)
        self.reader.frameReady.connect(self.on_frame)
        self.reader.finished.connect(self.on_reader_finished)
        self.reader.start()
        self.status_message("재생 시작")

    def on_pause(self) -> None:
        if self.reader and self.reader.isRunning():
            self.reader.pause(True)
            self.status_message("일시정지")

    def on_stop(self) -> None:
        if self.reader:
            self.reader.stop()
            self.reader.wait(1000)
            self.reader = None
        self.status_message("정지")

    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame_bgr: np.ndarray) -> None:
        try:
            vis, score = self.inferencer.infer_on_frame(frame_bgr)
        except Exception as e:
            self.status_message(f"추론 오류: {e}")
            vis = frame_bgr
            score = 0.0

        # 이상 여부 표시
        is_anomaly = score >= self.threshold
        self.lbl_score.setText(f"Score: {score:.3f}")
        self.lbl_anomaly.setText("ANOMALY" if is_anomaly else "NORMAL")
        self.lbl_anomaly.setStyleSheet("color: #ff4444; font-weight: bold;" if is_anomaly else "color: #00ff00; font-weight: bold;")

        # 프레임 표시
        self._display_frame(vis)

    def _display_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        # 라벨 크기에 맞춰 스케일
        pixmap = pixmap.scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        if self.video_label.pixmap():
            self.video_label.setPixmap(self.video_label.pixmap().scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if self.reader:
            self.reader.stop()
            self.reader.wait(1000)
        super().closeEvent(event)

    def on_reader_finished(self) -> None:
        self.status_message("영상 종료")

    def status_message(self, msg: str) -> None:
        self.lbl_status.setText(f"상태: {msg}")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


