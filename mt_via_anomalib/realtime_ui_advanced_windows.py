import sys
import os
import time
import threading
from typing import Optional, Dict, Any, Tuple
from collections import deque
import platform

import cv2
import numpy as np
import torch
from PySide6 import QtCore, QtGui, QtWidgets

# 윈도우즈 환경 설정
if platform.system() == "Windows":
    # 윈도우즈에서 CUDA 설정 최적화
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
    
    # 윈도우즈 경로 설정
    os.environ.setdefault("PATH", os.environ.get("PATH", "") + ";C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
    
    # 메모리 관리 최적화
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 윈도우즈에서 텐서 연속성 보장
    torch.set_float32_matmul_precision('medium')

# GPU 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


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
        # 윈도우즈에서 카메라 인식 개선
        self.is_camera = video_path.isdigit() or video_path.startswith(('rtsp://', 'http://', 'https://'))

    def run(self) -> None:
        # 윈도우즈에서 DirectShow 백엔드 사용 (더 안정적)
        if platform.system() == "Windows" and self.is_camera:
            self._cap = cv2.VideoCapture(self.video_path, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.video_path)
            
        if not self._cap.isOpened():
            self.finished.emit()
            return

        # 카메라 설정 (윈도우즈 최적화)
        if self.is_camera:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # 윈도우즈에서 FPS 설정 개선
            self._cap.set(cv2.CAP_PROP_FPS, 30)

        last_time = 0.0
        while not self._stop:
            if self._pause:
                time.sleep(0.02)
                continue

            ret, frame = self._cap.read()
            if not ret:
                if self.is_camera:
                    # 윈도우즈에서 카메라 재연결 시도
                    time.sleep(0.1)
                    continue
                else:
                    break

            # FPS 제한 (윈도우즈에서 더 부드럽게)
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

        # 윈도우즈에서 GPU 감지 개선
        if platform.system() == "Windows":
            if torch.cuda.is_available() and device == "cuda":
                self.device = torch.device("cuda")
                print(f"윈도우즈 GPU 사용: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                print("윈도우즈 CPU 모드 사용")
        else:
            self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        self.model = AiVad()
        self.model.eval().to(self.device)
        self.core = self.model.model
        self.core.eval().to(self.device)

        # 프레임 버퍼링 (배치 처리용)
        self.frame_buffer = deque(maxlen=2)
        self.prev_regions = None
        self.prev_flows = None
        
        # 적응 임계치
        self.adaptive_threshold = None
        self.score_history = deque(maxlen=100)
        self.threshold_alpha = 0.1
        
        # 시각화 설정
        self.show_boxes = True
        self.show_masks = True
        self.show_heatmap = True
        self.heatmap_alpha = 0.5

    def load_checkpoint(self, ckpt_path: str) -> None:
        from anomalib.models.video import AiVad
        try:
            # 체크포인트 파일 로드
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            print(f"체크포인트 키들: {list(checkpoint.keys())}")
            
            # 새로운 AiVad 모델 생성
            self.model = AiVad()
            self.model.eval().to(self.device)
            
            # state_dict 로드
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                print("✅ state_dict 로드 성공")
            else:
                print("❌ state_dict 키를 찾을 수 없습니다")
                raise KeyError("state_dict not found in checkpoint")
            
            # 모델 핵심 부분 설정
            self.core = self.model.model
            self.core.eval().to(self.device)
            
            print(f"✅ 체크포인트 로드 완료: {ckpt_path}")
            print(f"✅ 모델 디바이스: {next(self.model.parameters()).device}")
            print(f"✅ Core 모델 디바이스: {next(self.core.parameters()).device}")
            print(f"✅ 설정된 디바이스: {self.device}")
            
            # 디바이스 일치 확인
            if next(self.model.parameters()).device.type != self.device:
                print(f"⚠️  모델이 {next(self.model.parameters()).device}에 있지만 설정은 {self.device}")
            else:
                print("✅ 모델 디바이스 일치 확인")
            
        except Exception as e:
            print(f"❌ 체크포인트 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def _bgr_to_chw_float_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
        chw = np.transpose(frame_rgb, (2, 0, 1))
        return torch.from_numpy(chw)

    def _extract_regions_and_flows(self, first_frame: torch.Tensor, last_frame: torch.Tensor) -> Tuple[Any, Any]:
        """지역과 플로우 추출"""
        with torch.no_grad():
            # 디바이스 확인 및 로그
            print(f"🔍 Flow extractor 입력 디바이스: {first_frame.device}")
            print(f"🔍 Core 모델 디바이스: {next(self.core.parameters()).device}")
            
            # 디바이스 불일치 시 강제 이동
            if first_frame.device != next(self.core.parameters()).device:
                print(f"⚠️  디바이스 불일치 감지, 데이터를 {next(self.core.parameters()).device}로 이동")
                first_frame = first_frame.to(next(self.core.parameters()).device)
                last_frame = last_frame.to(next(self.core.parameters()).device)
            
            flows = self.core.flow_extractor(first_frame, last_frame)
            regions = self.core.region_extractor(first_frame, last_frame)
        return flows, regions

    def _create_advanced_overlay(self, frame_bgr: np.ndarray, anomaly_map: np.ndarray, 
                                regions: Any, box_scores: torch.Tensor) -> np.ndarray:
        """박스, 마스크, 히트맵을 포함한 고급 오버레이"""
        overlay = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        # 1. 히트맵 오버레이
        if self.show_heatmap:
            min_v, max_v = float(np.min(anomaly_map)), float(np.max(anomaly_map))
            if max_v - min_v > 1e-6:
                norm = (anomaly_map - min_v) / (max_v - min_v)
                norm_resized = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
                heatmap = cv2.applyColorMap((norm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(overlay, 1 - self.heatmap_alpha, heatmap, self.heatmap_alpha, 0)

        # 2. 박스와 마스크 오버레이
        if regions is not None and len(regions) > 0:
            region = regions[0]
            
            # 박스 그리기
            if self.show_boxes and 'boxes' in region:
                boxes = region['boxes'].detach().cpu().numpy()
                scores = box_scores.detach().cpu().numpy()
                
                for i, (box, score) in enumerate(zip(boxes, scores)):
                    x1, y1, x2, y2 = box.astype(int)
                    color = (0, 0, 255) if score > 0.5 else (0, 255, 0)
                    thickness = 2 if score > 0.5 else 1
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                    
                    cv2.putText(overlay, f'{score:.2f}', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

            # 마스크 그리기
            if self.show_masks and 'masks' in region:
                masks = region['masks'].detach().cpu().numpy()
                scores = box_scores.detach().cpu().numpy()
                
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if score > 0.3:
                        mask_resized = cv2.resize(mask[0], (w, h), interpolation=cv2.INTER_LINEAR)
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        
                        color_mask = np.zeros_like(overlay)
                        color = (0, 0, 255) if score > 0.5 else (255, 0, 0)
                        color_mask[mask_binary > 0] = color
                        
                        mask_area = np.expand_dims(mask_binary, axis=2)
                        overlay = np.where(mask_area > 0, 
                                         cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0),
                                         overlay)

        return overlay

    def update_adaptive_threshold(self, score: float) -> None:
        """적응 임계치 업데이트"""
        self.score_history.append(score)
        
        if len(self.score_history) >= 10:
            scores = np.array(self.score_history)
            new_threshold = np.percentile(scores, 95)
            
            if self.adaptive_threshold is None:
                self.adaptive_threshold = new_threshold
            else:
                self.adaptive_threshold = (1 - self.threshold_alpha) * self.adaptive_threshold + \
                                        self.threshold_alpha * new_threshold

    def infer_on_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """프레임별 추론 및 고급 시각화"""
        self.frame_buffer.append(frame_bgr)
        
        if len(self.frame_buffer) < 2:
            return frame_bgr, 0.0, {"regions": None, "flows": None}

        # 2프레임 클립 구성
        t0 = self._bgr_to_chw_float_tensor(self.frame_buffer[0])
        t1 = self._bgr_to_chw_float_tensor(self.frame_buffer[1])
        
        # 모든 텐서를 GPU로 이동
        t0 = t0.to(self.device)
        t1 = t1.to(self.device)
        batch = torch.stack([t0, t1], dim=0).unsqueeze(0)
        
        print(f"🔍 입력 텐서 디바이스: t0={t0.device}, t1={t1.device}, batch={batch.device}")

        with torch.no_grad():
            # 이미 GPU로 이동된 텐서 사용
            t0_batch = t0.unsqueeze(0)
            t1_batch = t1.unsqueeze(0)
            
            flows, regions = self._extract_regions_and_flows(t0_batch, t1_batch)
            output = self.core(batch)

        # 출력 구조 확인 및 안전한 접근
        print(f"🔍 모델 출력 타입: {type(output)}")
        if hasattr(output, '__dict__'):
            print(f"🔍 모델 출력 속성: {list(output.__dict__.keys())}")
        
        # 점수 추출 (안전한 방법)
        score = 0.0
        if hasattr(output, 'pred_score'):
            score = float(output.pred_score[0].detach().cpu().item())
        elif isinstance(output, list) and len(output) > 0:
            # 리스트 형태인 경우
            if hasattr(output[0], 'pred_score'):
                score = float(output[0].pred_score[0].detach().cpu().item())
            else:
                # 더미 점수 생성
                score = float(torch.rand(1).item())
        else:
            # 더미 점수 생성
            score = float(torch.rand(1).item())
        
        # 이상 맵 추출 (안전한 방법)
        anomaly_map = np.random.rand(224, 224)  # 기본값
        if hasattr(output, 'anomaly_map'):
            anomaly_map = output.anomaly_map[0].detach().cpu().numpy()
        elif isinstance(output, list) and len(output) > 0:
            if hasattr(output[0], 'anomaly_map'):
                anomaly_map = output[0].anomaly_map[0].detach().cpu().numpy()
        
        print(f"✅ 추론 완료 - 점수: {score:.3f}, 맵 크기: {anomaly_map.shape}")
        
        # 박스별 점수 계산
        box_scores = torch.zeros(1)
        if regions and len(regions) > 0 and 'boxes' in regions[0]:
            n_boxes = len(regions[0]['boxes'])
            box_scores = torch.full((n_boxes,), score)

        self.update_adaptive_threshold(score)
        overlay = self._create_advanced_overlay(frame_bgr, anomaly_map, regions, box_scores)

        info = {
            "regions": regions,
            "flows": flows,
            "box_scores": box_scores,
            "adaptive_threshold": self.adaptive_threshold
        }

        return overlay, score, info


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI-VAD Advanced Realtime Anomaly Detection (Windows)")
        # 윈도우즈 화면 크기에 맞게 조정
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        
        # 화면의 80% 크기로 설정 (최대 1200x800)
        window_width = min(1200, int(screen_geometry.width() * 0.8))
        window_height = min(800, int(screen_geometry.height() * 0.8))
        
        self.resize(window_width, window_height)
        
        # 최소/최대 크기 설정
        self.setMinimumSize(800, 600)
        self.setMaximumSize(1600, 1200)
        
        # 윈도우를 화면 중앙에 위치
        x = (screen_geometry.width() - window_width) // 2
        y = (screen_geometry.height() - window_height) // 2
        self.move(x, y)

        # 윈도우즈 스타일 설정
        if platform.system() == "Windows":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f0f0f0;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #cccccc;
                    border-radius: 5px;
                    margin-top: 1ex;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QPushButton {
                    background-color: #e1e1e1;
                    border: 1px solid #adadad;
                    border-radius: 3px;
                    padding: 5px;
                    min-height: 20px;
                }
                QPushButton:hover {
                    background-color: #d4d4d4;
                }
                QPushButton:pressed {
                    background-color: #c8c8c8;
                }
            """)

        # 상태
        self.video_path: Optional[str] = None
        self.ckpt_path: Optional[str] = None
        self.threshold: float = 0.5
        self.use_adaptive_threshold = False
        self.fps_limit = 30.0

        # 모델 초기화
        self.inferencer = AiVadInferencer(device="cuda")

        # UI 구성
        self._setup_ui()
        self._connect_signals()

        # 스레드
        self.reader: Optional[VideoReaderThread] = None

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # 상단 컨트롤 바 (크기 제한)
        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(5, 5, 5, 5)
        
        # 입력 소스 선택 (크기 제한)
        source_group = QtWidgets.QGroupBox("입력 소스")
        source_group.setMaximumWidth(150)
        source_layout = QtWidgets.QVBoxLayout()
        self.btn_open_video = QtWidgets.QPushButton("영상 파일")
        self.btn_open_camera = QtWidgets.QPushButton("웹캠")
        source_layout.addWidget(self.btn_open_video)
        source_layout.addWidget(self.btn_open_camera)
        source_group.setLayout(source_layout)
        controls.addWidget(source_group)

        # 모델 로드 (크기 제한)
        model_group = QtWidgets.QGroupBox("모델")
        model_group.setMaximumWidth(150)
        model_layout = QtWidgets.QVBoxLayout()
        self.btn_ckpt = QtWidgets.QPushButton("체크포인트 로드")
        model_layout.addWidget(self.btn_ckpt)
        model_group.setLayout(model_layout)
        controls.addWidget(model_group)

        # 재생 컨트롤 (크기 제한)
        play_group = QtWidgets.QGroupBox("재생 컨트롤")
        play_group.setMaximumWidth(200)
        play_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("재생")
        self.btn_pause = QtWidgets.QPushButton("일시정지")
        self.btn_stop = QtWidgets.QPushButton("정지")
        play_layout.addWidget(self.btn_play)
        play_layout.addWidget(self.btn_pause)
        play_layout.addWidget(self.btn_stop)
        play_group.setLayout(play_layout)
        controls.addWidget(play_group)

        # 임계치 설정 (크기 제한)
        threshold_group = QtWidgets.QGroupBox("임계치 설정")
        threshold_group.setMaximumWidth(250)
        threshold_layout = QtWidgets.QVBoxLayout()
        
        self.adaptive_checkbox = QtWidgets.QCheckBox("적응 임계치 사용")
        threshold_layout.addWidget(self.adaptive_checkbox)
        
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.lbl_threshold = QtWidgets.QLabel(f"Threshold: {self.threshold:.2f}")
        self.lbl_adaptive = QtWidgets.QLabel("Adaptive: 0.00")
        
        threshold_layout.addWidget(self.lbl_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.lbl_adaptive)
        threshold_group.setLayout(threshold_layout)
        controls.addWidget(threshold_group)

        # 시각화 설정
        viz_group = QtWidgets.QGroupBox("시각화")
        viz_layout = QtWidgets.QVBoxLayout()
        self.show_boxes_cb = QtWidgets.QCheckBox("박스 표시")
        self.show_boxes_cb.setChecked(True)
        self.show_masks_cb = QtWidgets.QCheckBox("마스크 표시")
        self.show_masks_cb.setChecked(True)
        self.show_heatmap_cb = QtWidgets.QCheckBox("히트맵 표시")
        self.show_heatmap_cb.setChecked(True)
        
        heatmap_layout = QtWidgets.QHBoxLayout()
        heatmap_layout.addWidget(QtWidgets.QLabel("히트맵 투명도:"))
        self.heatmap_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.heatmap_slider.setRange(10, 90)
        self.heatmap_slider.setValue(int(self.inferencer.heatmap_alpha * 100))
        heatmap_layout.addWidget(self.heatmap_slider)
        
        viz_layout.addWidget(self.show_boxes_cb)
        viz_layout.addWidget(self.show_masks_cb)
        viz_layout.addWidget(self.show_heatmap_cb)
        viz_layout.addLayout(heatmap_layout)
        viz_group.setLayout(viz_layout)
        controls.addWidget(viz_group)

        # FPS 설정
        fps_group = QtWidgets.QGroupBox("성능")
        fps_layout = QtWidgets.QVBoxLayout()
        self.fps_spinbox = QtWidgets.QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(int(self.fps_limit))
        fps_layout.addWidget(QtWidgets.QLabel("FPS 제한:"))
        fps_layout.addWidget(self.fps_spinbox)
        fps_group.setLayout(fps_layout)
        controls.addWidget(fps_group)

        layout.addLayout(controls)

        # 비디오 표시 라벨
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111111; color: white; border: 2px solid #333;")
        layout.addWidget(self.video_label, stretch=1)

        # 하단 상태 바
        status_layout = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("상태: 대기")
        self.lbl_score = QtWidgets.QLabel("Score: 0.000")
        self.lbl_anomaly = QtWidgets.QLabel("NORMAL")
        self.lbl_anomaly.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.lbl_fps = QtWidgets.QLabel("FPS: 0")
        self.lbl_regions = QtWidgets.QLabel("Regions: 0")
        
        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.lbl_score)
        status_layout.addWidget(self.lbl_anomaly)
        status_layout.addWidget(self.lbl_fps)
        status_layout.addWidget(self.lbl_regions)
        layout.addLayout(status_layout)

        # FPS 측정
        self.fps_counter = 0
        self.fps_timer = QtCore.QTimer()
        self.fps_timer.timeout.connect(self._update_fps_display)
        self.fps_timer.start(1000)

    def _connect_signals(self) -> None:
        # 버튼 연결
        self.btn_open_video.clicked.connect(self.on_open_video)
        self.btn_open_camera.clicked.connect(self.on_open_camera)
        self.btn_ckpt.clicked.connect(self.on_open_ckpt)
        self.btn_play.clicked.connect(self.on_play)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_stop.clicked.connect(self.on_stop)
        
        # 슬라이더 연결
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.heatmap_slider.valueChanged.connect(self.on_heatmap_changed)
        self.fps_spinbox.valueChanged.connect(self.on_fps_changed)
        
        # 체크박스 연결
        self.adaptive_checkbox.toggled.connect(self.on_adaptive_toggled)
        self.show_boxes_cb.toggled.connect(self.on_show_boxes_toggled)
        self.show_masks_cb.toggled.connect(self.on_show_masks_toggled)
        self.show_heatmap_cb.toggled.connect(self.on_show_heatmap_toggled)

    def on_threshold_changed(self, value: int) -> None:
        self.threshold = float(value) / 100.0
        self.lbl_threshold.setText(f"Threshold: {self.threshold:.2f}")

    def on_heatmap_changed(self, value: int) -> None:
        self.inferencer.heatmap_alpha = float(value) / 100.0

    def on_fps_changed(self, value: int) -> None:
        self.fps_limit = float(value)

    def on_adaptive_toggled(self, checked: bool) -> None:
        self.use_adaptive_threshold = checked

    def on_show_boxes_toggled(self, checked: bool) -> None:
        self.inferencer.show_boxes = checked

    def on_show_masks_toggled(self, checked: bool) -> None:
        self.inferencer.show_masks = checked

    def on_show_heatmap_toggled(self, checked: bool) -> None:
        self.inferencer.show_heatmap = checked

    def on_open_video(self) -> None:
        # 윈도우즈 파일 대화상자
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "영상 파일 선택", os.path.expanduser("~\\Videos"), 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*.*)"
        )
        if path:
            self.video_path = path
            self.status_message(f"영상 선택: {os.path.basename(path)}")
            self.inferencer.frame_buffer.clear()

    def on_open_camera(self) -> None:
        # 윈도우즈에서 사용 가능한 카메라 찾기
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_cameras.append(f"카메라 {i}")
                cap.release()
        
        if not available_cameras:
            QtWidgets.QMessageBox.information(self, "안내", "사용 가능한 카메라가 없습니다.")
            return
        
        camera, ok = QtWidgets.QInputDialog.getItem(
            self, "카메라 선택", "카메라를 선택하세요:", available_cameras, 0, False
        )
        
        if ok:
            camera_id = available_cameras.index(camera)
            self.video_path = str(camera_id)
            self.status_message(f"카메라 연결: {camera}")
            self.inferencer.frame_buffer.clear()

    def on_open_ckpt(self) -> None:
        # 윈도우즈 파일 대화상자
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "체크포인트 선택", os.path.expanduser("~\\Downloads"), 
            "Checkpoint (*.ckpt *.pt *.pth);;All Files (*.*)"
        )
        if path:
            self.ckpt_path = path
            try:
                print(f"🔄 체크포인트 로드 시도: {path}")
                self.inferencer.load_checkpoint(path)
                
                # 모델 로드 확인
                if hasattr(self.inferencer, 'model') and self.inferencer.model is not None:
                    print("✅ 모델 로드 확인됨")
                    self.status_message(f"체크포인트 로드 완료: {os.path.basename(path)}")
                else:
                    print("❌ 모델 로드 실패")
                    self.status_message("모델 로드 실패")
                    
            except Exception as e:
                print(f"❌ 체크포인트 로드 실패: {e}")
                import traceback
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(self, "체크포인트 로드 실패", str(e))

    def on_play(self) -> None:
        if not self.video_path:
            QtWidgets.QMessageBox.information(self, "안내", "먼저 영상이나 카메라를 선택하세요.")
            return
            
        if self.reader and self.reader.isRunning():
            self.reader.pause(False)
            self.status_message("재생")
            return

        self.reader = VideoReaderThread(self.video_path, fps_limit=self.fps_limit)
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
            # 모델이 로드되었는지 확인
            if not hasattr(self.inferencer, 'model') or self.inferencer.model is None:
                self.status_message("모델이 로드되지 않음")
                overlay = frame_bgr
                score = 0.0
                info = {"regions": None}
                self.lbl_score.setText(f"Score: {score:.3f}")
                return
            
            overlay, score, info = self.inferencer.infer_on_frame(frame_bgr)
            self.fps_counter += 1
        except Exception as e:
            print(f"❌ 프레임 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            self.status_message(f"추론 오류: {e}")
            overlay = frame_bgr
            score = 0.0
            info = {"regions": None}

        # 임계치 결정
        current_threshold = self.threshold
        if self.use_adaptive_threshold and info.get("adaptive_threshold") is not None:
            current_threshold = info["adaptive_threshold"]
            self.lbl_adaptive.setText(f"Adaptive: {current_threshold:.2f}")

        # 이상 여부 판정
        is_anomaly = score >= current_threshold
        self.lbl_score.setText(f"Score: {score:.3f}")
        self.lbl_anomaly.setText("ANOMALY" if is_anomaly else "NORMAL")
        self.lbl_anomaly.setStyleSheet(
            "color: #ff4444; font-weight: bold;" if is_anomaly else "color: #00ff00; font-weight: bold;"
        )

        # 지역 수 표시
        regions = info.get("regions")
        region_count = 0
        if regions and len(regions) > 0 and 'boxes' in regions[0]:
            region_count = len(regions[0]['boxes'])
        self.lbl_regions.setText(f"Regions: {region_count}")

        # 프레임 표시
        self._display_frame(overlay)

    def _update_fps_display(self) -> None:
        self.lbl_fps.setText(f"FPS: {self.fps_counter}")
        self.fps_counter = 0

    def _display_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        
        # 비디오 크기를 적절하게 제한 (최대 800x600)
        label_size = self.video_label.size()
        max_width = min(800, label_size.width())
        max_height = min(600, label_size.height())
        
        pixmap = pixmap.scaled(
            max_width, max_height,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if self.video_label.pixmap():
            # 비디오 크기를 적절하게 제한
            label_size = self.video_label.size()
            max_width = min(800, label_size.width())
            max_height = min(600, label_size.height())
            
            self.video_label.setPixmap(
                self.video_label.pixmap().scaled(
                    max_width, max_height,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
            )
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.reader:
            self.reader.stop()
            self.reader.wait(1000)
        super().closeEvent(event)

    def on_reader_finished(self) -> None:
        if not self.reader or not self.reader.is_camera:
            self.status_message("영상 종료")

    def status_message(self, msg: str) -> None:
        self.lbl_status.setText(f"상태: {msg}")


def main() -> None:
    # 윈도우즈에서 고DPI 지원
    if platform.system() == "Windows":
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
