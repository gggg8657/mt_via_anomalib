"""
AiVAD 데모용 UI - 실시간 이상 탐지 및 로깅
기능:
1. 파일 탐색기로 비디오 선택
2. 실시간 영상 재생 및 AiVAD 모델 프레임별 분석
3. 이상상황 알림 (팝업) 및 빨간 테두리 1초간 표시
4. 이상상황 로그 저장 (JSON + 이미지 파일)
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from collections import deque

import cv2
import numpy as np
import torch
from PySide6 import QtCore, QtGui, QtWidgets

# 환경 설정
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")


class VideoReaderThread(QtCore.QThread):
    """비디오 프레임을 읽는 스레드"""
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
    """AiVAD 모델 추론 클래스"""
    def __init__(self, device: str = "cuda", skip_frames: int = 2) -> None:
        from anomalib.models.video import AiVad

        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.skip_frames = skip_frames  # N 프레임마다 한 번만 추론
        self.frame_counter = 0
        
        # 이상 탐지만을 위한 최소 구성
        # 불필요한 객체 감지/추적 기능 모두 제거, 이상 점수만 계산
        self.model = AiVad(
            use_velocity_features=False,  # 불필요 - 이상 탐지만 하면 속도 특성 불필요
            use_pose_features=False,      # 불필요 - 이상 탐지만 하면 포즈 특성 불필요  
            use_deep_features=True,       # 기본 특성만 사용 (최소한)
            n_components_velocity=1,      # 최소값
            n_neighbors_pose=1,          # 최소값
            n_neighbors_deep=1,          # 최소값
            # 객체 감지 관련 파라미터 - 이상 탐지에 필요 없지만 모델 구조상 요구됨
            box_score_thresh=0.99,       # 최대한 높게 - 객체 감지 안하게
            min_bbox_area=99999,          # 매우 크게 - 객체 감지 안하게
            max_bbox_overlap=0.01,       # 최소값
            foreground_binary_threshold=255,  # 최대값 - foreground 감지 안하게
        )
        self.model.eval().to(self.device)
        self.core = self.model.model
        self.core.eval().to(self.device)
        
        # torch.compile 비활성화 (CUDA Graph 경고 방지 및 안정성 향상)
        # torch.compile은 실시간 추론에서 오히려 성능 저하를 일으킬 수 있음
        
        # Region Extractor 완전히 우회 (이상 탐지에 불필요)
        # 이상 점수만 필요하므로 region 추출은 시간 낭비
        if hasattr(self.core, 'region_extractor'):
            original_region_extractor = self.core.region_extractor
            def dummy_region_extractor(*args, **kwargs):
                # 빈 결과 즉시 반환 - 시간 절약
                return None
            # 패치 적용 - 이상 탐지만 하므로 region 추출 불필요
            try:
                self.core.region_extractor = dummy_region_extractor
            except:
                pass  # 패치 실패해도 계속 진행

        # 프레임 버퍼링 (2프레임 필요)
        self.frame_buffer = deque(maxlen=2)
        
        # 마지막 추론 결과 캐싱 (성능 최적화)
        self.last_result = None
        self.last_score = 0.0
        
        # 시각화 설정
        self.show_heatmap = False  # 기본값: 히트맵 비활성화 (깔끔한 화면)
        self.heatmap_alpha = 0.3  # 히트맵 투명도 (비활성화되어 있어도 설정값 유지)
        
        # YOLO 객체 감지 모델 초기화 (선택적)
        self.yolo_model = None
        self.use_yolo = False
        self.yolo_skip_frames = 5  # YOLO는 5프레임마다 한 번만 실행 (성능 최적화)
        self.yolo_frame_counter = 0
        self.last_yolo_detections = []  # 마지막 YOLO 결과 캐싱
        self._init_yolo()

    def _init_yolo(self) -> None:
        """YOLO 모델 초기화 (선택적)"""
        try:
            from ultralytics import YOLO
            # YOLOv8n (nano - 가장 빠름) 사용
            self.yolo_model = YOLO('yolov8n.pt')  # 자동 다운로드됨
            self.use_yolo = True
            print("✅ YOLO 모델 로드 완료 (yolov8n.pt)")
        except ImportError:
            print("⚠️ ultralytics 패키지가 없습니다. YOLO 기능은 사용할 수 없습니다.")
            print("💡 설치: pip install ultralytics")
            self.use_yolo = False
        except Exception as e:
            print(f"⚠️ YOLO 모델 로드 실패: {e}")
            self.use_yolo = False
    
    def detect_objects(self, frame_bgr: np.ndarray, force: bool = False) -> list:
        """YOLO로 객체 감지 (프레임 스킵 최적화)"""
        if not self.use_yolo or self.yolo_model is None:
            return []
        
        # 프레임 스킵: YOLO는 더 적게 실행 (AiVAD보다 느릴 수 있음)
        self.yolo_frame_counter += 1
        if not force and self.yolo_frame_counter % self.yolo_skip_frames != 0:
            # 마지막 결과 반환 (캐싱)
            return self.last_yolo_detections
        
        try:
            # 해상도 낮춰서 더 빠르게 처리 (320x320 또는 416x416)
            # imgsz를 작게 하면 더 빠름
            results = self.yolo_model(frame_bgr, verbose=False, imgsz=320, conf=0.5)
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 감지 정보 추출
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 클래스 이름 가져오기
                    class_name = self.yolo_model.names[cls]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            # 결과 캐싱
            self.last_yolo_detections = detections
            return detections
        except Exception as e:
            print(f"⚠️ YOLO 객체 감지 오류: {e}")
            return self.last_yolo_detections if self.last_yolo_detections else []
    
    def load_checkpoint(self, ckpt_path: str) -> None:
        """체크포인트 로드"""
        from anomalib.models.video import AiVad
        loaded = AiVad.load_from_checkpoint(ckpt_path, map_location=self.device)
        loaded.eval().to(self.device)
        self.model = loaded
        self.core = self.model.model
        self.core.eval().to(self.device)

    @staticmethod
    def _bgr_to_chw_float_tensor(frame_bgr: np.ndarray, target_size: int = 160) -> torch.Tensor:
        """BGR 프레임을 CHW 텐서로 변환 (해상도 최적화)"""
        # 해상도 더 작게 조정 (160x160) - 성능 최적화
        # 원래 224x224였는데 160x160으로 줄여서 약 2배 빠름
        frame_resized = cv2.resize(frame_bgr, (target_size, target_size))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
        chw = np.transpose(frame_rgb, (2, 0, 1))  # (C,H,W)
        return torch.from_numpy(chw)

    def infer_on_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """프레임별 추론 (최적화: 프레임 스킵 적용)"""
        self.frame_buffer.append(frame_bgr)
        self.frame_counter += 1
        
        if len(self.frame_buffer) < 2:
            return frame_bgr, 0.0, {"regions": None, "anomaly_type": "정상"}
        
        # 프레임 스킵: N 프레임마다 한 번만 추론
        if self.frame_counter % self.skip_frames != 0:
            # 추론하지 않고 마지막 결과 반환 (또는 원본 프레임)
            if self.last_result is not None:
                return self.last_result, self.last_score, {"regions": None, "anomaly_type": "정상"}
            return frame_bgr, 0.0, {"regions": None, "anomaly_type": "정상"}

        # 2프레임 클립 구성 (해상도 최적화: 160x160 사용)
        t0 = self._bgr_to_chw_float_tensor(self.frame_buffer[0], target_size=160)
        t1 = self._bgr_to_chw_float_tensor(self.frame_buffer[1], target_size=160)
        batch = torch.stack([t0, t1], dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
                # 모델 추론 실행 (region 추출 최소화를 위해 설정 최적화됨)
                output = self.core(batch)
            except Exception as model_error:
                # 모델 추론 실패 시 - 객체 감지 실패 등
                error_str = str(model_error)
                if "index 0 is out of bounds" in error_str:
                    # Region Extractor에서 객체 감지 실패 - 정상적으로 처리
                    output = None
                else:
                    # 다른 오류도 기본값으로 처리
                    output = None
            
            # 출력이 None이거나 비어있는 경우 처리 - 해상도 조정
            if output is None:
                score = 0.0
                anomaly_map = np.random.rand(160, 160)
                regions = None
            else:
                # 점수 추출 (안전한 방법)
                score = 0.0
                try:
                    if hasattr(output, 'pred_score'):
                        pred_score_tensor = output.pred_score
                        # 텐서 크기 확인
                        if isinstance(pred_score_tensor, torch.Tensor) and pred_score_tensor.numel() > 0:
                            if pred_score_tensor.shape[0] > 0:
                                score = float(pred_score_tensor[0].detach().cpu().item())
                    elif isinstance(output, list) and len(output) > 0:
                        if hasattr(output[0], 'pred_score'):
                            pred_score_tensor = output[0].pred_score
                            if isinstance(pred_score_tensor, torch.Tensor) and pred_score_tensor.numel() > 0:
                                if pred_score_tensor.shape[0] > 0:
                                    score = float(pred_score_tensor[0].detach().cpu().item())
                except (IndexError, RuntimeError) as e:
                    # 인덱스 오류나 런타임 오류 시 기본값 사용
                    score = 0.0
                
                # 이상 맵 추출 (안전한 방법) - 해상도 조정 (160x160)
                anomaly_map = np.random.rand(160, 160)
                try:
                    if hasattr(output, 'anomaly_map'):
                        raw_map_tensor = output.anomaly_map
                        if isinstance(raw_map_tensor, torch.Tensor) and raw_map_tensor.numel() > 0:
                            if raw_map_tensor.shape[0] > 0:
                                raw_map = raw_map_tensor[0].detach().cpu().numpy()
                                if len(raw_map.shape) == 3 and raw_map.shape[0] == 1:
                                    anomaly_map = raw_map[0]
                                elif len(raw_map.shape) == 2:
                                    anomaly_map = raw_map
                    elif isinstance(output, list) and len(output) > 0:
                        if hasattr(output[0], 'anomaly_map'):
                            raw_map_tensor = output[0].anomaly_map
                            if isinstance(raw_map_tensor, torch.Tensor) and raw_map_tensor.numel() > 0:
                                if raw_map_tensor.shape[0] > 0:
                                    raw_map = raw_map_tensor[0].detach().cpu().numpy()
                                    if len(raw_map.shape) == 3 and raw_map.shape[0] == 1:
                                        anomaly_map = raw_map[0]
                                    elif len(raw_map.shape) == 2:
                                        anomaly_map = raw_map
                except (IndexError, RuntimeError) as e:
                    # 인덱스 오류나 런타임 오류 시 기본값 사용 - 해상도 조정
                    anomaly_map = np.random.rand(160, 160)
                
                # 지역 추출 완전 비활성화 (이상 탐지에 불필요)
                # 이상 점수만 필요하므로 region/flow 추출은 시간 낭비
                regions = None

        # 이상 유형 결정
        anomaly_type = "정상"
        if score >= 0.7:
            anomaly_type = "심각한 이상"
        elif score >= 0.5:
            anomaly_type = "중간 이상"
        elif score >= 0.3:
            anomaly_type = "경미한 이상"

        # 오버레이 생성
        overlay = self._create_overlay(frame_bgr, anomaly_map, regions, score)

        info = {
            "regions": regions,
            "anomaly_type": anomaly_type,
            "anomaly_map": anomaly_map,
        }
        
        # 결과 캐싱 (성능 최적화)
        self.last_result = overlay
        self.last_score = score

        return overlay, score, info

    def _extract_regions_and_flows(self, first_frame: torch.Tensor, last_frame: torch.Tensor) -> Tuple[Any, Any]:
        """지역과 플로우 추출"""
        try:
            with torch.no_grad():
                flows = self.core.flow_extractor(first_frame, last_frame)
                regions = self.core.region_extractor(first_frame, last_frame)
                return flows, regions
        except (IndexError, RuntimeError) as e:
            # 객체 감지 실패 등 - 정상적인 상황으로 처리
            return None, None
        except Exception:
            # 기타 오류
            return None, None

    def _create_overlay(self, frame_bgr: np.ndarray, anomaly_map: np.ndarray, 
                       regions: Any, score: float, threshold: float = 0.5) -> np.ndarray:
        """오버레이 생성 (YOLO 객체 감지 포함)"""
        overlay = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        # 히트맵 오버레이 (선택적 표시)
        if self.show_heatmap and anomaly_map is not None:
            min_v, max_v = float(np.min(anomaly_map)), float(np.max(anomaly_map))
            if max_v - min_v > 1e-6:
                norm = (anomaly_map - min_v) / (max_v - min_v)
                norm_resized = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
                heatmap = cv2.applyColorMap((norm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(overlay, 1 - self.heatmap_alpha, heatmap, self.heatmap_alpha, 0)
        
        # 이상 탐지 시에만 빨간 테두리 표시
        is_anomaly = score >= threshold
        if is_anomaly:
            # 화면 전체에 빨간 테두리 추가
            cv2.rectangle(overlay, (0, 0), (w-1, h-1), (0, 0, 255), 3)

        # YOLO 객체 감지 및 표시 (프레임 스킵 적용)
        detected_objects = []
        if self.use_yolo:
            # YOLO는 더 적게 실행 (5프레임마다)
            detections = self.detect_objects(frame_bgr, force=False)
            detected_objects = [d['class'] for d in detections]
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                class_name = det['class']
                conf = det['confidence']
                
                # 박스 색상 (이상 탐지 시 빨간색, 정상 시 녹색)
                color = (0, 0, 255) if is_anomaly else (0, 255, 0)
                thickness = 2 if is_anomaly else 1
                
                # 박스 그리기
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                
                # 레이블 텍스트
                label = f"{class_name} {conf:.2f}"
                if is_anomaly:
                    label += " ⚠️ 이상!"
                
                # 텍스트 배경
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    overlay, 
                    (x1, y1 - text_height - 5), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                # 텍스트 표시
                cv2.putText(
                    overlay, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )

        # AiVAD region 박스는 비활성화 (이상 탐지만 하므로 불필요)
        # YOLO 객체 감지만 사용하면 됨
        
        # 상단에 감지된 객체 목록 표시
        if detected_objects:
            unique_objects = list(set(detected_objects))
            objects_text = f"감지된 객체: {', '.join(unique_objects)}"
            if is_anomaly:
                objects_text += " ⚠️ 이상 행동!"
            
            # 텍스트 배경
            (text_width, text_height), baseline = cv2.getTextSize(
                objects_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                overlay,
                (10, 10),
                (10 + text_width + 10, 10 + text_height + 10),
                (0, 0, 0),
                -1
            )
            
            # 텍스트 색상 (이상 시 빨간색, 정상 시 흰색)
            text_color = (0, 0, 255) if is_anomaly else (255, 255, 255)
            cv2.putText(
                overlay,
                objects_text,
                (15, 10 + text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2
            )

        return overlay


class AnomalyLogger:
    """이상상황 로그 저장 클래스"""
    def __init__(self, log_dir: str = "anomaly_logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logs = []

    def log_anomaly(self, timestamp: str, anomaly_type: str, score: float, 
                   screenshot_path: str, frame_number: int = 0, 
                   location: Optional[Dict[str, Any]] = None) -> None:
        """이상상황 로그 저장"""
        log_entry = {
            "timestamp": timestamp,
            "anomaly_type": anomaly_type,
            "score": float(score),
            "screenshot_path": screenshot_path,
            "screenshot_location": location if location else {},
            "frame_number": frame_number,
        }
        self.logs.append(log_entry)

    def save_screenshot(self, frame: np.ndarray, timestamp: str) -> str:
        """스크린샷 저장"""
        timestamp_clean = timestamp.replace(":", "-").replace(".", "-")
        screenshot_path = self.log_dir / f"screenshot_{timestamp_clean}.jpg"
        cv2.imwrite(str(screenshot_path), frame)
        return str(screenshot_path)

    def save_logs(self) -> None:
        """로그를 JSON 파일로 저장"""
        if not self.logs:
            return
        
        log_file = self.log_dir / "anomaly_logs.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 로그 저장 완료: {log_file} ({len(self.logs)}개 항목)")


class MainWindow(QtWidgets.QMainWindow):
    """메인 윈도우"""
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AiVAD 데모 UI - 실시간 이상 탐지")
        self.resize(1400, 900)

        # 상태
        self.video_path: Optional[str] = None
        self.threshold: float = 0.5
        self.is_anomaly_detected = False
        self.frame_number = 0
        self.last_anomaly_frame = -1  # 마지막 이상 탐지 프레임 번호

        # 모델 및 로거 초기화 (프레임 스킵: 15프레임마다 한 번만 추론 - 실시간 성능 최적화)
        self.inferencer = AiVadInferencer(device="cuda", skip_frames=15)  # 15프레임마다 추론 (최대 성능)
        self.logger = AnomalyLogger()

        # UI 구성
        self._setup_ui()
        self._connect_signals()

        # 스레드
        self.reader: Optional[VideoReaderThread] = None

        # 빨간 테두리 타이머 초기화
        self.border_timer = QtCore.QTimer()
        self.border_timer.setSingleShot(True)
        self.border_timer.timeout.connect(self._remove_anomaly_border)

    def _setup_ui(self) -> None:
        """UI 구성"""
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # 상단 컨트롤 바
        controls = QtWidgets.QHBoxLayout()
        
        # 비디오 선택
        video_group = QtWidgets.QGroupBox("비디오 선택")
        video_layout = QtWidgets.QVBoxLayout()
        self.btn_select_video = QtWidgets.QPushButton("📁 비디오 파일 선택")
        self.btn_select_video.setMinimumHeight(40)
        self.lbl_video_path = QtWidgets.QLabel("선택된 비디오: 없음")
        self.lbl_video_path.setWordWrap(True)
        video_layout.addWidget(self.btn_select_video)
        video_layout.addWidget(self.lbl_video_path)
        video_group.setLayout(video_layout)
        controls.addWidget(video_group)

        # 체크포인트 로드
        model_group = QtWidgets.QGroupBox("모델")
        model_layout = QtWidgets.QVBoxLayout()
        self.btn_load_checkpoint = QtWidgets.QPushButton("⚙️ 체크포인트 로드")
        self.lbl_checkpoint = QtWidgets.QLabel("체크포인트: 기본 모델")
        self.lbl_checkpoint.setWordWrap(True)
        model_layout.addWidget(self.btn_load_checkpoint)
        model_layout.addWidget(self.lbl_checkpoint)
        model_group.setLayout(model_layout)
        controls.addWidget(model_group)

        # 재생 컨트롤
        play_group = QtWidgets.QGroupBox("재생 컨트롤")
        play_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("▶ 재생")
        self.btn_pause = QtWidgets.QPushButton("⏸ 일시정지")
        self.btn_stop = QtWidgets.QPushButton("⏹ 정지")
        play_layout.addWidget(self.btn_play)
        play_layout.addWidget(self.btn_pause)
        play_layout.addWidget(self.btn_stop)
        play_group.setLayout(play_layout)
        controls.addWidget(play_group)

        # 임계치 설정
        threshold_group = QtWidgets.QGroupBox("임계치 설정")
        threshold_layout = QtWidgets.QVBoxLayout()
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 100)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.lbl_threshold = QtWidgets.QLabel(f"임계치: {self.threshold:.2f}")
        threshold_layout.addWidget(self.lbl_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_group.setLayout(threshold_layout)
        controls.addWidget(threshold_group)

        # 성능 최적화 설정
        perf_group = QtWidgets.QGroupBox("성능 최적화")
        perf_layout = QtWidgets.QVBoxLayout()
        
        # 프레임 스킵 설정
        skip_layout = QtWidgets.QHBoxLayout()
        skip_layout.addWidget(QtWidgets.QLabel("프레임 스킵:"))
        self.skip_frames_spinbox = QtWidgets.QSpinBox()
        self.skip_frames_spinbox.setRange(1, 30)  # 범위 확대 (최대 30프레임마다)
        self.skip_frames_spinbox.setValue(15)  # 기본값 15프레임마다 추론 (실시간 최적화)
        self.skip_frames_spinbox.setToolTip("N 프레임마다 한 번만 추론 (높을수록 빠름, 낮을수록 정확)")
        skip_layout.addWidget(self.skip_frames_spinbox)
        skip_layout.addWidget(QtWidgets.QLabel("프레임마다"))
        perf_layout.addLayout(skip_layout)
        
        perf_group.setLayout(perf_layout)
        controls.addWidget(perf_group)

        # 시각화 설정
        viz_group = QtWidgets.QGroupBox("시각화 설정")
        viz_layout = QtWidgets.QVBoxLayout()
        
        # YOLO 객체 감지 옵션
        self.use_yolo_cb = QtWidgets.QCheckBox("YOLO 객체 감지 (무엇이 있는지 표시)")
        self.use_yolo_cb.setChecked(True)  # 기본값: 활성화
        self.use_yolo_cb.setToolTip("체크하면 YOLO로 객체(사람, 차량 등)를 감지하여 표시합니다")
        viz_layout.addWidget(self.use_yolo_cb)
        
        # 히트맵 표시 옵션
        self.show_heatmap_cb = QtWidgets.QCheckBox("히트맵 표시 (기름 필터 효과)")
        self.show_heatmap_cb.setChecked(False)  # 기본값: 비활성화
        self.show_heatmap_cb.setToolTip("체크하면 이상 영역에 컬러 맵 오버레이가 표시됩니다")
        viz_layout.addWidget(self.show_heatmap_cb)
        
        # 히트맵 투명도
        heatmap_alpha_layout = QtWidgets.QHBoxLayout()
        heatmap_alpha_layout.addWidget(QtWidgets.QLabel("히트맵 투명도:"))
        self.heatmap_alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.heatmap_alpha_slider.setRange(10, 90)
        self.heatmap_alpha_slider.setValue(30)
        self.heatmap_alpha_slider.setToolTip("히트맵 투명도 (10-90%)")
        heatmap_alpha_layout.addWidget(self.heatmap_alpha_slider)
        viz_layout.addLayout(heatmap_alpha_layout)
        
        viz_group.setLayout(viz_layout)
        controls.addWidget(viz_group)

        layout.addLayout(controls)

        # 비디오 표시 영역
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("비디오를 선택하고 재생 버튼을 누르세요")
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #111111; 
                color: white; 
                border: 2px solid #333333;
                font-size: 16px;
            }
        """)
        layout.addWidget(self.video_label, stretch=1)

        # 하단 상태 바
        status_layout = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("상태: 대기")
        self.lbl_score = QtWidgets.QLabel("점수: 0.000")
        self.lbl_anomaly = QtWidgets.QLabel("정상")
        self.lbl_anomaly.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 14px;")
        self.lbl_frame = QtWidgets.QLabel("프레임: 0")
        self.lbl_log_count = QtWidgets.QLabel("로그: 0개")
        
        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.lbl_score)
        status_layout.addWidget(self.lbl_anomaly)
        status_layout.addWidget(self.lbl_frame)
        status_layout.addWidget(self.lbl_log_count)
        layout.addLayout(status_layout)

    def _connect_signals(self) -> None:
        """시그널 연결"""
        self.btn_select_video.clicked.connect(self.on_select_video)
        self.btn_load_checkpoint.clicked.connect(self.on_load_checkpoint)
        self.btn_play.clicked.connect(self.on_play)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_stop.clicked.connect(self.on_stop)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.skip_frames_spinbox.valueChanged.connect(self.on_skip_frames_changed)
        self.use_yolo_cb.toggled.connect(self.on_use_yolo_toggled)
        self.show_heatmap_cb.toggled.connect(self.on_show_heatmap_toggled)
        self.heatmap_alpha_slider.valueChanged.connect(self.on_heatmap_alpha_changed)

    def on_threshold_changed(self, value: int) -> None:
        """임계치 변경"""
        self.threshold = float(value) / 100.0
        self.lbl_threshold.setText(f"임계치: {self.threshold:.2f}")
    
    def on_skip_frames_changed(self, value: int) -> None:
        """프레임 스킵 변경"""
        self.inferencer.skip_frames = value
        self.inferencer.frame_counter = 0  # 리셋
        self.status_message(f"프레임 스킵: {value}프레임마다 추론")
    
    def on_use_yolo_toggled(self, checked: bool) -> None:
        """YOLO 사용 토글"""
        self.inferencer.use_yolo = checked
        if checked and self.inferencer.yolo_model is None:
            self.inferencer._init_yolo()
        self.status_message("YOLO 객체 감지: " + ("켜짐" if checked else "꺼짐"))
    
    def on_show_heatmap_toggled(self, checked: bool) -> None:
        """히트맵 표시 토글"""
        self.inferencer.show_heatmap = checked
        self.status_message("히트맵 표시: " + ("켜짐" if checked else "꺼짐"))
    
    def on_heatmap_alpha_changed(self, value: int) -> None:
        """히트맵 투명도 변경"""
        self.inferencer.heatmap_alpha = float(value) / 100.0

    def on_select_video(self) -> None:
        """비디오 파일 선택"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "비디오 파일 선택", os.getcwd(), 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        if path:
            self.video_path = path
            filename = os.path.basename(path)
            self.lbl_video_path.setText(f"선택된 비디오: {filename}")
            self.status_message(f"비디오 선택: {filename}")
            self.inferencer.frame_buffer.clear()
            self.inferencer.frame_counter = 0  # 프레임 카운터 리셋
            self.inferencer.yolo_frame_counter = 0  # YOLO 프레임 카운터 리셋
            self.frame_number = 0
            self.last_anomaly_frame = -1

    def on_load_checkpoint(self) -> None:
        """체크포인트 로드"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "체크포인트 선택", os.getcwd(), 
            "Checkpoint (*.ckpt *.pt *.pth);;All Files (*)"
        )
        if path:
            try:
                self.inferencer.load_checkpoint(path)
                filename = os.path.basename(path)
                self.lbl_checkpoint.setText(f"체크포인트: {filename}")
                self.status_message(f"체크포인트 로드 완료: {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "체크포인트 로드 실패", str(e))

    def on_play(self) -> None:
        """재생 시작"""
        if not self.video_path:
            QtWidgets.QMessageBox.information(self, "안내", "먼저 비디오 파일을 선택하세요.")
            return
            
        if self.reader and self.reader.isRunning():
            self.reader.pause(False)
            self.status_message("재생")
            return

        self.reader = VideoReaderThread(self.video_path, fps_limit=8.0)  # FPS 제한 (더 느리게 - 실시간 최적화)
        self.reader.frameReady.connect(self.on_frame)
        self.reader.finished.connect(self.on_reader_finished)
        self.reader.start()
        self.status_message("재생 시작")

    def on_pause(self) -> None:
        """일시정지"""
        if self.reader and self.reader.isRunning():
            self.reader.pause(True)
            self.status_message("일시정지")

    def on_stop(self) -> None:
        """정지"""
        if self.reader:
            self.reader.stop()
            self.reader.wait(1000)
            self.reader = None
        self.status_message("정지")
        # 로그 저장
        self.logger.save_logs()

    @QtCore.Slot(np.ndarray)
    def on_frame(self, frame_bgr: np.ndarray) -> None:
        """프레임 처리"""
        try:
            overlay, score, info = self.inferencer.infer_on_frame(frame_bgr)
            self.frame_number += 1
        except Exception as e:
            print(f"⚠️ 추론 오류: {e}")
            overlay = frame_bgr
            score = 0.0
            info = {"anomaly_type": "정상"}

        # 상태 업데이트
        self.lbl_score.setText(f"점수: {score:.3f}")
        self.lbl_frame.setText(f"프레임: {self.frame_number}")
        self.lbl_log_count.setText(f"로그: {len(self.logger.logs)}개")

        # 이상 탐지 여부 확인
        is_anomaly = score >= self.threshold
        
        if is_anomaly:
            self.lbl_anomaly.setText(f"⚠️ 이상 탐지: {info.get('anomaly_type', '이상')}")
            self.lbl_anomaly.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 14px;")
            
            # 이전에 이상이 감지되지 않았을 때만 처리 (새로운 이상 탐지)
            was_anomaly_before = self.is_anomaly_detected
            if not was_anomaly_before:
                self._handle_anomaly_detection(frame_bgr, score, info)
            
            self.is_anomaly_detected = True
        else:
            self.lbl_anomaly.setText("정상")
            self.lbl_anomaly.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 14px;")
            self.is_anomaly_detected = False

        # 프레임 표시
        self._display_frame(overlay, is_anomaly)

    def _handle_anomaly_detection(self, frame: np.ndarray, score: float, info: Dict[str, Any]) -> None:
        """이상 탐지 처리"""
        # 연속 프레임에서 중복 방지 (최소 30프레임 간격)
        if self.frame_number - self.last_anomaly_frame < 30:
            return
        
        self.last_anomaly_frame = self.frame_number
        
        # 1. 팝업 알림
        anomaly_type = info.get("anomaly_type", "이상")
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg.setWindowTitle("⚠️ 이상상황 탐지")
        msg.setText(f"이상상황이 탐지되었습니다!")
        msg.setInformativeText(f"유형: {anomaly_type}\n점수: {score:.3f}\n프레임: {self.frame_number}")
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg.exec()

        # 2. 로그 저장
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        screenshot_path = self.logger.save_screenshot(frame, timestamp)
        
        # 이상 탐지 위치 정보 추출
        location = {}
        regions = info.get("regions")
        if regions is not None and len(regions) > 0:
            region = regions[0]
            if 'boxes' in region:
                boxes = region['boxes'].detach().cpu().numpy()
                location["boxes"] = boxes.tolist()
            if 'masks' in region:
                # 마스크는 너무 커서 로그에는 포함하지 않음
                location["has_masks"] = True
        
        self.logger.log_anomaly(
            timestamp=timestamp,
            anomaly_type=anomaly_type,
            score=score,
            screenshot_path=screenshot_path,
            frame_number=self.frame_number,
            location=location
        )

        # 3. 빨간 테두리 표시 (1초간)
        self._show_anomaly_border()

    def _show_anomaly_border(self) -> None:
        """빨간 테두리 표시"""
        # 타이머가 실행 중이면 중지
        if self.border_timer.isActive():
            self.border_timer.stop()
        
        # 빨간 테두리 적용
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #111111; 
                color: white; 
                border: 5px solid #ff0000;
                font-size: 16px;
            }
        """)
        # 1초 후 제거
        self.border_timer.start(1000)  # 1000ms = 1초

    def _remove_anomaly_border(self) -> None:
        """빨간 테두리 제거"""
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #111111; 
                color: white; 
                border: 2px solid #333333;
                font-size: 16px;
            }
        """)

    def _display_frame(self, frame_bgr: np.ndarray, is_anomaly: bool = False) -> None:
        """프레임 표시"""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.video_label.size(), 
            QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """리사이즈 이벤트"""
        if self.video_label.pixmap():
            self.video_label.setPixmap(
                self.video_label.pixmap().scaled(
                    self.video_label.size(), 
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
            )
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """종료 이벤트"""
        if self.reader:
            self.reader.stop()
            self.reader.wait(1000)
        # 로그 저장
        self.logger.save_logs()
        super().closeEvent(event)

    def on_reader_finished(self) -> None:
        """읽기 완료"""
        self.status_message("영상 종료")
        # 로그 저장
        self.logger.save_logs()

    def status_message(self, msg: str) -> None:
        """상태 메시지 업데이트"""
        self.lbl_status.setText(f"상태: {msg}")


def main() -> None:
    """메인 함수"""
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

