import torch
import os

# CPU 사용을 위한 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU 비활성화
torch.backends.cudnn.enabled = False

from anomalib.models.video import AiVad
from anomalib.data import Avenue
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.engine import Engine

# Initialize model and datamodule
datamodule = Avenue(
    clip_length_in_frames=2,
    frames_between_clips=1,
    target_frame=VideoTargetFrame.LAST,
    num_workers=0,  # 멀티프로세싱 비활성화
)
model = AiVad()

# Train using the engine with CPU
engine = Engine(
    devices=1,  # 단일 디바이스 사용
    accelerator='cpu',  # CPU 사용
    precision='32',  # 32-bit precision 사용
    max_epochs=1,  # 테스트를 위해 1 에포크만 실행
    limit_train_batches=2,  # 배치 수 제한 (메모리 사용량 감소)
    limit_val_batches=1,
    accumulate_grad_batches=1,  # 그래디언트 누적 비활성화
)

print("CPU에서 실행 시작...")
engine.fit(model=model, datamodule=datamodule)
print("실행 완료!")

