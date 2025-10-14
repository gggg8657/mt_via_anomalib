import torch
import os

# cuDNN 오류 해결을 위한 환경 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # CUDA 디버깅 활성화


# cuDNN 설정 조정 - cuDNN을 완전히 비활성화
torch.backends.cudnn.enabled = False  # cuDNN 완전 비활성화
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False

# 텐서 연속성 보장을 위한 설정
torch.set_float32_matmul_precision('medium')  # H100 Tensor Core 최적화

# GPU 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

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
class AiVadWithMoreEpochs(_BaseAiVad):
    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        # 기본 AI-VAD가 max_epochs=1로 고정하므로 여기서 원하는 값으로 덮어씀
        return {"gradient_clip_val": 0, "max_epochs": 5, "num_sanity_val_steps": 0}

model = AiVadWithMoreEpochs()

# Train using the engine with specific settings to avoid cuDNN issues
engine = Engine(
    devices=1,  # 단일 GPU 사용
    accelerator='gpu',
    precision='32',  # 32-bit precision 사용 (cuDNN 호환성)
    max_epochs=5,
    limit_train_batches=5,
    limit_val_batches=2,
    accumulate_grad_batches=1,
)
engine.fit(model=model, datamodule=datamodule)