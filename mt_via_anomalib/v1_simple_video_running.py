from anomalib.models.video import AiVad
from anomalib.data import Avenue
from anomalib.data.utils import VideotargetFrame
from anomalib.engine import Engine

# Initialize model and datamodule
datamodule = Avenue(
    clip_length_in_frames=2,
    frames_between_clips=1,
    target_frame=VideoTargetFrame.LAST
)
model = AiVad()