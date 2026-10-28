import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True


def _infer_sp_simple(q, *, text_len: int, latent_frames: int) -> int:
    video_len = q.shape[2] - text_len
    assert video_len % latent_frames == 0, (video_len, latent_frames)
    return video_len // latent_frames
