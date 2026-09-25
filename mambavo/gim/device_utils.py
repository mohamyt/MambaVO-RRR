import torch


# NOTE: auto-detects CUDA availability. Falls back to CPU if no GPU is found
def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"