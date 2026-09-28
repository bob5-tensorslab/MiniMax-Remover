"""Shared CUDA device selection for all MiniMax-Remover entry points."""

import os

import torch


def get_device():
    """Select the configured CUDA GPU, defaulting to physical GPU 1."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    raw_index = os.environ.get("MINIMAX_GPU_INDEX", "1")
    try:
        gpu_index = int(raw_index)
    except ValueError as exc:
        raise ValueError(
            f"MINIMAX_GPU_INDEX must be a non-negative integer, got {raw_index!r}."
        ) from exc

    device_count = torch.cuda.device_count()
    if gpu_index < 0 or gpu_index >= device_count:
        raise ValueError(
            f"MINIMAX_GPU_INDEX={gpu_index} is unavailable; this process can see "
            f"{device_count} CUDA device(s)."
        )

    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)
    return device
