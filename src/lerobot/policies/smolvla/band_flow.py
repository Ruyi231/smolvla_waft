"""Earlier A2 denoising for the two complete Haar blocks in LIBERO H10."""

import math

import torch
from torch import Tensor


def band_flow_bridge(
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    alpha: float,
    actions_is_pad: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the interpolated actions and their velocity with respect to flow time."""
    t = time[:, None, None]
    velocity = noise - actions
    state = t * noise + (1 - t) * actions
    if alpha == 0:
        return state, velocity

    arm_blocks = velocity[:, :8, :6].reshape(actions.shape[0], 2, 4, 6)
    block_means = arm_blocks.mean(dim=2, keepdim=True)
    if actions_is_pad is not None:
        valid_blocks = ~actions_is_pad[:, :8].reshape(actions.shape[0], 2, 4).any(dim=2)
        block_means = block_means * valid_blocks[:, :, None, None]

    coarse = torch.zeros_like(velocity)
    coarse[:, :8, :6] = block_means.expand_as(arm_blocks).reshape(actions.shape[0], 8, 6)
    schedule = t - alpha * (math.pi * t).sin().square() / math.pi
    derivative = 1 - alpha * (2 * math.pi * t).sin()
    return state + (schedule - t) * coarse, velocity + (derivative - 1) * coarse
