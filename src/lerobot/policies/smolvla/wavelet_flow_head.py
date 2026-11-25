"""PyTorch port of Ruyi231/Wafm wavelet_flow_head.py (bf54f865).

Haar decomposition of noisy actions and expert tokens, independent FiLM heads,
and inverse Haar synthesis of the velocity field. Time integration stays in
action coordinates, as in the reference implementation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def multi_level_haar_dwt(x, levels):
    if levels < 0:
        raise ValueError("levels must be nonnegative")
    actual = min(levels, (x.shape[1] - 1).bit_length())
    pad = (-x.shape[1]) % (2**actual)
    if pad:
        x = torch.cat((x, x[:, -1:].expand(-1, pad, -1)), dim=1)
    details = []
    for _ in range(actual):
        even, odd = x[:, 0::2], x[:, 1::2]
        x = (even + odd) / math.sqrt(2)
        details.append((even - odd) / math.sqrt(2))
    return x, details, actual


def multi_level_haar_idwt(approx, details, target_length=None):
    x = approx
    for detail in reversed(details):
        even = (x + detail) / math.sqrt(2)
        odd = (x - detail) / math.sqrt(2)
        x = torch.stack((even, odd), dim=2).flatten(1, 2)
    return x if target_length is None else x[:, :target_length]


class SubbandFiLMHead(nn.Module):
    def __init__(self, action_dim, token_dim, bottleneck_dim):
        super().__init__()
        self.state_proj = nn.Linear(action_dim, bottleneck_dim)
        self.film_proj = nn.Linear(token_dim, 2 * bottleneck_dim)
        self.out_proj = nn.Linear(bottleneck_dim, action_dim)

    def forward(self, state, condition):
        gamma, beta = self.film_proj(condition).chunk(2, dim=-1)
        hidden = (1 + 0.1 * gamma.tanh()) * self.state_proj(state) + 0.1 * beta
        return self.out_proj(F.gelu(hidden, approximate="tanh"))


class WaveletSubbandFlowHead(nn.Module):
    def __init__(self, action_dim, token_dim, levels=2, bottleneck_dim=128):
        super().__init__()
        self.levels = levels
        self.approx_head = SubbandFiLMHead(action_dim, token_dim, bottleneck_dim)
        self.detail_heads = nn.ModuleList(
            [SubbandFiLMHead(action_dim, token_dim, bottleneck_dim) for _ in range(levels)]
        )

    @torch.no_grad()
    def initialize_from_linear(self, linear: nn.Linear, *, horizon: int):
        """Preserve the loaded linear readout using GELU(z)-GELU(-z)=z.

        Called after loading a non-wavelet base checkpoint, never after loading
        saved wavelet weights. Extra hidden units keep their random input
        features and start with zero output weights, so they can learn rather
        than becoming permanently inactive. This method consumes no RNG state.
        """
        action_dim = linear.out_features
        levels = min(self.levels, (horizon - 1).bit_length())
        for index, head in enumerate([self.approx_head, *self.detail_heads]):
            hidden = head.state_proj.out_features
            if hidden < 2 * action_dim:
                raise ValueError("Linear-preserving initialization needs at least 2*action_dim units")
            band_bias = linear.bias if linear.bias is not None else linear.weight.new_zeros(action_dim)
            band_bias = band_bias * (2 ** (levels / 2)) if index == 0 else torch.zeros_like(band_bias)
            head.state_proj.weight[:2 * action_dim].zero_()
            head.state_proj.bias[:2 * action_dim].zero_()
            head.film_proj.weight[hidden:hidden + action_dim].copy_(10 * linear.weight)
            head.film_proj.weight[hidden + action_dim:hidden + 2 * action_dim].copy_(-10 * linear.weight)
            head.film_proj.bias[hidden:hidden + action_dim].copy_(10 * band_bias)
            head.film_proj.bias[hidden + action_dim:hidden + 2 * action_dim].copy_(-10 * band_bias)
            head.out_proj.weight.zero_()
            head.out_proj.bias.zero_()
            identity = torch.eye(action_dim, device=linear.weight.device, dtype=linear.weight.dtype)
            head.out_proj.weight[:, :action_dim].copy_(identity)
            head.out_proj.weight[:, action_dim:2 * action_dim].copy_(-identity)

    def forward(self, noisy_actions, action_tokens):
        approx, details, actual = multi_level_haar_dwt(noisy_actions, self.levels)
        cond_a, cond_d, _ = multi_level_haar_dwt(action_tokens, actual)
        velocity_a = self.approx_head(approx, cond_a)
        velocity_d = [
            head(detail, condition)
            for head, detail, condition in zip(self.detail_heads, details, cond_d, strict=False)
        ]
        velocity = multi_level_haar_idwt(velocity_a, velocity_d, noisy_actions.shape[1])
        return velocity, (velocity_a, velocity_d, actual)


def wavelet_flow_loss(target, info, action_dim, actions_is_pad=None):
    """Sum of per-band MSEs, reduced per example over valid coefficients.

    Adaptation for LeRobot: exclude padded action dimensions and any coefficient
    whose Haar support crosses the episode boundary. Edge extension of the chunk
    follows the reference, including for the validity mask.
    """
    pred_a, pred_d, levels = info
    target_a, target_d, _ = multi_level_haar_dwt(target, levels)
    valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
    if actions_is_pad is not None:
        valid = ~actions_is_pad
    pad = (-valid.shape[1]) % (2**levels)
    if pad:
        valid = torch.cat((valid, valid[:, -1:].expand(-1, pad)), dim=1)
    masks = []
    for _ in range(levels):
        valid = valid[:, 0::2] & valid[:, 1::2]
        masks.append(valid)
    result = target.new_zeros(target.shape[0])
    for pred, truth, mask in zip([pred_a] + pred_d, [target_a] + target_d, [valid] + masks, strict=True):
        squared = (pred[..., :action_dim] - truth[..., :action_dim]).square()
        result = result + (squared * mask[..., None]).sum((1, 2)) / (
            mask.sum(1) * action_dim
        ).clamp_min(1)
    return result
