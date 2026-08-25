# Copyright 2026 WNM-3D-RL contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backward-only gradient gain shared by both WNM model integrations."""

from __future__ import annotations

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)

_ENV_NAME = "WAM_ACTION_BACKBONE_GRAD_GAIN"
_DEFAULT_GAIN = 1.0


class _ScaleGradient(torch.autograd.Function):
    """Preserve the forward tensor exactly while scaling its backward gradient."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, gain: float) -> torch.Tensor:
        ctx.gain = float(gain)
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient * ctx.gain, None


def resolve_action_backbone_gradient_gain() -> float:
    """Return the configured action-to-backbone gradient gain."""

    gain = float(os.environ.get(_ENV_NAME, str(_DEFAULT_GAIN)))
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"{_ENV_NAME} must be finite and positive, got {gain}.")
    return gain


def install_action_backbone_gradient_gain(
    module: torch.nn.Module,
    *,
    gain: float | None = None,
) -> float:
    """Scale only the action decoder input gradient flowing into the shared DiT.

    The pre-hook is installed only on the actor-side joint DiT.  Its forward is
    an exact identity, so rollout and deployment values are unchanged.  The
    action decoder parameter gradients are also unchanged; only the gradient
    returned from the decoder input to the shared token backbone is scaled.
    """

    resolved_gain = resolve_action_backbone_gradient_gain() if gain is None else float(gain)
    if not math.isfinite(resolved_gain) or resolved_gain <= 0.0:
        raise ValueError(f"action backbone gradient gain must be finite and positive, got {resolved_gain}.")

    existing = getattr(module, "_verl_action_backbone_gradient_gain", None)
    if existing is not None:
        if float(existing) != resolved_gain:
            raise RuntimeError(
                "WNM action backbone gradient gain is already installed "
                f"with gain={existing}, requested={resolved_gain}."
            )
        return resolved_gain

    action_decoder = getattr(module, "action_decoder", None)
    if not isinstance(action_decoder, torch.nn.Module):
        raise AttributeError("WNM joint DiT has no action_decoder for action-backbone gradient scaling.")

    def scale_decoder_input(
        _decoder: torch.nn.Module,
        args: tuple[object, ...],
    ) -> tuple[object, ...] | None:
        if resolved_gain == 1.0 or not torch.is_grad_enabled():
            return None
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("WNM action_decoder must receive its shared hidden tensor as the first positional input.")
        return (_ScaleGradient.apply(args[0], resolved_gain), *args[1:])

    action_decoder.register_forward_pre_hook(scale_decoder_input, prepend=True)
    module._verl_action_backbone_gradient_gain = resolved_gain
    logger.warning(
        "Installed backward-only WNM action-to-backbone gradient gain %.4g; forward values are unchanged",
        resolved_gain,
    )
    return resolved_gain


__all__ = [
    "install_action_backbone_gradient_gain",
    "resolve_action_backbone_gradient_gain",
]
