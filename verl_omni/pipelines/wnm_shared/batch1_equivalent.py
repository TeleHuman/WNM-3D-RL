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

"""Shared WNM kernels that preserve deployed batch-one semantics."""

from __future__ import annotations

import logging
import os
from types import MethodType

import torch

logger = logging.getLogger(__name__)


def _batch1_equivalent_category_linear_forward(
    self: torch.nn.Module,
    x: torch.Tensor,
    cat_ids: torch.Tensor,
) -> torch.Tensor:
    """Evaluate every row through the exact B=1 ``torch.bmm`` path.

    WNM deployment runs one request at a time. On H100/BF16, cuBLAS
    selects a different batched-BMM reduction for B>1 in the small action
    encoder. Its sub-ULP output drift is then amplified by the 30 joint-DiT
    blocks. Keeping only these three tiny projections row-wise makes a B>1 RL
    rollout numerically equivalent to independent deployment requests while
    preserving autograd and batching for the expensive transformer.
    """

    if x.ndim != 3:
        raise ValueError(f"WNM CategorySpecificLinear expects x with shape (B,T,D), got {tuple(x.shape)}.")
    batch_size = int(x.shape[0])
    cat_ids = cat_ids.reshape(-1)
    if int(cat_ids.numel()) != batch_size:
        raise ValueError(
            "WNM CategorySpecificLinear requires one category per row: "
            f"batch={batch_size}, categories={int(cat_ids.numel())}."
        )

    outputs = []
    for row in range(batch_size):
        row_ids = cat_ids[row : row + 1]
        row_weight = self.W[row_ids]
        row_bias = self.b[row_ids]
        outputs.append(torch.bmm(x[row : row + 1], row_weight) + row_bias.unsqueeze(1))
    return torch.cat(outputs, dim=0)


def install_batch1_equivalent_action_encoder(
    module: torch.nn.Module,
    *,
    component: str,
) -> None:
    """Patch only one joint DiT's three action projections, never GN0 code."""

    enabled = os.getenv("WAM_BATCH1_EQUIVALENT_ACTION_ENCODER", "false").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        logger.warning(
            "%s deployment-equivalent action encoder is disabled; set "
            "WAM_BATCH1_EQUIVALENT_ACTION_ENCODER=true only for an explicit parity experiment",
            component,
        )
        return

    action_encoder = getattr(module, "action_encoder", None)
    if action_encoder is None:
        raise AttributeError(f"{component} has no action_encoder to make batch-one equivalent.")

    projection_names = ("W1", "W2", "W3")
    for name in projection_names:
        projection = getattr(action_encoder, name, None)
        if projection is None or not hasattr(projection, "W") or not hasattr(projection, "b"):
            raise TypeError(f"{component} action_encoder.{name} is not a CategorySpecificLinear projection.")
        projection.forward = MethodType(
            _batch1_equivalent_category_linear_forward,
            projection,
        )
        projection._wam_batch1_equivalent = True

    logger.warning(
        "%s installed deployment-equivalent action encoder: projections=%s B>1 executes independent B=1 bmm kernels",
        component,
        projection_names,
    )


__all__ = ["install_batch1_equivalent_action_encoder"]
