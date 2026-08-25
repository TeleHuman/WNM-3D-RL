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

"""Shared configuration helpers for temporal action-chunk credit."""

from __future__ import annotations

import math
import os


def action_chunk_credit_enabled() -> bool:
    value = os.environ.get("WAM_ACTION_CHUNK_CREDIT_ENABLED", "false")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"WAM_ACTION_CHUNK_CREDIT_ENABLED must be a boolean, got {value!r}.")


def action_chunk_size() -> int:
    value = int(os.environ.get("WAM_ACTION_CHUNK_SIZE", "8"))
    if value <= 0:
        raise ValueError(f"WAM_ACTION_CHUNK_SIZE must be positive, got {value}.")
    return value


def action_chunk_weights(*, expected_chunks: int | None = None) -> tuple[float, ...]:
    raw = os.environ.get("WAM_ACTION_CHUNK_WEIGHTS", "1,0.5,0.25,0.125")
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError(f"WAM_ACTION_CHUNK_WEIGHTS must be a comma-separated list of numbers, got {raw!r}.") from exc
    if not values:
        raise ValueError("WAM_ACTION_CHUNK_WEIGHTS must not be empty.")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"WAM_ACTION_CHUNK_WEIGHTS must contain finite positive values, got {values}.")
    if expected_chunks is not None and len(values) != expected_chunks:
        raise ValueError(
            "WAM_ACTION_CHUNK_WEIGHTS count must match the action horizon chunks: "
            f"expected={expected_chunks}, actual={len(values)}, weights={values}."
        )
    return values


def normalized_action_chunk_weights(*, expected_chunks: int | None = None) -> tuple[float, ...]:
    values = action_chunk_weights(expected_chunks=expected_chunks)
    total = sum(values)
    return tuple(value / total for value in values)
