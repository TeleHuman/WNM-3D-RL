# Modified by the WNM-3D-RL contributors, 2026.
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import math
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils.profiler import ProfilerConfig
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.config.optimizer import OptimizerConfig

from .model import DiffusionModelConfig

__all__ = [
    "DiffusionLossConfig",
    "DiffusionActorConfig",
    "FSDPDiffusionActorConfig",
]


@dataclass
class DiffusionLossConfig(BaseConfig):
    loss_mode: str = "flow_grpo"
    clip_ratio: float = 0.0001
    visual_clip_ratio: Optional[float] = None
    action_clip_ratio: Optional[float] = None
    adv_clip_max: float = 5.0
    visual_log_prob_weight: float = 1.0
    action_log_prob_weight: float = 1.0
    action_stop_loss_weight: float = 0.0
    action_stop_use_chunk_weights: bool = False
    action_collision_loss_weight: float = 0.0
    action_collision_use_chunk_weights: bool = False
    mix_beta: float = 0.5
    ref_kl_coef: float = 0.0
    adaptive_weight_min: float = 1e-5
    dpo_beta: float = 2000.0
    kl_mask_threshold: float = 1e-5
    add_kl_coefficient: bool = True

    def __post_init__(self):
        """Validate diffusion loss configuration."""
        valid_modes = ["flow_grpo", "flow_dppo", "grpo_guard", "diffusion_nft", "dpo", "dance_grpo"]
        if self.loss_mode not in valid_modes:
            raise ValueError(f"Invalid diffusion loss_mode: {self.loss_mode}. Must be one of {valid_modes}")
        if self.adv_clip_max <= 0:
            raise ValueError(f"Diffusion adv_clip_max must be positive, got {self.adv_clip_max}.")
        for name in ("clip_ratio", "visual_clip_ratio", "action_clip_ratio"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Diffusion {name} must be finite and non-negative, got {value}.")
        for name in (
            "visual_log_prob_weight",
            "action_log_prob_weight",
            "action_stop_loss_weight",
            "action_collision_loss_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Diffusion {name} must be finite and non-negative, got {value}.")
        if self.visual_log_prob_weight == 0 and self.action_log_prob_weight == 0:
            raise ValueError("At least one diffusion log-probability weight must be positive.")
        if self.mix_beta <= 0:
            raise ValueError(f"mix_beta must be positive, got {self.mix_beta}.")
        if self.adaptive_weight_min <= 0:
            raise ValueError(f"adaptive_weight_min must be positive, got {self.adaptive_weight_min}.")
        if self.kl_mask_threshold <= 0:
            raise ValueError(f"kl_mask_threshold must be positive, got {self.kl_mask_threshold}.")


@dataclass
class DiffusionActorConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size_per_gpu: int = MISSING
    diffusion_loss: DiffusionLossConfig = field(default_factory=DiffusionLossConfig)
    loss_scale_factor: Optional[float] = None
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 42
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: DiffusionModelConfig = field(default_factory=BaseConfig)
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    profiler: Optional[ProfilerConfig] = None

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Rollout Correction config.
    # When bypass_mode=True, ``diffusion_loss`` computes per-step RS from here.
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)

    def __post_init__(self):
        """Validate diffusion actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING


@dataclass
class FSDPDiffusionActorConfig(DiffusionActorConfig):
    # Training strategy: fsdp or fsdp2
    strategy: str = "fsdp"
    grad_clip: float = 1.0
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)

    def __post_init__(self):
        """Validate diffusion FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config
        # Sync strategy to engine config so engine_workers can pick the right FSDP version.
        # EngineConfig.strategy defaults to None, so without this, engine_workers.py always
        # falls back to FSDP1 even when actor.strategy="fsdp2".
        object.__setattr__(self.engine, "strategy", self.strategy)
