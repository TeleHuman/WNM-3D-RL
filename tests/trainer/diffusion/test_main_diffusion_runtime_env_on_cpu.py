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

from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import Role

import verl_omni.trainer.main_diffusion as main_diffusion
from verl_omni.trainer.main_diffusion import (
    TaskRunner,
    _wnm_ray_runtime_env_vars,
)


def test_wnm_runtime_env_propagates_controls_without_secrets():
    env = {
        "WAM_ROLLOUT_BULK_CPU_OUTPUT": "true",
        "WNM3D_SOURCE_ROOT": "/opt/wnm-3d",
        "WNM_AUTH_TOKEN": "do-not-forward",
        "NCCL_IB_HCA": "=mlx5_0,mlx5_1",
        "NUM_INFERENCE_STEPS": "16",
        "ENABLE_DIT_CACHE": "true",
        "CFG_SCALE": "5.0",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_TOKEN": "do-not-forward",
        "AWS_SECRET_ACCESS_KEY": "do-not-forward",
        "UNRELATED_FLAG": "false",
    }

    propagated = _wnm_ray_runtime_env_vars(env)

    assert propagated == {
        "WAM_ROLLOUT_BULK_CPU_OUTPUT": "true",
        "WNM3D_SOURCE_ROOT": "/opt/wnm-3d",
        "NCCL_IB_HCA": "=mlx5_0,mlx5_1",
        "NUM_INFERENCE_STEPS": "16",
        "ENABLE_DIT_CACHE": "true",
        "CFG_SCALE": "5.0",
        "OPENBLAS_NUM_THREADS": "1",
    }


def test_validation_reference_kl_selects_immutable_reference_worker(monkeypatch):
    config = OmegaConf.create(
        {
            "algorithm": {
                "sample_source": "online",
                "validation_reference_kl": {"enabled": True},
            },
            "actor_rollout_ref": {"model": {"lora": {"rank": 0}, "lora_adapter_path": None}},
        }
    )
    monkeypatch.setattr(main_diffusion, "need_reference_policy", lambda _: False)
    monkeypatch.setattr(main_diffusion.ray, "remote", lambda worker: worker)

    runner = TaskRunner()
    runner.add_actor_rollout_worker(config)

    assert Role.ActorRolloutRef in runner.role_worker_mapping
    assert Role.ActorRollout not in runner.role_worker_mapping
