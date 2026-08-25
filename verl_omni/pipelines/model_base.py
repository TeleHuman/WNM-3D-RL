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

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Optional

import torch
from diffusers import ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl_omni.utils.action_chunk_credit import (
    action_chunk_credit_enabled,
    action_chunk_size,
)
from verl_omni.workers.config import DiffusionModelConfig

logger = logging.getLogger(__name__)


class DiffusionModelBase(ABC):
    """Abstract base class for diffusion model training helpers.

    Different diffusion models have very different forward / sampling logic.
    Subclass this ABC and implement the three abstract methods to plug your
    model into the verl training loop.

    To register, decorate your subclass with
    ``@DiffusionModelBase.register("name", algorithm="...")``. The *name* must match the
    ``_class_name`` value in the pipeline's ``model_index.json`` (which is
    auto-detected into ``DiffusionModelConfig.architecture``). The *algorithm*
    must match ``DiffusionModelConfig.algorithm``.

    Example::

        @DiffusionModelBase.register("WanPipeline", algorithm="dance_grpo")
        class Wan22DanceGRPO(DiffusionModelBase):
            ...
    """

    _registry: dict[tuple[str, str], type["DiffusionModelBase"]] = {}

    @classmethod
    def register(cls, architecture: str, algorithm: str):
        """Class decorator that registers a subclass for ``(architecture, algorithm)``."""

        def decorator(subclass: type["DiffusionModelBase"]) -> type["DiffusionModelBase"]:
            cls._registry[(architecture, algorithm)] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, model_config: DiffusionModelConfig) -> type["DiffusionModelBase"]:
        """Return the registered subclass for ``(architecture, algorithm)``."""
        architecture = model_config.architecture
        algorithm = model_config.algorithm

        return cls.get_class_by_name(architecture, algorithm, model_config.external_lib)

    @classmethod
    def get_class_by_name(
        cls,
        architecture: str,
        algorithm: str,
        external_lib: Optional[str] = None,
    ) -> type["DiffusionModelBase"]:
        """Resolve an adapter before a full ``DiffusionModelConfig`` exists."""
        key = (architecture, algorithm)
        if external_lib is not None:
            from verl.utils.import_utils import import_external_libs

            import_external_libs(external_lib)
        try:
            return cls._registry[key]
        except KeyError:
            registered = sorted(cls._registry.keys())
            raise NotImplementedError(
                f"No diffusion model registered for (architecture={architecture!r}, "
                f"algorithm={algorithm!r}). Registered: {registered}. "
                f"Set ``external_lib`` in DiffusionModelConfig to load your implementation."
            ) from None

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype) -> Optional[torch.nn.Module]:
        """Load the model without ``diffusers.AutoModel``.

        Return ``None`` to use the default ``AutoModel`` path.
        Override this for models that diffusers cannot load.
        """
        return None

    @classmethod
    def configure_train_mode(cls, module: torch.nn.Module) -> None:
        """Hook called after ``module.train()`` for architecture-specific overrides."""
        return

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> Optional[str]:
        """Prepare model-specific processor files before ``hf_processor()`` loads them.

        Override this when a model ships a ``processor`` directory that needs
        adapter-owned config fixes before Hugging Face can load it. Return an
        alternate processor path when the model directory should not be
        modified in place.
        """
        return None

    @classmethod
    def configure_trainable_params(
        cls,
        module: torch.nn.Module,
        model_config: DiffusionModelConfig,
    ) -> None:
        """Hook called after module build to set ``requires_grad`` on trainable params.

        Args:
            module: The loaded model module (pre-FSDP).
            model_config: The ``DiffusionModelConfig``.
        """
        return

    @classmethod
    @abstractmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> SchedulerMixin:
        """Build and configure the diffusion scheduler for this model.
        The returned scheduler should have timesteps and sigmas already set.

        Args:
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
        """
        pass

    @classmethod
    @abstractmethod
    def set_timesteps(cls, scheduler: SchedulerMixin, model_config: DiffusionModelConfig, device: str):
        """Set timesteps and sigmas on the scheduler and move them to *device*.

        Args:
            scheduler (SchedulerMixin): the scheduler used for the diffusion process.
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
            device (str): the device to move the timesteps and sigmas to.
        """
        pass

    @classmethod
    @abstractmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        """Build architecture-specific inputs for a model forward.
        For reverse-trajectory algorithms, ``latents`` and ``timesteps`` usually
        contain the full rollout trajectory and ``step`` selects the current
        slice. For forward-process objectives, callers may pass an already
        selected/noised latent and timestep directly.
        The caller is responsible for universal pre-processing (common tensor extraction
        and nested-embed unpadding) before invoking this method.

        Args:
            module (ModelMixin): the diffusion transformer module.
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
            latents (torch.Tensor): latent tensor from the micro-batch; either a full trajectory
                of shape (B, T, ...) or a selected/noised latent of shape (B, ...).
            timesteps (torch.Tensor): timestep tensor from the micro-batch; either a full
                trajectory of shape (B, T) or a selected timestep of shape (B,).
            prompt_embeds (torch.Tensor): dense positive prompt embeddings, shape (B, L, D).
            prompt_embeds_mask (torch.Tensor): attention mask for prompt_embeds, shape (B, L).
            negative_prompt_embeds (torch.Tensor): dense negative prompt embeddings, shape (B, L, D).
            negative_prompt_embeds_mask (torch.Tensor): attention mask for negative_prompt_embeds.
            micro_batch (TensorDict): the full micro-batch, available for architecture-specific
                metadata (e.g. height, width, vae_scale_factor).
            step (int): the current denoising step index.
        """
        pass

    @classmethod
    @abstractmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Forward the model and sample the previous step.
        Used for RL-algorithms based on reversed-sampling (FlowGRPO, DanceGRPO, etc.).

        Args:
            module (ModelMixin): the diffusion model to be forwarded.
            scheduler (SchedulerMixin): the scheduler used for the diffusion process.
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
            model_inputs (dict[str, torch.Tensor]): the inputs to the diffusion model.
            negative_model_inputs (Optional[dict[str, torch.Tensor]]): the negative inputs for guidance.
            scheduler_inputs (Optional[TensorDict | dict[str, torch.Tensor]]): the extra inputs for the scheduler,
                which may contain the latents and timesteps.
            step (int): the current step in the diffusion process.

        Returns:
            Visual-only adapters return ``(log_prob, prev_sample_mean,
            std_dev_t, sqrt_dt)``. World-action adapters return a mapping with
            that visual group plus ``action_log_probs``,
            ``action_prev_sample_mean``, ``action_std_dev_t`` and
            ``action_sqrt_dt``.
        """
        pass

    @classmethod
    def forward(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run a single model prediction.
        Used both for forward-process objectives (noising clean latents ``x0 -> xt``
        then optimizing predictions directly) and as the prediction step inside
        reverse-sampling algorithms (FlowGRPO et al.). Model adapters only need to
        override this when prediction requires extra handling such as CFG, negative
        inputs, or output conversion.
        """
        return module(**model_inputs)[0]


class WorldActionDiffusionModelBase(DiffusionModelBase):
    """Base adapter for diffusion policies that jointly predict visuals and actions.

    A WAM model forward is expected to produce both the visual and action flow
    predictions.  The expensive model forward is executed exactly once per
    denoising step; the shared SDE scheduler then evaluates the two recorded
    transitions independently.

    Rollout/replay batches must contain ``all_latents`` / ``all_timesteps`` for
    the visual trajectory and ``all_action_latents`` /
    ``all_action_timesteps`` for the action trajectory.  ``actions`` remains the
    final environment-facing action and is not used as a substitute for the
    recorded action diffusion trajectory.
    """

    @classmethod
    def inject_action_inputs(
        cls,
        model_inputs: dict[str, torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> dict[str, torch.Tensor]:
        """Inject the recorded action state into the joint WAM forward.

        WNM-style modules use ``action`` and ``timestep_action``. Models
        with different argument names may override this hook.
        """
        required_keys = ("all_action_latents", "all_action_timesteps")
        missing_keys = [key for key in required_keys if key not in micro_batch]
        if missing_keys:
            raise KeyError(f"World-action model inputs are missing: {missing_keys}.")
        model_inputs = dict(model_inputs)
        action = micro_batch["all_action_latents"][:, step].detach()
        action_timestep = micro_batch["all_action_timesteps"][:, step].detach()

        visual_reference = None
        for name in ("x", "hidden_states"):
            candidate = model_inputs.get(name)
            if isinstance(candidate, torch.Tensor):
                visual_reference = candidate
                break
        if visual_reference is None:
            visual_reference = next(
                (
                    value
                    for value in model_inputs.values()
                    if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
                ),
                None,
            )
        if visual_reference is not None:
            if torch.is_floating_point(action) and torch.is_floating_point(visual_reference):
                action = action.to(device=visual_reference.device, dtype=visual_reference.dtype)
            else:
                action = action.to(device=visual_reference.device)
            # Timesteps may be integer indices or floating schedule values. Move
            # them with the model inputs, but never cast their dtype here.
            action_timestep = action_timestep.to(device=visual_reference.device)

        if action_timestep.ndim == 1 and action.ndim > 2:
            action_timestep = action_timestep[:, None].expand(action.shape[:-1])
        model_inputs["action"] = action
        model_inputs["timestep_action"] = action_timestep
        return model_inputs

    @classmethod
    def forward_world_action(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return visual/action predictions from one model invocation.

        The default contract accepts either ``(visual, action, ...)`` or a
        mapping with ``video_noise_pred`` and ``action_noise_pred``.  WAMs with
        CFG-specific semantics should override this hook; in particular, an
        action prediction must not silently inherit visual CFG behavior.
        """
        if negative_model_inputs:
            raise NotImplementedError(
                f"{cls.__name__} received negative_model_inputs. Override "
                "forward_world_action() to define visual/action CFG semantics."
            )

        output = module(**model_inputs)
        if isinstance(output, Mapping):
            try:
                return output["video_noise_pred"], output["action_noise_pred"]
            except KeyError as exc:
                raise KeyError(
                    "World-action model mapping output must contain `video_noise_pred` and `action_noise_pred`."
                ) from exc

        if isinstance(output, tuple | list) and len(output) >= 2:
            return output[0], output[1]

        raise TypeError(
            "World-action model forward must return (visual_prediction, action_prediction) "
            "or a mapping with video_noise_pred/action_noise_pred."
        )

    @staticmethod
    def _algo_value(model_config: DiffusionModelConfig, action_name: str, shared_name: str):
        algo = model_config.algo
        action_value = getattr(algo, action_name, None)
        if action_value is None and hasattr(algo, "get"):
            action_value = algo.get(action_name, None)
        return action_value if action_value is not None else getattr(algo, shared_name)

    @classmethod
    def replay_prediction_source_steps(
        cls,
        scheduler_inputs: TensorDict | dict[str, torch.Tensor],
    ) -> tuple[int, ...]:
        """Return the DiT prediction consumed by every recorded transition.

        Most world-action policies evaluate DiT once per scheduler transition.
        Cached inference adapters override this hook and validate their recorded
        cache map before the actor is allowed to reuse a prediction graph.
        """

        timesteps = scheduler_inputs.get("all_timesteps")
        if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 2:
            raise ValueError("World-action replay requires all_timesteps with shape (batch, policy_steps).")
        return tuple(range(int(timesteps.shape[1])))

    @staticmethod
    def _runtime_noise_level(
        scheduler_inputs: TensorDict | dict[str, torch.Tensor],
        *,
        key: str,
        fallback: float,
    ) -> float:
        """Read the noise actually used by rollout, rejecting mixed replay batches."""

        value = scheduler_inputs.get(key, None)
        if value is None:
            return float(fallback)
        if not isinstance(value, torch.Tensor):
            return float(value)
        flat = value.detach().float().reshape(-1)
        if flat.numel() == 0:
            raise ValueError(f"World-action replay field {key!r} is empty.")
        if not torch.allclose(flat, flat[:1].expand_as(flat), rtol=0.0, atol=1e-7):
            raise ValueError(
                f"World-action actor micro-batch mixes different {key} values; "
                "split the batch before likelihood replay."
            )
        return float(flat[0].item())

    @classmethod
    def sample_previous_step_from_predictions(
        cls,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        visual_prediction: torch.Tensor,
        action_prediction: torch.Tensor,
        scheduler_inputs: TensorDict | dict[str, torch.Tensor],
        step: int,
    ) -> dict[str, torch.Tensor]:
        """Evaluate one recorded transition from an already-computed joint prediction.

        Separating DiT prediction from scheduler evaluation is required for exact
        deployed 8/16 cache replay: one source prediction graph may contribute
        policy losses from several consecutive scheduler transitions.
        """

        required_keys = ("all_latents", "all_timesteps", "all_action_latents", "all_action_timesteps")
        missing_keys = [key for key in required_keys if key not in scheduler_inputs]
        if missing_keys:
            raise KeyError(f"World-action replay is missing trajectory fields: {missing_keys}.")

        visual_latents = scheduler_inputs["all_latents"]
        visual_timesteps = scheduler_inputs["all_timesteps"]
        action_latents = scheduler_inputs["all_action_latents"]
        action_timesteps = scheduler_inputs["all_action_timesteps"]
        if visual_timesteps.ndim != 2 or action_timesteps.ndim != 2:
            raise ValueError(
                "World-action scheduler timesteps must have shape (batch, policy_steps); "
                f"visual={tuple(visual_timesteps.shape)}, action={tuple(action_timesteps.shape)}."
            )
        if visual_timesteps.shape != action_timesteps.shape:
            raise ValueError(
                "World-action visual/action schedules must have identical batch and transition counts; "
                f"visual={tuple(visual_timesteps.shape)}, action={tuple(action_timesteps.shape)}."
            )
        batch_size, policy_steps = visual_timesteps.shape
        expected_trajectory_prefix = (batch_size, policy_steps + 1)
        trajectory_shapes = {
            "all_latents": tuple(visual_latents.shape),
            "all_action_latents": tuple(action_latents.shape),
        }
        for name, trajectory in (("all_latents", visual_latents), ("all_action_latents", action_latents)):
            if trajectory.ndim < 2 or tuple(trajectory.shape[:2]) != expected_trajectory_prefix:
                raise ValueError(
                    f"World-action {name} must contain exactly one more state than policy transitions and share "
                    f"the schedule batch size; expected prefix={expected_trajectory_prefix}, "
                    f"actual={tuple(trajectory.shape)}."
                )
        if step < 0 or step >= policy_steps:
            raise IndexError(
                f"World-action replay step {step} is outside the valid range [0, {policy_steps}); "
                f"trajectories={trajectory_shapes}."
            )

        action_policy_mask = scheduler_inputs.get("action_policy_mask", None)
        if action_policy_mask is not None:
            if not isinstance(action_policy_mask, torch.Tensor):
                raise TypeError(
                    f"World-action action_policy_mask must be a torch.Tensor, got {type(action_policy_mask).__name__}."
                )
            if action_policy_mask.ndim == action_latents.ndim:
                if action_policy_mask.shape[0] not in (1, batch_size) or action_policy_mask.shape[1] not in (
                    1,
                    policy_steps,
                ):
                    raise ValueError(
                        "Transition-wise action_policy_mask must have batch/step dimensions broadcastable to "
                        f"({batch_size}, {policy_steps}); got {tuple(action_policy_mask.shape)}."
                    )
                mask_step = 0 if action_policy_mask.shape[1] == 1 else step
                action_policy_mask = action_policy_mask[:, mask_step].detach()
            elif action_policy_mask.ndim > action_latents.ndim - 1:
                raise ValueError(
                    "Static action_policy_mask must be broadcastable to one action state, or include exactly one "
                    f"transition dimension; mask={tuple(action_policy_mask.shape)}, "
                    f"action_state={tuple(action_latents[:, step].shape)}."
                )
            else:
                action_policy_mask = action_policy_mask.detach()

        visual_noise_level = cls._runtime_noise_level(
            scheduler_inputs,
            key="noise_level",
            fallback=model_config.algo.noise_level,
        )
        action_noise_level = cls._runtime_noise_level(
            scheduler_inputs,
            key="action_noise_level",
            fallback=cls._algo_value(model_config, "action_noise_level", "noise_level"),
        )
        _, visual_log_prob, visual_mean, visual_std, visual_sqrt_dt = scheduler.sample_previous_step(
            sample=visual_latents[:, step].detach().float(),
            model_output=visual_prediction.float(),
            timestep=visual_timesteps[:, step].detach(),
            noise_level=visual_noise_level,
            prev_sample=visual_latents[:, step + 1].detach().float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        _, action_log_prob, action_mean, action_std, action_sqrt_dt = scheduler.sample_previous_step(
            sample=action_latents[:, step].detach().float(),
            model_output=action_prediction.float(),
            timestep=action_timesteps[:, step].detach(),
            noise_level=action_noise_level,
            prev_sample=action_latents[:, step + 1].detach().float(),
            sde_type=cls._algo_value(model_config, "action_sde_type", "sde_type"),
            return_logprobs=True,
            return_sqrt_dt=True,
            log_prob_mask=action_policy_mask,
            log_prob_chunk_size=(action_chunk_size() if action_chunk_credit_enabled() else None),
        )

        return {
            "log_probs": visual_log_prob,
            "prev_sample_mean": visual_mean,
            "std_dev_t": visual_std,
            "sqrt_dt": visual_sqrt_dt,
            "action_log_probs": action_log_prob,
            "action_prev_sample_mean": action_mean,
            "action_std_dev_t": action_std,
            "action_sqrt_dt": action_sqrt_dt,
        }

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ) -> dict[str, torch.Tensor]:
        """Compute visual and action transition log-probabilities in one forward."""
        if scheduler_inputs is None:
            raise ValueError("World-action replay requires scheduler_inputs.")
        visual_prediction, action_prediction = cls.forward_world_action(
            module=module,
            model_config=model_config,
            model_inputs=model_inputs,
            negative_model_inputs=negative_model_inputs,
        )
        return cls.sample_previous_step_from_predictions(
            scheduler=scheduler,
            model_config=model_config,
            visual_prediction=visual_prediction,
            action_prediction=action_prediction,
            scheduler_inputs=scheduler_inputs,
            step=step,
        )


class DiffusionI2IModelBase(DiffusionModelBase):
    """Base class for image-conditioned diffusion model training helpers.

    Inherits all T2I logic from :class:`DiffusionModelBase`. Adds a two-step
    condition injection hook:

    1. ``prepare_condition`` extracts condition tensors from ``micro_batch``.
    2. ``inject_condition`` merges condition tensors into ``model_inputs``.

    The training dispatcher requires I2I adapters to return a non-empty
    condition. ``inject_condition`` itself remains a no-op for direct callers
    that pass ``None``.

    The default ``inject_condition`` implements a common concat-crop pattern:
    concatenate ``image_latents`` onto ``hidden_states``
    along the token dimension and set ``_target_seq_len`` so that
    :meth:`DiffusionI2IModelBase.forward` slices the prediction back to the
    noise segment. Models with non-concat conditioning (Wan I2V, LTX2 I2AV)
    override ``inject_condition``.
    """

    @classmethod
    def forward(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run concat-conditioned I2I prediction and keep the target-token prefix."""
        model_inputs = dict(model_inputs)
        target_seq_len = model_inputs.pop("_target_seq_len", None)
        if negative_model_inputs is not None:
            negative_model_inputs = dict(negative_model_inputs)
            negative_target_seq_len = negative_model_inputs.pop("_target_seq_len", None)
            if target_seq_len is None:
                target_seq_len = negative_target_seq_len
            elif negative_target_seq_len is not None and negative_target_seq_len != target_seq_len:
                raise ValueError(
                    "Positive and negative I2I inputs have different target sequence lengths: "
                    f"{target_seq_len} and {negative_target_seq_len}."
                )
        noise_pred = super().forward(module, model_config, model_inputs, negative_model_inputs)
        if target_seq_len is None:
            return noise_pred
        if noise_pred.shape[1] < target_seq_len:
            raise ValueError(
                f"forward: model output seq_len ({noise_pred.shape[1]}) < "
                f"target_seq_len ({target_seq_len}). The condition concat may "
                f"have been dropped or the model truncated the output."
            )
        return noise_pred[:, :target_seq_len]

    @classmethod
    def prepare_condition(
        cls,
        micro_batch: TensorDict,
        latents: torch.Tensor,
        step: int,
    ) -> Optional[dict]:
        """Extract condition fields from ``micro_batch``.

        T2I default returns ``None``. I2I adapters override this to pull
        model-specific condition tensors from the micro-batch and return them
        under the keys that :meth:`inject_condition` expects. The default
        concat-crop implementation requires ``image_latents``. Adapters that
        need position metadata or non-concat conditioning must override
        :meth:`inject_condition`.

        Note: the *micro-batch* keys carrying condition tensors must not
        collide with keys the MFU FLOPs counter interprets as the denoised
        latent (``image_latents``, ``latents_clean``, ``all_latents``,
        ``audio_latents``). Use a distinct key such as
        ``condition_image_latents`` on the micro-batch, then map it to the
        ``image_latents`` slot in the returned condition dict.

        Args:
            micro_batch (TensorDict): the full micro-batch.
            latents (torch.Tensor): the latent tensor for the current step.
            step (int): the current denoising step index.

        Returns:
            Optional[dict]: a flat dict of condition tensors, or ``None``
            when no condition is present (T2I degenerate path).
        """
        return None

    @classmethod
    def inject_condition(
        cls,
        model_inputs: dict,
        negative_model_inputs: Optional[dict],
        condition: Optional[dict],
    ) -> tuple[dict, Optional[dict]]:
        """Merge condition tensors into ``model_inputs``.

        Default implementation: concatenate ``image_latents`` onto
        ``hidden_states`` along the token dimension and set
        ``_target_seq_len`` so that
        :meth:`DiffusionI2IModelBase.forward` slices the prediction back.

        When ``condition`` is ``None`` or empty, this is a no-op (T2I
        degenerate path). Models with non-concat conditioning (Wan I2V,
        LTX2 I2AV) override this method.

        """
        if not condition:
            return model_inputs, negative_model_inputs

        image_latents = condition.get("image_latents")
        if image_latents is None:
            raise ValueError("inject_condition requires condition['image_latents']")

        # Guard: "image_latents" is reserved by the MFU FLOPs counter.
        if "image_latents" in model_inputs:
            raise ValueError(
                "inject_condition: 'image_latents' found in model_inputs; "
                "this key is reserved by the MFU FLOPs counter for the denoised "
                "latent. The rollout adapter likely output 'image_latents' instead "
                "of 'condition_image_latents'. Check the rollout adapter's "
                "custom_output keys."
            )

        hidden_states = model_inputs["hidden_states"]
        if image_latents.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "inject_condition: condition image_latents batch size "
                f"({image_latents.shape[0]}) does not match hidden_states batch size "
                f"({hidden_states.shape[0]})."
            )

        if image_latents.dim() != 3:
            raise ValueError(
                f"inject_condition: condition image_latents must be 3-D "
                f"(batch, seq, dim), got shape {image_latents.shape}"
            )

        target_seq_len = hidden_states.shape[1]
        for inputs in (model_inputs, negative_model_inputs):
            if inputs is None:
                continue
            inputs["hidden_states"] = torch.cat(
                [
                    inputs["hidden_states"],
                    image_latents.to(
                        device=inputs["hidden_states"].device,
                        dtype=inputs["hidden_states"].dtype,
                    ),
                ],
                dim=1,
            )
            inputs["_target_seq_len"] = target_seq_len

        return model_inputs, negative_model_inputs


class VllmOmniPipelineBase:
    """Registry base for vllm-omni custom diffusion pipeline classes.

    To register, decorate your custom pipeline class with
    ``@VllmOmniPipelineBase.register("name", algorithm="...")``. The *name* must match the
    ``_class_name`` value in the pipeline's ``model_index.json`` (which is
    auto-detected into ``DiffusionModelConfig.architecture``). The *algorithm*
    must match ``DiffusionModelConfig.algorithm``.

    Example::

        @VllmOmniPipelineBase.register("WanPipeline", algorithm="dance_grpo")
        class Wan22DanceGRPOPipelineWithLogProb(WanPipeline):
            ...
    """

    _registry: dict[tuple[str, str], type] = {}

    @classmethod
    def register(cls, architecture: str, algorithm: str):
        """Class decorator that registers a pipeline for ``(architecture, algorithm)``."""

        def decorator(subclass: type) -> type:
            if "supports_request_batch" not in subclass.__dict__:
                subclass.supports_request_batch = False
            cls._registry[(architecture, algorithm)] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, architecture: str, algorithm: str) -> type | None:
        """Return the registered pipeline class for ``(architecture, algorithm)``, or ``None``."""
        return cls._registry.get((architecture, algorithm))

    @classmethod
    def get_pipeline_path(cls, architecture: str, algorithm: str) -> str | None:
        """Return the fully-qualified dotted import path for ``(architecture, algorithm)``, or ``None``."""
        pipeline_cls = cls.get_class(architecture, algorithm)
        if pipeline_cls is None:
            return None
        return f"{pipeline_cls.__module__}.{pipeline_cls.__qualname__}"
