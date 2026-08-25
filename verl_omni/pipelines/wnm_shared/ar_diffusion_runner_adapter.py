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

"""Request-batch runner for stateless WNM rollouts.

The WNM RL pipeline uses the stateless joint Dance-SDE path and implements
``DiffusionRequestBatch`` itself.  It must therefore use vLLM-Omni's generic
diffusion runner: the AR-Diffusion runner owns a session-scoped paged KV cache
whose slot mappings intentionally support only one sequence.  Selecting that
runner for an eight-request WAM batch is both unnecessary and invalid.
"""

from typing import Any

from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


class WNMDiffusionModelRunner(DiffusionModelRunner):
    """Use the generic runner for either WNM model variant."""

    def execute_model(self, req: Any, kv_prefetch_jobs: dict | None = None):
        del kv_prefetch_jobs
        return super().execute_model(req)
