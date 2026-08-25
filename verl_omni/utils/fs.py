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

import os

from verl.utils.fs import copy_to_local

__all__ = ["resolve_model_local_dir"]


def resolve_model_local_dir(path: str, use_shm: bool = False, local_files_only: bool = False) -> str:
    """Resolve ``path`` to an on-disk directory.

    ``local_files_only`` is intended for air-gapped training jobs.  In that
    mode a missing directory is an error and the Hugging Face Hub fallback is
    never imported or called.
    """
    path = os.path.expanduser(path)
    if local_files_only and not os.path.isdir(path):
        raise FileNotFoundError(f"Local model directory does not exist: {path}")

    local_path = copy_to_local(path, use_shm=use_shm)
    if not os.path.isdir(local_path):
        if local_files_only:
            raise FileNotFoundError(f"Resolved local model directory does not exist: {local_path}")
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(path)
    return local_path
