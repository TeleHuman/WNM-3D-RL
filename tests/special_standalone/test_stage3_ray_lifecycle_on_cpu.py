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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAY_CLUSTER = REPO_ROOT / "recipes" / "wnm_3d" / "stage3" / "runtime" / "ray_cluster.sh"


def _run_ray_cluster(
    tmp_path: Path,
    fake_ray_body: str,
    *,
    raylet_present: bool = False,
    **overrides: str,
):
    wnm_source_root = tmp_path / "wnm-3d"
    wnm_source_root.mkdir()
    ray_log = tmp_path / "ray.log"
    fake_ray = tmp_path / "ray"
    fake_ray.write_text(
        f'#!/usr/bin/env bash\nset -eu\nprintf \'%s\\n\' "$*" >> "$FAKE_RAY_LOG"\n{fake_ray_body}\n',
        encoding="utf-8",
    )
    fake_ray.chmod(0o755)
    fake_pgrep = tmp_path / "pgrep"
    fake_pgrep.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'if [[ "$*" == "-x raylet" ]]; then\n'
        f"  exit {0 if raylet_present else 1}\n"
        "fi\n"
        'exec /usr/bin/pgrep "$@"\n',
        encoding="utf-8",
    )
    fake_pgrep.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "WNM3D_SOURCE_ROOT": str(wnm_source_root),
            "FAKE_RAY_LOG": str(ray_log),
            "MASTER_ADDR": "127.0.0.1",
            "NODE_IP": "127.0.0.1",
            "NODE_RANK": "1",
            "NNODES": "2",
            "PYTHON_BIN": sys.executable,
            "PATH": f"{tmp_path}:{env['PATH']}",
            "RAY_BIN": str(fake_ray),
            "WAM_REPO_ROOT": str(REPO_ROOT),
            "WAM_VERIFY_EIGHT_HCA_TOPOLOGY": "false",
        }
    )
    env.update(overrides)
    result = subprocess.run(
        ["bash", str(RAY_CLUSTER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    calls = ray_log.read_text(encoding="utf-8").splitlines() if ray_log.exists() else []
    return result, calls


def test_existing_ray_fails_without_implicit_stop(tmp_path):
    result, calls = _run_ray_cluster(
        tmp_path,
        '[[ "$1" == "status" ]] && exit 0\nexit 99',
        raylet_present=True,
    )

    assert result.returncode == 2
    assert calls == []
    assert "refusing to stop it implicitly" in result.stderr


def test_explicit_replacement_calls_node_local_stop(tmp_path):
    result, calls = _run_ray_cluster(
        tmp_path,
        '[[ "$1" == "stop" ]] && exit 23\nexit 99',
        WAM_REPLACE_EXISTING_RAY="true",
    )

    assert result.returncode == 23
    assert calls == ["stop --force"]


def test_invalid_replacement_flag_fails_before_ray_mutation(tmp_path):
    result, calls = _run_ray_cluster(
        tmp_path,
        "exit 99",
        WAM_REPLACE_EXISTING_RAY="yes",
    )

    assert result.returncode == 2
    assert calls == []
    assert "must be true or false" in result.stderr
