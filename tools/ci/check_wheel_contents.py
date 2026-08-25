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
"""Validate the public contents and metadata of a WNM-3D-RL wheel."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def _single_dist_info(names: set[str]) -> str:
    candidates = sorted({PurePosixPath(name).parts[0] for name in names if ".dist-info/" in name})
    if len(candidates) != 1:
        raise AssertionError(f"expected exactly one .dist-info directory, found {candidates}")
    return candidates[0]


def check_wheel(path: Path) -> None:
    """Raise AssertionError when *path* violates the release wheel contract."""
    with ZipFile(path) as wheel:
        names = set(wheel.namelist())
        dist_info = _single_dist_info(names)
        top_level = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        expected_top_level = {"verl_omni", dist_info}
        if top_level != expected_top_level:
            raise AssertionError(
                f"unexpected wheel top-level entries: expected {sorted(expected_top_level)}, found {sorted(top_level)}"
            )

        required = {
            f"{dist_info}/licenses/LICENSE",
            f"{dist_info}/licenses/THIRD_PARTY_NOTICES.md",
            f"{dist_info}/METADATA",
        }
        missing = required - names
        if missing:
            raise AssertionError(f"missing required wheel files: {sorted(missing)}")

        metadata = BytesParser().parsebytes(wheel.read(f"{dist_info}/METADATA"))
        if metadata["Name"] != "verl-omni":
            raise AssertionError(f"unexpected distribution name: {metadata['Name']!r}")
        if metadata["Requires-Python"] != "<3.13,>=3.11":
            raise AssertionError(f"unexpected Requires-Python: {metadata['Requires-Python']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    check_wheel(args.wheel)
    print(f"WHEEL_CONTENTS_OK {args.wheel}")


if __name__ == "__main__":
    main()
