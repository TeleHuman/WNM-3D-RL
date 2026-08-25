# Third-party and external dependency notices

Source code distributed in this repository is under the Apache License 2.0 in
`LICENSE` unless a file states otherwise. That license does not include or
relicense external model code, weights, datasets, or assets.

## VERL-Omni and VERL

This repository is derived from ByteDance's
[`verl-project/verl-omni`](https://github.com/verl-project/verl-omni), licensed
under Apache-2.0, at upstream revision
`04259f651dc7ca80db39deeac195065fc3f3d5f7`. Upstream-derived files retain
their applicable copyright and license notices; files changed by this project
also carry a modification notice. WNM-specific integrations, recipes, rewards,
data tools, and tests include both new project files and modifications to the
upstream foundation.

[`verl-project/verl`](https://github.com/verl-project/verl) is a separate
Apache-2.0 runtime dependency pinned at revision
`8a694930275061f52ebd538c906ef8819af56dbd`. VERL is installed externally and
is not vendored or relicensed by this repository.

## WNM-3D integration in this repository

This repository contains project-specific Stage-III training integrations,
recipes, rewards, configuration, and tooling for WNM-3D. Original source code
contributed directly to this repository is released under Apache-2.0 unless a
file states otherwise. This notice identifies the provenance and license of
the distributed source; it does not make claims about ownership of abstract
ideas or methods and does not alter any third-party rights.

The separately published
[`TeleHuman/WNM-3D`](https://github.com/TeleHuman/WNM-3D) implementation and
model artifacts are not redistributed or relicensed here; their own
component-level terms apply. Nothing in this repository relicenses third-party
material incorporated into or used with the model.

## NVIDIA DreamZero

DreamZero is NVIDIA GEAR Lab work. Its official source repository is Copyright
2025 NVIDIA Corporation and licensed under Apache-2.0:
[`dreamzero0/dreamzero`](https://github.com/dreamzero0/dreamzero). The external
project checkout is based on revision
`ab790c198fbce33503358efbbd4187ce9a89adf3` plus project-owned modifications.

Redistribution of NVIDIA-derived source must comply with Apache-2.0, including
providing the license, marking modified files, and retaining applicable
copyright, patent, trademark, attribution, and NOTICE material. DreamZero model
artifacts have their own per-checkpoint metadata: the official
`DreamZero-AgiBot` checkpoint is marked Apache-2.0, while `DreamZero-DROID` is
marked CC BY-NC 4.0. This repository distributes neither checkpoint and does
not infer one checkpoint's terms for another artifact.

Research using DreamZero should cite *World Action Models are Zero-shot
Policies* (Ye et al., 2026, arXiv:2602.15922).

## Meta VGGT and VGGT-Omega

The current Stage-3 recipe uses Meta's **VGGT-Omega**, not the ordinary VGGT-1B
checkpoint. VGGT-Omega code, model weights, outputs, results, and derivative
works are governed by the
[`FAIR Noncommercial Research License`](https://github.com/facebookresearch/vggt-omega/blob/main/LICENSE).
That license limits them to noncommercial research use, requires derivatives to
be distributed under the same agreement with a copy of it, requires research
publication acknowledgement, and incorporates Meta's acceptable-use policy.

As between the project owner and Meta, the project owner owns its modifications
and derivative additions; Meta retains its rights in the VGGT-Omega research
materials. Consequently, the internally developed combination model is
project-owned, but any use or distribution that includes, derives from, or
depends on VGGT-Omega materials remains subject to Meta's license.

For clarity, ordinary
[`facebookresearch/vggt`](https://github.com/facebookresearch/vggt) has a
different, commercial-use-friendly code license. Meta states that only the
separate `VGGT-1B-Commercial` checkpoint permits commercial use; the original
VGGT-1B checkpoint remains noncommercial. Those terms do not replace the
VGGT-Omega license used by this recipe.

Research using VGGT-Omega should cite *VGGT-Ω* (Wang et al., 2026,
arXiv:2605.15195).

The external WNM-3D distribution also incorporates DINOv3 components under
Meta's separate DINOv3 License. This repository does not vendor those
components; users of the WNM-3D source or checkpoints must follow the license
files and third-party notices shipped by WNM-3D.

## Models, weights, and data

This repository does not distribute Wan2.2, UMT5, CLIP, VGGT-Omega, DINOv3,
NVIDIA DreamZero checkpoints, WNM-3D checkpoints, InteriorGS data, or generated
evaluation media. Their respective licenses and data-use terms apply
independently; no rights to those artifacts are granted here.
