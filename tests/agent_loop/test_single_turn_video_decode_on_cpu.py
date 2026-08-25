# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_single_turn_module():
    modules = {
        "verl": _package("verl"),
        "verl.experimental": _package("verl.experimental"),
        "verl.experimental.agent_loop": _package("verl.experimental.agent_loop"),
        "verl.experimental.agent_loop.agent_loop": types.ModuleType("verl.experimental.agent_loop.agent_loop"),
        "verl.utils": _package("verl.utils"),
        "verl.utils.profiler": types.ModuleType("verl.utils.profiler"),
        "verl_omni": _package("verl_omni"),
        "verl_omni.agent_loop": _package("verl_omni.agent_loop"),
        "verl_omni.agent_loop.diffusion_agent_loop": types.ModuleType("verl_omni.agent_loop.diffusion_agent_loop"),
    }
    modules["verl.experimental.agent_loop.agent_loop"].AgentLoopBase = object
    modules["verl.experimental.agent_loop.agent_loop"].register = lambda _: lambda cls: cls
    modules["verl.utils.profiler"].simple_timer = lambda *args, **kwargs: None
    modules["verl_omni.agent_loop.diffusion_agent_loop"].DiffusionAgentLoopOutput = object
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        module_path = Path(__file__).resolve().parents[2] / "verl_omni" / "agent_loop" / "single_turn_agent_loop.py"
        spec = importlib.util.spec_from_file_location("_single_turn_video_decode_test", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


def test_distinct_video_cache_misses_can_decode_concurrently():
    module = _load_single_turn_module()
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def fake_decode(path, expected_frames):
        nonlocal active, maximum_active
        del path, expected_frames
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1
        return object()

    first_path = "/tmp/a.mp4"
    first_stripe = hash((first_path, 66)) % len(module._EXACT_VIDEO_CACHE_LOCKS)
    second_path = next(
        f"/tmp/b-{index}.mp4"
        for index in range(128)
        if hash((f"/tmp/b-{index}.mp4", 66)) % len(module._EXACT_VIDEO_CACHE_LOCKS) != first_stripe
    )
    with patch.object(module, "_decode_exact_rgb_video_cached", side_effect=fake_decode):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(module._decode_exact_rgb_video, first_path, 66),
                executor.submit(module._decode_exact_rgb_video, second_path, 66),
            ]
            for future in futures:
                future.result()

    assert maximum_active == 2


def test_matching_video_cache_misses_remain_serialized():
    module = _load_single_turn_module()
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def fake_decode(path, expected_frames):
        nonlocal active, maximum_active
        del path, expected_frames
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1
        return object()

    with patch.object(module, "_decode_exact_rgb_video_cached", side_effect=fake_decode):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(module._decode_exact_rgb_video, "/tmp/same.mp4", 66) for _ in range(2)]
            for future in futures:
                future.result()

    assert maximum_active == 1
