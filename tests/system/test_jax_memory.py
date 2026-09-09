"""JAX memory profiling and assertion system tests.

This module provides end-to-end system tests to track, profile, and assert
GPU/device memory allocation and deallocation across different evolutionary stages
of an EDGAR run. It implements non-invasive patching of the stage-timed decorators
in the main orchestrator to measure memory footprints.

Example:
    To run this test specifically, use:
        $ uv run pytest tests/system/test_jax_memory.py -s
"""

import asyncio
import gc
import inspect
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import jax

from edgar import run as run_mod
from tests.system.fake_runner import build_fake_spec


def get_device_memory_stats() -> Dict[str, Dict[str, Any]]:
    """Retrieves current memory usage and active array allocations.

    Returns:
        A dictionary mapping device identifier or tracker to its respective memory usage metrics.
    """
    # Force garbage collection to clean up any temporary dereferenced JAX objects
    gc.collect()

    stats: Dict[str, Dict[str, Any]] = {}

    # Query active array count and memory usage across all devices using JAX APIs
    try:
        live_arrays = jax.live_arrays()
        active_arrays = [arr for arr in live_arrays if arr.size > 0]
        stats["live_arrays"] = {
            "count": len(active_arrays),
            "shapes": [arr.shape for arr in active_arrays],
        }
    except AttributeError:
        # Fallback for JAX versions without live_arrays
        stats["live_arrays"] = {"count": 0, "shapes": []}

    # Query hardware memory statistics for GPU devices if available
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []

    for device in gpu_devices:
        dev_stats = device.memory_stats()
        if dev_stats:
            stats[f"gpu:{device.id}"] = {
                "bytes_in_use": dev_stats.get("bytes_in_use", 0),
                "peak_bytes_in_use": dev_stats.get("peak_bytes_in_use", 0),
            }

    return stats


def test_gpu_memory_profiling() -> None:
    """Profiles GPU and device memory usage across different execution stages of an EDGAR run.

    This test runs the evolutionary pipeline using fake LLMs and hooks into the
    stage-timed wrappers at the top of edgar/run.py to capture memory telemetry before
    and after each phase. It verifies that memory is tracked and no leaks are retained
    by asserting that net memory and JAX array allocations do not increase across stages.
    """
    # Use a temporary output directory for the test run
    test_output_dir = Path(__file__).parents[2] / "test_output_gpu_profile"
    spec = build_fake_spec(test_output_dir)

    # Set up stage tracking
    stages_to_patch: List[str] = [
        "t_seed",
        "t_translate_seeds",
        "t_score_seeds",
        "t_fits_seeds",
        "t_spawn",
        "t_generate_models",
        "t_generate_param_ests",
        "t_translate_programs",
        "t_score",
        "t_fits",
        "t_deduplicate",
        "t_prune",
        "t_migrate",
        "t_score_validate",
    ]
    memory_history: Dict[str, Dict[str, Any]] = {}

    def create_tracker(stage_name: str, original_func: Any) -> Any:
        """Wraps the target function to record memory telemetry.

        Args:
            stage_name: Name of the stage being profiled.
            original_func: The callable being wrapped.

        Returns:
            A wrapped function that logs memory telemetry before and after execution.
        """
        if inspect.iscoroutinefunction(original_func):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                before = get_device_memory_stats()
                result = await original_func(*args, **kwargs)
                after = get_device_memory_stats()
                memory_history[stage_name] = {"before": before, "after": after}
                return result

            return async_wrapper
        else:

            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                before = get_device_memory_stats()
                result = original_func(*args, **kwargs)
                after = get_device_memory_stats()
                memory_history[stage_name] = {"before": before, "after": after}
                return result

            return sync_wrapper

    # Apply patching
    patches = []
    for stage in stages_to_patch:
        if hasattr(run_mod, stage):
            original = getattr(run_mod, stage)
            tracker = create_tracker(stage, original)
            p = patch.object(run_mod, stage, side_effect=tracker)
            patches.append(p)
            p.start()

    try:
        # Run the EDGAR experiment
        asyncio.run(run_mod.run(spec))
    finally:
        # Clean up all patches regardless of failure/success
        for p in patches:
            p.stop()

    # Log/Assert findings
    print("\n--- GPU and Device Memory Profile Across Stages ---")
    for stage, stats in memory_history.items():
        before_stats = stats["before"]
        after_stats = stats["after"]

        print(f"Stage: {stage}")
        print(f"  Before: {before_stats}")
        print(f"  After:  {after_stats}")

        # Assert no net increase in live JAX arrays in the main process
        before_count = before_stats["live_arrays"]["count"]
        after_count = after_stats["live_arrays"]["count"]
        assert after_count <= before_count, (
            f"JAX array leak detected in stage '{stage}': "
            f"live arrays increased from {before_count} to {after_count}."
        )

        # Assert no net increase in GPU bytes in use for any detected GPU devices
        for key in after_stats.keys():
            if key.startswith("gpu:"):
                before_bytes = before_stats.get(key, {}).get("bytes_in_use", 0)
                after_bytes = after_stats.get(key, {}).get("bytes_in_use", 0)
                assert after_bytes <= before_bytes, (
                    f"GPU memory leak detected on {key} in stage '{stage}': "
                    f"bytes in use increased from {before_bytes} to {after_bytes}."
                )

    # Ensure that we actually hit and recorded telemetry for the targeted stages
    assert len(memory_history) == 14, (
        "Memory profile information missing from some target function."
    )
