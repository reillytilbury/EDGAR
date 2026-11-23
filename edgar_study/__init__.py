"""EDGAR study tools.

Before JAX starts, expose one XLA CPU device per core so `edgar_study.fitting` can split cells
across cores; XLA otherwise runs this elementwise work on about one thread. This only adds
host devices: arrays still default to the first device, and GPU hosts are unaffected.
"""

import os

if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + f" --xla_force_host_platform_device_count={os.cpu_count()}").strip()
