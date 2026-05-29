"""Local custom ManiSkill environments for this repo."""

import os
from pathlib import Path
import sys


def _prefer_venv_cmeel_libs() -> None:
    """Keep pip Pinocchio from resolving against Homebrew's older dylibs."""
    py_mm = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.prefix) / "lib" / py_mm / "site-packages" / "cmeel.prefix" / "lib",
        # Backward-compatible fallback for existing local setups.
        Path(__file__).resolve().parents[1] / ".venv-ms3-of2" / "lib" / py_mm / "site-packages" / "cmeel.prefix" / "lib",
    ]
    cmeel_lib = next((p for p in candidates if p.exists()), None)
    if cmeel_lib is None:
        return

    cmeel_lib_s = str(cmeel_lib)
    old_dyld = os.environ.get("DYLD_LIBRARY_PATH")
    paths = [cmeel_lib_s]
    if old_dyld:
        paths.extend(p for p in old_dyld.split(":") if p and p != cmeel_lib_s)
    os.environ["DYLD_LIBRARY_PATH"] = ":".join(paths)


from .mug_contact_probe import DejaVuEnv

DejaVuMemoryEnv = None
MugContactProbeEnv = None

__all__ = ["DejaVuEnv", "DejaVuMemoryEnv", "MugContactProbeEnv"]
