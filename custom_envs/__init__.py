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
    if old_dyld and not old_dyld.split(":")[0] == cmeel_lib_s and not os.environ.get("CUSTOM_ENVS_DYLD_REEXEC"):
        env = os.environ.copy()
        paths = [cmeel_lib_s]
        paths.extend(p for p in old_dyld.split(":") if p and p != cmeel_lib_s)
        env["DYLD_LIBRARY_PATH"] = ":".join(paths)
        env["CUSTOM_ENVS_DYLD_REEXEC"] = "1"
        if sys.argv and sys.argv[0] == "-m":
            os.execve(
                sys.executable,
                [sys.executable, "-m", "custom_envs.mug_contact_probe.run_probe", *sys.argv[1:]],
                env,
            )
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

    paths = [cmeel_lib_s]
    if old_dyld:
        paths.extend(p for p in old_dyld.split(":") if p and p != cmeel_lib_s)
    os.environ["DYLD_LIBRARY_PATH"] = ":".join(paths)


_prefer_venv_cmeel_libs()

from .mass_memory_bin_sort import MassMemoryBinSortEnv
from .mug_contact_probe import MugContactProbeEnv
from .sort_ycb_into_bins import SortYCBIntoBinsEnv

__all__ = ["SortYCBIntoBinsEnv", "MassMemoryBinSortEnv", "MugContactProbeEnv"]
