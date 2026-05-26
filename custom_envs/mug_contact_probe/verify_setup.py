#!/usr/bin/env python3
"""Minimal setup check for ManiSkill3 + ObjectFolder 2.0 + UniTouch checkpoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OBJECTFOLDER_ROOT = ROOT / "third_party" / "ObjectFolder"
MUG_OBJECT_FILE = ROOT / "assets" / "objectfolder2" / "025_mug_ObjectFile.pth"
UNITOUCH_CKPT = ROOT / "assets" / "unitouch" / "last_new.ckpt"


def check_maniskill_import() -> None:
    import mani_skill  # noqa: F401
    print("[OK] Imported ManiSkill3")


def check_objectfolder_import() -> None:
    if str(OBJECTFOLDER_ROOT) not in sys.path:
        sys.path.insert(0, str(OBJECTFOLDER_ROOT))
    importlib.import_module("load_osf")
    print("[OK] Imported ObjectFolder 2.0 code (load_osf)")


def check_mug_objectfile_load() -> None:
    if not MUG_OBJECT_FILE.exists():
        raise FileNotFoundError(f"Missing mug ObjectFile: {MUG_OBJECT_FILE}")
    checkpoint = torch.load(MUG_OBJECT_FILE, map_location="cpu", weights_only=False)
    required_keys = {"VisionNet", "AudioNet", "TouchNet"}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise RuntimeError(f"Mug ObjectFile loaded, but missing expected keys: {sorted(missing)}")
    print(f"[OK] Loaded mug ObjectFile: {MUG_OBJECT_FILE}")


def check_unitouch_checkpoint_load() -> None:
    if not UNITOUCH_CKPT.exists():
        raise FileNotFoundError(f"Missing UniTouch checkpoint: {UNITOUCH_CKPT}")
    _ = torch.load(UNITOUCH_CKPT, map_location="cpu", weights_only=False)
    print(f"[OK] Loaded UniTouch checkpoint: {UNITOUCH_CKPT}")


def main() -> None:
    check_maniskill_import()
    check_objectfolder_import()
    check_mug_objectfile_load()
    check_unitouch_checkpoint_load()
    print("All requested components loaded without error.")


if __name__ == "__main__":
    main()
