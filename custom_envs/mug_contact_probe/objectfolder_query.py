"""Minimal ObjectFolder query bridge for touch and audio rendering."""

from __future__ import annotations

import collections.abc
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


OBJECTFOLDER_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "ObjectFolder"
DEFAULT_OBJECT_FILE = OBJECTFOLDER_ROOT / "demo" / "ObjectFile.pth"

TOUCH_THETA_MAX_RAD = float(np.radians(15.0))
TOUCH_DEPTH_MIN_M = 0.0005
TOUCH_DEPTH_MAX_M = 0.0020
TOUCH_MAP_DEPTH_MIN = 0.0339
TOUCH_MAP_DEPTH_MAX = 0.04
TOUCH_W = 120
TOUCH_H = 160
AUDIO_SR = 44_100
AUDIO_SECONDS = 3.0


@dataclass
class TouchSpec:
    """Object-local touch query for one contact point."""

    contact_xyz_local: np.ndarray
    orientation: np.ndarray
    press_depth: float
    force_xyz: Optional[np.ndarray] = None


@dataclass
class TouchModalityResult:
    tactile_rgb: np.ndarray
    audio_waveform: np.ndarray
    audio_sample_rate: int
    object_file_path: Path
    contact_xyz_local: np.ndarray
    force_xyz: np.ndarray
    orientation: np.ndarray
    press_depth: float


ObjectFolderQueryResult = TouchModalityResult


@dataclass
class _LoadedObjectFile:
    object_file_path: Path
    device: torch.device
    checkpoint: dict
    touch_model: torch.nn.Module
    touch_embed_fn: object
    audio_model: torch.nn.Module
    audio_embed_fn: object
    taxim: object
    wh_grid: np.ndarray


_OBJECTFILE_CACHE: Dict[Path, _LoadedObjectFile] = {}


def _patch_torch_six() -> None:
    """Provide compatibility for ObjectFolder code on new torch versions."""
    if "torch._six" in sys.modules:
        return
    module = types.ModuleType("torch._six")
    module.container_abcs = collections.abc
    module.string_classes = (str, bytes)
    module.int_classes = (int,)
    sys.modules["torch._six"] = module


def _import_objectfolder_modules():
    _patch_torch_six()
    root_str = str(OBJECTFOLDER_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    import AudioNet_model
    import AudioNet_utils
    import TouchNet_model
    import TouchNet_utils
    from taxim_render import TaximRender

    return AudioNet_model, AudioNet_utils, TouchNet_model, TouchNet_utils, TaximRender


def _resolve_object_file(object_id: Union[int, str, Path]) -> Path:
    if isinstance(object_id, Path):
        candidate = object_id
    else:
        candidate = Path(str(object_id))
    if not candidate.is_absolute():
        root = Path(__file__).resolve().parents[2]
        candidate_abs = root / candidate
        if candidate_abs.exists():
            candidate = candidate_abs
    if candidate.exists():
        if candidate.is_dir():
            nested = candidate / "ObjectFile.pth"
            if nested.exists():
                return nested.resolve()
        return candidate.resolve()

    # The repo currently only ships the demo ObjectFile.
    warnings.warn(
        f"ObjectFile for object_id={object_id!r} not found locally. "
        f"Falling back to demo ObjectFile: {DEFAULT_OBJECT_FILE}",
        RuntimeWarning,
        stacklevel=2,
    )
    return DEFAULT_OBJECT_FILE.resolve()


def _build_wh_grid() -> np.ndarray:
    w = np.repeat(np.arange(TOUCH_W, dtype=np.float32).reshape(TOUCH_W, 1), TOUCH_H, axis=1)
    h = np.repeat(np.arange(TOUCH_H, dtype=np.float32).reshape(1, TOUCH_H), TOUCH_W, axis=0)
    w = (w - float(w.min())) / (float(w.max()) - float(w.min()))
    h = (h - float(h.min())) / (float(h.max()) - float(h.min()))
    return np.stack([w.reshape(-1), h.reshape(-1)], axis=1).astype(np.float32)


def _load_object_file(object_file_path: Path) -> _LoadedObjectFile:
    cached = _OBJECTFILE_CACHE.get(object_file_path)
    if cached is not None:
        return cached

    AudioNet_model, AudioNet_utils, TouchNet_model, TouchNet_utils, TaximRender = _import_objectfolder_modules()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(object_file_path, map_location="cpu", weights_only=False)

    touch_embed_fn, touch_input_ch = TouchNet_model.get_embedder(10, 0)
    touch_model = TouchNet_model.NeRF(D=8, input_ch=touch_input_ch, output_ch=1)
    touch_state = TouchNet_utils.strip_prefix_if_present(checkpoint["TouchNet"]["model_state_dict"], "module.")
    touch_model.load_state_dict(touch_state)
    touch_model = touch_model.to(device).eval()

    g = int(np.asarray(checkpoint["AudioNet"]["frequencies"]).shape[0])
    audio_embed_fn, audio_input_ch = AudioNet_model.get_embedder(10, 0)
    audio_model = AudioNet_model.AudioNeRF(D=8, input_ch=audio_input_ch, output_ch=g)
    audio_state = AudioNet_utils.strip_prefix_if_present(checkpoint["AudioNet"]["model_state_dict"], "module.")
    audio_model.load_state_dict(audio_state)
    audio_model = audio_model.to(device).eval()

    loaded = _LoadedObjectFile(
        object_file_path=object_file_path,
        device=device,
        checkpoint=checkpoint,
        touch_model=touch_model,
        touch_embed_fn=touch_embed_fn,
        audio_model=audio_model,
        audio_embed_fn=audio_embed_fn,
        taxim=TaximRender(str(OBJECTFOLDER_ROOT / "calibs")),
        wh_grid=_build_wh_grid(),
    )
    _OBJECTFILE_CACHE[object_file_path] = loaded
    return loaded


def _force_from_orientation(orientation: np.ndarray, press_depth: float) -> np.ndarray:
    theta, phi = float(orientation[0]), float(orientation[1])
    # Keep a mostly normal (z+) strike direction and rotate azimuth by phi.
    vec = np.array(
        [
            np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta),
        ],
        dtype=np.float64,
    )
    vec = vec / max(1e-8, np.linalg.norm(vec))
    # Scale force by press depth around ObjectFolder's demo range.
    mag = float(np.clip(press_depth / 0.001, 0.5, 2.5))
    return (vec * mag).astype(np.float32)


def _render_touch(loaded: _LoadedObjectFile, contact_xyz_local: np.ndarray, orientation: np.ndarray, press_depth: float) -> np.ndarray:
    ckpt = loaded.checkpoint

    xyz_min = float(ckpt["TouchNet"]["xyz_min"])
    xyz_max = float(ckpt["TouchNet"]["xyz_max"])
    xyz_norm = (contact_xyz_local.astype(np.float32) - xyz_min) / (xyz_max - xyz_min)

    theta, phi = float(orientation[0]), float(orientation[1])
    theta_norm = theta / TOUCH_THETA_MAX_RAD
    phi_x = np.cos(phi)
    phi_y = np.sin(phi)
    depth_norm = (press_depth - TOUCH_DEPTH_MIN_M) / (TOUCH_DEPTH_MAX_M - TOUCH_DEPTH_MIN_M)

    n_pix = TOUCH_W * TOUCH_H
    data = np.zeros((n_pix, 9), dtype=np.float32)
    data[:, 0:3] = xyz_norm[None, :]
    data[:, 3] = theta_norm
    data[:, 4] = phi_x
    data[:, 5] = phi_y
    data[:, 6] = depth_norm
    data[:, 7:9] = loaded.wh_grid

    preds = []
    with torch.no_grad():
        for i in range(0, data.shape[0], 4096):
            batch = torch.from_numpy(data[i : i + 4096]).to(loaded.device)
            pred = loaded.touch_model(loaded.touch_embed_fn(batch))
            preds.append(pred.detach().cpu().numpy())
    depth_map = np.concatenate(preds, axis=0)
    depth_map = depth_map * (TOUCH_MAP_DEPTH_MAX - TOUCH_MAP_DEPTH_MIN) + TOUCH_MAP_DEPTH_MIN
    depth_map = depth_map.reshape(TOUCH_W, TOUCH_H)

    _, _, tactile_map = loaded.taxim.render(depth_map, press_depth)
    return np.clip(tactile_map, 0.0, 255.0).astype(np.uint8)


def _render_audio(
    loaded: _LoadedObjectFile,
    contact_xyz_local: np.ndarray,
    force_xyz: np.ndarray,
    n_samples: int = int(AUDIO_SR * AUDIO_SECONDS),
) -> np.ndarray:
    ckpt = loaded.checkpoint
    normalizer = ckpt["AudioNet"]["normalizer"]

    xyz_min = float(normalizer["xyz_min"])
    xyz_max = float(normalizer["xyz_max"])
    xyz_norm = (contact_xyz_local.astype(np.float32) - xyz_min) / (xyz_max - xyz_min)

    q = torch.from_numpy(xyz_norm.reshape(1, 3)).to(loaded.device)
    with torch.no_grad():
        embedded = loaded.audio_embed_fn(q)
        gain_x, gain_y, gain_z = loaded.audio_model(embedded, embedded, embedded)

    gain_x = gain_x[0] * (float(normalizer["f1_max"]) - float(normalizer["f1_min"])) + float(normalizer["f1_min"])
    gain_y = gain_y[0] * (float(normalizer["f2_max"]) - float(normalizer["f2_min"])) + float(normalizer["f2_min"])
    gain_z = gain_z[0] * (float(normalizer["f3_max"]) - float(normalizer["f3_min"])) + float(normalizer["f3_min"])

    gains = (
        float(force_xyz[0]) * gain_x
        + float(force_xyz[1]) * gain_y
        + float(force_xyz[2]) * gain_z
    )
    gains = gains.detach().cpu().numpy().astype(np.float64)
    freqs = np.asarray(ckpt["AudioNet"]["frequencies"], dtype=np.float64)
    damps = np.asarray(ckpt["AudioNet"]["dampings"], dtype=np.float64)

    t = np.arange(n_samples, dtype=np.float64) / float(AUDIO_SR)
    signal = np.zeros((n_samples,), dtype=np.float64)
    for gain, freq, damp in zip(gains, freqs, damps):
        signal += gain * np.exp(-abs(damp) * t) * np.sin(2.0 * np.pi * freq * t)

    max_abs = float(np.max(np.abs(signal)))
    if max_abs > 1e-8:
        signal = signal / max_abs
    return signal.astype(np.float32)


class ObjectFolderModalityRenderer:
    """Reusable ObjectFolder touch/audio renderer for one object instance."""

    def __init__(self, object_id: Union[int, str, Path]):
        self.object_file_path = _resolve_object_file(object_id)
        self._loaded = _load_object_file(self.object_file_path)

    def make_touch_spec(
        self,
        contact_xyz_local: np.ndarray,
        orientation: Tuple[float, float] | np.ndarray = (float(np.radians(5.0)), 0.0),
        press_depth: float = TOUCH_DEPTH_MIN_M,
        force_xyz: Optional[np.ndarray] = None,
    ) -> TouchSpec:
        """Normalize the raw touch description used by all modality render calls."""
        local = np.asarray(contact_xyz_local, dtype=np.float32).reshape(3)
        orient = np.asarray(orientation, dtype=np.float32).reshape(2)
        depth = float(np.clip(press_depth, TOUCH_DEPTH_MIN_M, TOUCH_DEPTH_MAX_M))
        force = None if force_xyz is None else np.asarray(force_xyz, dtype=np.float32).reshape(3)
        return TouchSpec(contact_xyz_local=local, orientation=orient, press_depth=depth, force_xyz=force)

    def render_tactile(self, touch: TouchSpec) -> np.ndarray:
        """Return a uint8 RGB GelSight-style tactile image with shape (120, 160, 3)."""
        return _render_touch(self._loaded, touch.contact_xyz_local, touch.orientation, touch.press_depth)

    def render_audio(self, touch: TouchSpec) -> np.ndarray:
        """Return a mono float32 impact waveform sampled at AUDIO_SR."""
        force_xyz = touch.force_xyz
        if force_xyz is None:
            force_xyz = _force_from_orientation(touch.orientation, touch.press_depth)
        return _render_audio(self._loaded, touch.contact_xyz_local, force_xyz)

    def render_touch(self, touch: TouchSpec) -> TouchModalityResult:
        """Render tactile and audio outputs for one object-local contact point."""
        force_xyz = touch.force_xyz
        if force_xyz is None:
            force_xyz = _force_from_orientation(touch.orientation, touch.press_depth)
        tactile = self.render_tactile(touch)
        audio = _render_audio(self._loaded, touch.contact_xyz_local, force_xyz)
        return TouchModalityResult(
            tactile_rgb=tactile,
            audio_waveform=audio,
            audio_sample_rate=AUDIO_SR,
            object_file_path=self.object_file_path,
            contact_xyz_local=touch.contact_xyz_local,
            force_xyz=force_xyz,
            orientation=touch.orientation,
            press_depth=touch.press_depth,
        )

    def query(
        self,
        contact_xyz_local: np.ndarray,
        orientation: Tuple[float, float] | np.ndarray = (float(np.radians(5.0)), 0.0),
        press_depth: float = TOUCH_DEPTH_MIN_M,
        force_xyz: Optional[np.ndarray] = None,
    ) -> TouchModalityResult:
        """Convenience wrapper: raw touch parameters in, tactile/audio modalities out."""
        return self.render_touch(
            self.make_touch_spec(
                contact_xyz_local=contact_xyz_local,
                orientation=orientation,
                press_depth=press_depth,
                force_xyz=force_xyz,
            )
        )


def query_objectfolder(
    object_id: Union[int, str, Path],
    contact_xyz_local: np.ndarray,
    orientation: Tuple[float, float],
    press_depth: float,
) -> TouchModalityResult:
    """Render one tactile image and one impact sound from an ObjectFolder ObjectFile.

    Args:
        object_id: Object identifier or ObjectFile path. If unavailable, falls back to demo ObjectFile.
        contact_xyz_local: Contact point in object local coordinates [x, y, z].
        orientation: (theta, phi) in radians for touch/impact orientation.
        press_depth: Gel pressing depth in meters.
    """
    return ObjectFolderModalityRenderer(object_id).query(
        contact_xyz_local=contact_xyz_local,
        orientation=orientation,
        press_depth=press_depth,
    )
