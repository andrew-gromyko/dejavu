import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import custom_envs.mass_memory_bin_sort  # noqa: F401
import gymnasium as gym
import numpy as np
import tyro
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


@dataclass
class Args:
    mode: str = "video"
    """Launch mode: 'video' to save mp4, 'human' for live viewer."""

    obs_mode: str = "rgb+depth"
    """Observation mode passed to ManiSkill."""

    sim_backend: str = "cpu"
    """Simulation backend passed to ManiSkill demo launcher."""

    render_backend: str = "cpu"
    """Render backend passed to ManiSkill demo launcher."""

    record_dir: str = "runs/mass_memory_preview"
    """Where to save mp4 files when mode='video'."""

    seed: Optional[int] = 0
    """Seed for environment reset and random actions."""

    quiet: bool = True
    """Suppress per-step logs."""

    num_envs: int = 1
    """Number of parallel envs."""

    max_steps: int = 0
    """If >0, force-stop after this many steps; else run until done."""


def _normalize_bool_flags(argv: list[str]) -> list[str]:
    out = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--quiet" and i + 1 < len(argv):
            value = argv[i + 1].strip().lower()
            if value in {"true", "1", "yes", "y"}:
                out.append("--quiet")
                i += 2
                continue
            if value in {"false", "0", "no", "n"}:
                out.append("--no-quiet")
                i += 2
                continue
        out.append(token)
        i += 1
    return out


def main():
    args = tyro.cli(Args, args=_normalize_bool_flags(sys.argv[1:]))
    if args.mode not in ("video", "human"):
        raise ValueError("mode must be one of: 'video', 'human'")

    record_dir = None
    render_mode = "human" if args.mode == "human" else "rgb_array"
    obs_mode = args.obs_mode if args.mode == "video" else "state"
    if args.mode == "video":
        requested_dir = Path(args.record_dir)
        if requested_dir.is_absolute():
            resolved_record_dir = requested_dir
        else:
            resolved_record_dir = PROJECT_ROOT / requested_dir
        resolved_record_dir.mkdir(parents=True, exist_ok=True)
        record_dir = str(resolved_record_dir)
        print(f"[mass_memory_env] cwd={Path.cwd()}")
        print(f"[mass_memory_env] record_dir={resolved_record_dir.resolve()}")

    env: BaseEnv = gym.make(
        "MassMemoryBinSort-v1",
        obs_mode=obs_mode,
        render_mode=render_mode,
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        enable_shadow=True,
    )
    if record_dir is not None:
        env = RecordEpisode(
            env,
            output_dir=record_dir,
            info_on_video=False,
            save_trajectory=False,
            avoid_overwriting_video=True,
            max_steps_per_video=gym_utils.find_max_episode_steps_value(env),
        )

    if isinstance(args.seed, int):
        np.random.seed(args.seed)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if env.action_space is not None and args.seed is not None:
        env.action_space.seed(args.seed)

    step_count = 0
    while True:
        action = env.action_space.sample() if env.action_space is not None else None
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        if args.max_steps > 0 and step_count >= args.max_steps:
            break
        if (terminated | truncated).any():
            break
    env.close()

    if record_dir is not None:
        mp4_files = sorted(Path(record_dir).rglob("*.mp4"))
        if len(mp4_files) == 0:
            raise RuntimeError(
                f"No MP4 files were produced in {record_dir}. "
                "Run with --quiet false and check render backend."
            )
        valid_files = [p for p in mp4_files if p.is_file() and p.stat().st_size > 0]
        if len(valid_files) == 0:
            raise RuntimeError(
                f"MP4 paths were found but empty/non-files in {Path(record_dir).resolve()}."
            )
        for p in valid_files:
            print(f"[mass_memory_env] video={p.resolve()} size={p.stat().st_size}")
        print(f"Saved preview video(s) in: {Path(record_dir).resolve()}")


if __name__ == "__main__":
    main()
