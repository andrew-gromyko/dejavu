import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import custom_envs.sort_ycb_into_bins  # noqa: F401
from mani_skill.examples.demo_random_action import Args as DemoArgs
from mani_skill.examples.demo_random_action import main as demo_main
import tyro


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

    record_dir: str = "runs/sort_env_preview"
    """Where to save mp4 files when mode='video'."""

    seed: Optional[int] = 0
    """Seed for environment reset and random actions."""

    quiet: bool = True
    """Suppress per-step logs."""


def _normalize_bool_flags(argv: list[str]) -> list[str]:
    """Allow '--quiet false' in addition to Tyro's '--no-quiet' syntax."""
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
        Path(args.record_dir).mkdir(parents=True, exist_ok=True)
        record_dir = args.record_dir

    demo_main(
        DemoArgs(
            env_id="SortYCBIntoBins-v1",
            obs_mode=obs_mode,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            render_mode=render_mode,
            record_dir=record_dir,
            quiet=args.quiet,
            seed=args.seed,
        )
    )

    if record_dir is not None:
        print(f"Saved preview video(s) in: {record_dir}")


if __name__ == "__main__":
    main()
