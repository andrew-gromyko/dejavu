import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import custom_envs.mass_memory_bin_sort  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import tyro

from mani_skill.utils.structs.pose import Pose


@dataclass
class Args:
    seeds: int = 3
    """Number of reset seeds to validate."""

    num_objects: int = 6
    """Objects per validation episode."""

    num_ambiguous: int = 2
    """Ambiguous near-threshold objects per validation episode."""

    settle_steps: int = 60
    """Zero-action steps used for spawn/bin stability checks."""

    random_steps: int = 50
    """Random-action runtime health steps per seed."""

    output: str = "runs/mass_memory_validation/summary.json"
    """Where to write the validation summary."""

    quiet: bool = False
    """Suppress per-check stdout."""


def _to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _finite_tree(x):
    if isinstance(x, dict):
        return all(_finite_tree(v) for v in x.values())
    if isinstance(x, (tuple, list)):
        return all(_finite_tree(v) for v in x)
    arr = _to_np(x)
    return np.isfinite(arr).all()


def _zero_action(env):
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def _step_zero(env, steps):
    obs = None
    info = None
    for _ in range(steps):
        obs, _, _, _, info = env.step(_zero_action(env))
    return obs, info


def _assert(cond, msg, failures):
    if not bool(cond):
        failures.append(msg)


def _object_positions(base):
    return np.stack([_to_np(obj.pose.p)[0] for obj in base.objects], axis=0)


def _place_object(base, obj_idx, xyz):
    p = torch.tensor(np.asarray(xyz, dtype=np.float32)[None, :], device=base.device)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=base.device)
    base.objects[obj_idx].set_pose(Pose.create_from_pq(p=p, q=q))
    if hasattr(base.objects[obj_idx], "set_linear_velocity"):
        base.objects[obj_idx].set_linear_velocity(torch.zeros((1, 3), device=base.device))
    if hasattr(base.objects[obj_idx], "set_angular_velocity"):
        base.objects[obj_idx].set_angular_velocity(torch.zeros((1, 3), device=base.device))


def _assigned_bin(base, obj_idx):
    is_heavy = bool(base.object_is_heavy[0, obj_idx].detach().cpu())
    light_bin = int(base.light_bin_index[0].detach().cpu())
    return 1 - light_bin if is_heavy else light_bin


def _validate_seed(seed: int, args: Args):
    failures = []
    metrics = {"seed": seed}
    env = gym.make(
        "MassMemoryBinSort-v1",
        obs_mode="state",
        render_mode=None,
        sim_backend="cpu",
        render_backend="cpu",
        num_envs=1,
        num_objects=args.num_objects,
        num_ambiguous=args.num_ambiguous,
    )
    try:
        obs, info = env.reset(seed=seed, options=dict(reconfigure=True))
        base = env.unwrapped
        _assert(_finite_tree(obs), "reset observation contains non-finite values", failures)
        _assert(_finite_tree(info), "reset info contains non-finite values", failures)

        masses = _to_np(base.object_masses)[0]
        actor_masses = np.asarray([float(_to_np(obj.mass)[0]) for obj in base.objects])
        metrics["masses_kg"] = masses.round(6).tolist()
        metrics["actor_masses_kg"] = actor_masses.round(6).tolist()
        _assert(np.allclose(masses, actor_masses, rtol=1e-5, atol=1e-6), "actor masses do not match env mass metadata", failures)
        _assert(np.all(masses > 0), "non-positive object mass", failures)
        _assert(int(_to_np(base.object_is_ambiguous).sum()) == args.num_ambiguous, "ambiguous object count mismatch", failures)

        p0 = _object_positions(base)
        obs, info = _step_zero(env, args.settle_steps)
        p1 = _object_positions(base)
        xy_drift = np.linalg.norm(p1[:, :2] - p0[:, :2], axis=1)
        metrics["spawn_max_xy_drift_m"] = float(xy_drift.max())
        metrics["spawn_all_static"] = bool(np.all([bool(obj.is_static()[0].detach().cpu()) for obj in base.objects]))
        _assert(_finite_tree(obs) and _finite_tree(info), "settle step produced non-finite values", failures)
        _assert(float(info["sorted_count"][0].detach().cpu()) == 0.0, "objects completed without robot interaction", failures)
        _assert(metrics["spawn_max_xy_drift_m"] < 0.04, f"spawn drift too large: {metrics['spawn_max_xy_drift_m']:.4f}m", failures)

        # Diagnostic-only placement: validates bins and completion logic independently of robot control.
        bin_failures = []
        for obj_idx in range(args.num_objects):
            assigned = _assigned_bin(base, obj_idx)
            center = _to_np(base.bin_centers)[assigned]
            z = float(base.bin_wall_height + 0.16)
            _place_object(base, obj_idx, [float(center[0]), float(center[1]), z])
            _step_zero(env, args.settle_steps)
            pos = _to_np(base.objects[obj_idx].pose.p)[0]
            delta_xy = pos[:2] - center[:2]
            inside = (
                abs(float(delta_xy[0])) <= float(base.bin_inner_half_xy[0].detach().cpu())
                and abs(float(delta_xy[1])) <= float(base.bin_inner_half_xy[1].detach().cpu())
            )
            if not inside:
                bin_failures.append({"object_index": obj_idx, "assigned_bin": assigned, "final_xy_delta": delta_xy.round(4).tolist()})

            base.object_hold_done[0, obj_idx] = True
            base.current_object_idx[0] = obj_idx
            info2 = base.evaluate()
            completed = bool(info2["obj_completed"][0, obj_idx].detach().cpu())
            if not completed:
                bin_failures.append({"object_index": obj_idx, "assigned_bin": assigned, "reason": "completion_logic_false_after_settled_bin_drop"})

        metrics["bin_failures"] = bin_failures
        _assert(len(bin_failures) == 0, f"bin containment/completion failures: {bin_failures}", failures)

        obs, info = env.reset(seed=seed + 10_000, options=dict(reconfigure=True))
        env.action_space.seed(seed)
        for step in range(args.random_steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            _assert(_finite_tree(obs), f"random step {step} observation contains non-finite values", failures)
            _assert(np.isfinite(_to_np(reward)).all(), f"random step {step} reward contains non-finite values", failures)
            _assert(_finite_tree(info), f"random step {step} info contains non-finite values", failures)
            if len(failures) > 0:
                break
        metrics["random_steps_checked"] = args.random_steps
    finally:
        env.close()
    metrics["passed"] = len(failures) == 0
    metrics["failures"] = failures
    return metrics


def main():
    args = tyro.cli(Args)
    summary = {"args": asdict(args), "seeds": []}
    for seed in range(args.seeds):
        result = _validate_seed(seed, args)
        summary["seeds"].append(result)
        if not args.quiet:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{status}] seed={seed} failures={len(result['failures'])}")
            for failure in result["failures"]:
                print(f"  - {failure}")
    summary["passed"] = all(s["passed"] for s in summary["seeds"])
    out = Path(args.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not args.quiet:
        print(f"Wrote validation summary: {out.resolve()}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
