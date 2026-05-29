#!/usr/bin/env python3
"""View a DejaVu memory benchmark episode in the ManiSkill simulator."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from custom_envs.mug_contact_probe.dejavu_ms3_env import DejaVuMemoryEnv
    from mani_skill.envs.sapien_env import BaseEnv


def _ensure_pinocchio_available(root: Path) -> None:
    """Force the venv cmeel libraries ahead of Homebrew before SAPIEN imports."""

    py_mm = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.prefix) / "lib" / py_mm / "site-packages" / "cmeel.prefix" / "lib",
        root / ".venv-ms3-of2" / "lib" / py_mm / "site-packages" / "cmeel.prefix" / "lib",
    ]
    cmeel_lib = next((p for p in candidates if p.exists()), None)
    if cmeel_lib is not None:
        cmeel_lib_s = str(cmeel_lib)
        old_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
        paths = [p for p in old_dyld.split(":") if p]
        desired = [cmeel_lib_s, *[p for p in paths if p != cmeel_lib_s]]
        if paths[:1] != [cmeel_lib_s] and os.environ.get("DEJAVU_DYLD_REEXEC") != "1":
            env = os.environ.copy()
            env["DYLD_LIBRARY_PATH"] = ":".join(desired)
            env["DEJAVU_DYLD_REEXEC"] = "1"
            os.execve(sys.executable, [sys.executable, *sys.argv], env)
        os.environ["DYLD_LIBRARY_PATH"] = ":".join(desired)

    try:
        import pinocchio  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Pinocchio failed to import from the venv. Check that DYLD_LIBRARY_PATH "
            "starts with the venv cmeel lib directory, or reinstall inside this venv:\n"
            "  .venv-ms3-of2/bin/python -m pip install --force-reinstall --no-cache-dir pin libpinocchio"
        ) from exc


def _to_action(env: "BaseEnv", arm_command: np.ndarray, gripper: float):
    from mani_skill.utils import common

    action_dict = {
        "arm": arm_command.astype(np.float32),
        "gripper": np.array([gripper], dtype=np.float32),
    }
    return env.unwrapped.agent.controller.from_action_dict(
        common.to_tensor(action_dict, device=env.unwrapped.device)
    )


def _tcp_world(ue: "DejaVuMemoryEnv") -> np.ndarray:
    return ue.agent.tcp.pose.p[0].detach().cpu().numpy().astype(np.float64)


def _object_z(ue: "DejaVuMemoryEnv", encounter_index: int) -> float:
    return float(ue.objects[encounter_index].pose.p[0, 2].detach().cpu().item())


def _arm_delta_to_action(ue: "DejaVuMemoryEnv", ee_delta_world: np.ndarray) -> np.ndarray:
    arm_controller = ue.agent.controller.controllers["arm"]
    ee_delta_world = ee_delta_world.astype(np.float32)

    if bool(getattr(arm_controller.config, "normalize_action", False)):
        raw_upper = arm_controller.action_space_high.detach().cpu().numpy().astype(np.float32)
        raw_upper = np.where(np.abs(raw_upper) < 1e-6, 1.0, raw_upper)
        return np.clip(ee_delta_world / raw_upper, -1.0, 1.0)

    raw_lower = arm_controller.action_space_low.detach().cpu().numpy().astype(np.float32)
    raw_upper = arm_controller.action_space_high.detach().cpu().numpy().astype(np.float32)
    return np.clip(ee_delta_world, raw_lower, raw_upper)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a DejaVu memory episode in ManiSkill.")
    parser.add_argument("--task", default="mass", choices=["mass", "material", "hardness", "texture"])
    parser.add_argument("--H", type=int, default=3, help="Number of encounters between target and query.")
    parser.add_argument("--D", type=int, default=1, help="Number of intervening distractor encounters.")
    parser.add_argument("--seed", type=int, default=0, help="Benchmark generation seed.")
    parser.add_argument("--render-mode", default="human", choices=["human", "rgb_array"])
    parser.add_argument("--max-delta", type=float, default=0.018, help="Max TCP delta per control step.")
    parser.add_argument("--step-sleep", type=float, default=0.0, help="Sleep after rendered frames for viewer pacing.")
    parser.add_argument("--render-every", type=int, default=3, help="Render every N control steps in human mode.")
    parser.add_argument("--hold-steps", type=int, default=75, help="Control steps to hold each lifted object.")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated episode without starting ManiSkill.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if args.dry_run:
        from custom_envs.mug_contact_probe.dejavu_env import DejaVuEnv

        env = DejaVuEnv(task=args.task, H=args.H, D=args.D, seed=args.seed)
        env.print_episode()
        return

    _ensure_pinocchio_available(root)

    import gymnasium as gym
    from custom_envs.mug_contact_probe.dejavu_ms3_env import DejaVuMemoryEnv  # noqa: F401 - registers env

    os.environ.setdefault("MS_ASSET_DIR", str(root / "data"))

    env = gym.make(
        "DejaVuMemory-v1",
        task=args.task,
        H=args.H,
        D=args.D,
        benchmark_seed=args.seed,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_ee_delta_pos",
        render_mode=args.render_mode,
        sim_backend="cpu",
    )
    ue: DejaVuMemoryEnv = env.unwrapped
    env.reset(seed=args.seed)
    control_step = 0

    if args.render_mode == "human":
        env.render()

    def step_toward(target: np.ndarray, gripper: float, max_delta: float) -> bool:
        nonlocal control_step

        tcp = _tcp_world(ue)
        delta = np.clip(target - tcp, -max_delta, max_delta)
        action = _to_action(env, _arm_delta_to_action(ue, delta), gripper=gripper)
        _, _, terminated, truncated, _ = env.step(action)
        control_step += 1
        if args.render_mode == "human" and control_step % max(1, args.render_every) == 0:
            env.render()
            if args.step_sleep > 0:
                time.sleep(args.step_sleep)
        return not (bool(terminated[0]) or bool(truncated[0]))

    def move_to(target: np.ndarray, gripper: float, max_steps: int, tol: float, max_delta: float | None = None) -> bool:
        step_delta = args.max_delta if max_delta is None else max_delta
        for _ in range(max_steps):
            if np.linalg.norm(target - _tcp_world(ue)) <= tol:
                return True
            if not step_toward(target, gripper=gripper, max_delta=step_delta):
                return False
        return False

    def hold_position(target: np.ndarray, gripper: float, steps: int) -> None:
        for _ in range(steps):
            if not step_toward(target, gripper=gripper, max_delta=0.002):
                break

    def probe_object(encounter_index: int, grasp: np.ndarray, safe: np.ndarray) -> float:
        move_to(grasp, gripper=1.0, max_steps=160, tol=0.006, max_delta=0.004)
        hold_position(grasp, gripper=1.0, steps=10)
        hold_position(grasp, gripper=-1.0, steps=45)
        z_before_lift = _object_z(ue, encounter_index)

        lift = grasp.copy()
        lift[2] = 0.20
        move_to(lift, gripper=-1.0, max_steps=180, tol=0.010, max_delta=0.006)
        hold_position(lift, gripper=-1.0, steps=args.hold_steps)
        z_after_lift = _object_z(ue, encounter_index)

        move_to(grasp, gripper=-1.0, max_steps=180, tol=0.008, max_delta=0.005)
        hold_position(grasp, gripper=1.0, steps=35)
        move_to(safe, gripper=1.0, max_steps=220, tol=0.012, max_delta=0.004)
        return z_after_lift - z_before_lift

    print("Generated benchmark episode:")
    ue.benchmark.print_episode()

    reset_high = _tcp_world(ue)
    reset_high[2] = 0.25
    move_to(reset_high, gripper=1.0, max_steps=160, tol=0.012, max_delta=0.006)

    while ue.active_step is not None:
        encounter_index = ue.active_step["encounter_index"]
        ue.print_current_step()
        obj_pos = ue.grasp_position(encounter_index).astype(np.float64)
        safe = obj_pos.copy()
        safe[2] = 0.24
        grasp = obj_pos + np.array([0.0, 0.0, 0.002], dtype=np.float64)

        if not move_to(safe, gripper=1.0, max_steps=320, tol=0.012, max_delta=0.006):
            print("Failed to reach high pregrasp; continuing to next encounter.")
        else:
            lift_delta = 0.0
            for attempt, offset in enumerate(
                (
                    np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    np.array([0.0, -0.008, 0.0], dtype=np.float64),
                    np.array([0.0, 0.008, 0.0], dtype=np.float64),
                    np.array([-0.008, 0.0, 0.0], dtype=np.float64),
                    np.array([0.008, 0.0, 0.0], dtype=np.float64),
                ),
                start=1,
            ):
                attempt_grasp = grasp + offset
                attempt_safe = attempt_grasp.copy()
                attempt_safe[2] = safe[2]
                move_to(attempt_safe, gripper=1.0, max_steps=120, tol=0.012, max_delta=0.006)
                lift_delta = probe_object(encounter_index, attempt_grasp, attempt_safe)
                if lift_delta >= 0.03:
                    break
                print(f"  retrying grasp after lift_delta_z={lift_delta:.3f} m")
            print(f"  lift_delta_z={lift_delta:.3f} m")

        more = ue.advance_encounter()
        if args.render_mode == "human":
            env.render()
            time.sleep(0.4)
        if not more:
            break

    ue.print_current_step()
    ue.benchmark.validate_episode()
    print("Assertions passed: H/D controls and answer are correct.")

    if args.render_mode == "human":
        print("Episode complete. Close the ManiSkill viewer window or press Ctrl-C to exit.")
        try:
            while True:
                env.render()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
    env.close()


if __name__ == "__main__":
    main()
