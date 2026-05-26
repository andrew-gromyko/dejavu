#!/usr/bin/env python3
"""Run the mug contact probe and print contact info."""

from __future__ import annotations

import argparse
import importlib
import os
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import imageio.v3 as iio
import numpy as np

from custom_envs.mug_contact_probe.dashboard import DashboardPayload, DashboardPresenter
from custom_envs.mug_contact_probe.objectfolder_worker import objectfolder_worker

if TYPE_CHECKING:
    from custom_envs.mug_contact_probe.env import MugContactProbeEnv
    from mani_skill.envs.sapien_env import BaseEnv


def _ensure_pinocchio_available() -> None:
    """Fail early if SAPIEN cannot use the pip-installed Pinocchio bindings."""
    py_mm = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.prefix) / "lib" / py_mm / "site-packages" / "cmeel.prefix" / "lib",
        # Backward-compatible fallback for existing local setups.
        Path(__file__).resolve().parents[2] / ".venv-ms3-of2" / "lib" / py_mm / "site-packages" / "cmeel.prefix" / "lib",
    ]
    cmeel_lib = next((p for p in candidates if p.exists()), None)
    if cmeel_lib is not None:
        cmeel_lib_str = str(cmeel_lib)
        old_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
        paths = [p for p in old_dyld.split(":") if p]
        if cmeel_lib_str not in paths:
            paths.insert(0, cmeel_lib_str)
            os.environ["DYLD_LIBRARY_PATH"] = ":".join(paths)
            try:
                os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
            except Exception as e:
                print(f"Failed to auto-restart python with DYLD_LIBRARY_PATH: {e}", file=sys.stderr)

    try:
        import pinocchio  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Pinocchio failed to import. Reinstall it inside this venv with:\n"
            "  python -m pip install --force-reinstall --no-cache-dir pin libpinocchio"
        ) from exc

    pm = importlib.import_module("sapien.wrapper.pinocchio_model")
    if getattr(pm, "PinocchioModel", None) is None:
        pm = importlib.reload(pm)
    if getattr(pm, "PinocchioModel", None) is None:
        raise RuntimeError(
            "SAPIEN still sees PinocchioModel=None after importing pinocchio. "
            "Restart the shell, re-activate .venv-ms3-of2, then rerun."
        )


@dataclass
class ContactResult:
    world_point: np.ndarray
    local_point: np.ndarray
    force_world: np.ndarray


PAD_IDS = ("left", "right")


def _to_action(env: "BaseEnv", arm_command: np.ndarray, gripper: float):
    from mani_skill.utils import common

    action_dict = {
        "arm": arm_command.astype(np.float32),
        "gripper": np.array([gripper], dtype=np.float32),
    }
    return env.unwrapped.agent.controller.from_action_dict(
        common.to_tensor(action_dict, device=env.unwrapped.device)
    )


def _tcp_world(ue: "MugContactProbeEnv") -> np.ndarray:
    return ue.agent.tcp.pose.p[0].detach().cpu().numpy().astype(np.float64)


def _grasp_center_world(ue: "MugContactProbeEnv") -> np.ndarray:
    # YCB mugs are not centered at actor origin; compute center from mesh bounds.
    mesh = ue.mug.get_first_collision_mesh()
    bounds = np.asarray(mesh.bounding_box.bounds, dtype=np.float64)
    local_center = 0.5 * (bounds[0] + bounds[1])
    t_world_mug = ue.mug.pose[0].sp.to_transformation_matrix()
    center_h = t_world_mug @ np.array([local_center[0], local_center[1], local_center[2], 1.0], dtype=np.float64)
    return center_h[:3]


def _arm_delta_to_action(ue: "MugContactProbeEnv", ee_delta_world: np.ndarray) -> np.ndarray:
    arm_controller = ue.agent.controller.controllers["arm"]
    ee_delta_world = ee_delta_world.astype(np.float32)

    if bool(getattr(arm_controller.config, "normalize_action", False)):
        # pd_ee_delta_pos expects normalized actions in [-1, 1].
        raw_upper = arm_controller.action_space_high.detach().cpu().numpy().astype(np.float32)
        raw_upper = np.where(np.abs(raw_upper) < 1e-6, 1.0, raw_upper)
        return np.clip(ee_delta_world / raw_upper, -1.0, 1.0)

    raw_lower = arm_controller.action_space_low.detach().cpu().numpy().astype(np.float32)
    raw_upper = arm_controller.action_space_high.detach().cpu().numpy().astype(np.float32)
    return np.clip(ee_delta_world, raw_lower, raw_upper)


def _extract_mug_contacts(env: "MugContactProbeEnv", min_force_n: float = 0.05) -> dict[str, ContactResult]:
    contacts = env.scene.get_contacts()

    mug_body = env.mug._bodies[0]
    finger_body_to_pad = {
        env.agent.finger1_link._bodies[0]: "left",
        env.agent.finger2_link._bodies[0]: "right",
    }
    dt = float(env.scene.px.timestep)

    best: dict[str, ContactResult] = {}
    best_force_norm = {pad_id: -np.inf for pad_id in PAD_IDS}

    for contact in contacts:
        b0, b1 = contact.bodies
        mug_first = b0 == mug_body and b1 in finger_body_to_pad
        mug_second = b1 == mug_body and b0 in finger_body_to_pad
        if not (mug_first or mug_second):
            continue
        pad_id = finger_body_to_pad[b1 if mug_first else b0]

        for pt in contact.points:
            impulse = np.asarray(pt.impulse, dtype=np.float64)
            if mug_second:
                impulse = -impulse
            force = impulse / dt
            force_norm = np.linalg.norm(force)
            if force_norm < min_force_n or force_norm <= best_force_norm[pad_id]:
                continue

            world_point = np.asarray(pt.position, dtype=np.float64)
            t_mug_inv = np.linalg.inv(env.mug.pose[0].sp.to_transformation_matrix())
            local_h = t_mug_inv @ np.array([world_point[0], world_point[1], world_point[2], 1.0])

            best[pad_id] = ContactResult(
                world_point=world_point,
                local_point=local_h[:3],
                force_world=np.asarray(force, dtype=np.float64),
            )
            best_force_norm[pad_id] = force_norm

    return best


def _pad_contact_force_norms(env: "MugContactProbeEnv") -> dict[str, float]:
    pad_links = {
        "left": env.agent.finger1_link,
        "right": env.agent.finger2_link,
    }
    force_norms = {}
    for pad_id, link in pad_links.items():
        pair_force = env.scene.get_pairwise_contact_forces(link, env.mug)
        force_norms[pad_id] = float(np.linalg.norm(pair_force[0].detach().cpu().numpy()))
    return force_norms


def _press_depth_from_force(force_norm_n: float) -> float:
    return 0.0005 + (0.0020 - 0.0005) * min(force_norm_n / 4.0, 1.0)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image[..., :3])

    if np.issubdtype(rgb.dtype, np.floating):
        # SAPIEN float RGB textures are in [0, 1]. Convert each frame independently.
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.max(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb = rgb.copy()

    return np.ascontiguousarray(rgb)


def _render_video_rgb(ue: "MugContactProbeEnv", settle_frames: int = 2) -> np.ndarray:
    frame = None
    for _ in range(max(1, settle_frames)):
        frame = ue.render_rgb_array(camera_name="render_camera")
    if frame is None:
        raise RuntimeError("render_camera did not produce an RGB frame")
    if hasattr(frame, "detach"):
        frame = frame[0].detach().cpu().numpy()
    elif frame.ndim == 4:
        frame = frame[0]
    return _to_uint8_rgb(frame)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scripted mug contact probe.")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        choices=["human", "rgb_array"],
        help="Use 'human' for live viewer or 'rgb_array' for headless.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=320,
        help="Maximum scripted control steps.",
    )
    parser.add_argument(
        "--objectfolder-object-id",
        type=str,
        default="assets/objectfolder2/025_mug_ObjectFile.pth",
        help="ObjectFolder object id or ObjectFile path.",
    )
    parser.add_argument(
        "--skip-objectfolder",
        action="store_true",
        help="Skip TouchNet/AudioNet query rendering.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the real-time dashboard window.",
    )
    parser.add_argument(
        "--dashboard-backend",
        type=str,
        default="browser",
        choices=["browser", "opencv"],
        help="Use a local browser stream or OpenCV HighGUI for the dashboard.",
    )
    parser.add_argument(
        "--render-settle-frames",
        type=int,
        default=2,
        help="Render the video camera this many times and display the last frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    _ensure_pinocchio_available()
    import gymnasium as gym
    import custom_envs  # noqa: F401 - registers custom envs
    from custom_envs.mug_contact_probe.env import MugContactProbeEnv
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    request_queue = multiprocessing.Queue()
    response_queue = multiprocessing.Queue()
    worker = None
    neutral_tactile = None

    if not args.skip_objectfolder:
        print("Starting ObjectFolder background worker process...")
        worker = multiprocessing.Process(
            target=objectfolder_worker,
            args=(request_queue, response_queue, args.objectfolder_object_id),
            daemon=True,
        )
        worker.start()

        # Request neutral tactile frame.
        request_queue.put(
            {
                "type": "tactile",
                "pad_id": "neutral",
                "local_point": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "press_depth": 0.0005,
            }
        )

        print("Waiting for background worker to load models and return neutral frame...")
        try:
            response = response_queue.get(timeout=30)
            if len(response) == 3:
                _, _, neutral_tactile = response
            else:
                _, neutral_tactile = response
            print("Background worker initialized successfully.")
        except Exception as e:
            print(f"Error: background worker failed to start within timeout: {e}", file=sys.stderr)
            args.skip_objectfolder = True

    env = gym.make(
        "MugContactProbe-v1",
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_ee_delta_pos",
        render_mode=args.render_mode,
        sim_backend="cpu",
    )
    ue: MugContactProbeEnv = env.unwrapped

    env.reset(seed=0)
    video_rgb = _render_video_rgb(ue, settle_frames=args.render_settle_frames)
    dashboard = DashboardPresenter(enabled=not args.no_dashboard, backend=args.dashboard_backend)
    dashboard.start()

    show_native_viewer = args.render_mode == "human" and not dashboard.enabled
    if args.render_mode == "human" and dashboard.enabled:
        print(
            "Dashboard enabled; skipping ManiSkill native env.render() to avoid "
            "double-rendering. Use --no-dashboard with --render-mode human for "
            "the native viewer."
        )
    if show_native_viewer:
        env.render()

    contact: Optional[ContactResult] = None
    contacts_by_pad: dict[str, Optional[ContactResult]] = {pad_id: None for pad_id in PAD_IDS}
    contact_rgb: Optional[np.ndarray] = None
    touch_step: Optional[int] = None
    mug_z_at_touch: Optional[float] = None
    mug_z_max = float(ue.mug.pose.p[0, 2].item())
    control_step = 0
    terminated_early = False

    tactile_rgbs: dict[str, Optional[np.ndarray]] = {pad_id: neutral_tactile for pad_id in PAD_IDS}
    tactile_request_pending = {pad_id: False for pad_id in PAD_IDS}
    audio_waveforms: dict[str, Optional[np.ndarray]] = {pad_id: None for pad_id in PAD_IDS}
    audio_play_start_times: dict[str, Optional[float]] = {pad_id: None for pad_id in PAD_IDS}
    was_contact_active = {pad_id: False for pad_id in PAD_IDS}

    def _drain_worker_responses() -> None:
        if worker is None:
            return
        try:
            while True:
                response = response_queue.get_nowait()
                if len(response) == 3:
                    resp_type, pad_id, resp_val = response
                else:
                    resp_type, resp_val = response
                    pad_id = "left"

                if resp_type == "tactile":
                    if pad_id in tactile_rgbs:
                        tactile_rgbs[pad_id] = resp_val
                        tactile_request_pending[pad_id] = False
                elif resp_type == "audio":
                    if pad_id in audio_waveforms:
                        audio_waveforms[pad_id] = resp_val
                        audio_play_start_times[pad_id] = time.time()
        except queue.Empty:
            pass

    def _make_dashboard_payload(active_contacts: dict[str, Optional[ContactResult]]) -> DashboardPayload:
        return DashboardPayload(
            video_rgb=video_rgb,
            control_step=control_step,
            max_steps=args.max_steps,
            tactile_rgbs=tactile_rgbs,
            audio_waveforms=audio_waveforms,
            audio_play_start_times=audio_play_start_times,
            contact_active={pad_id: active_contacts.get(pad_id) is not None for pad_id in PAD_IDS},
            force_norm_n={
                pad_id: float(np.linalg.norm(active_contacts[pad_id].force_world))
                if active_contacts.get(pad_id) is not None
                else 0.0
                for pad_id in PAD_IDS
            },
            is_grasping=bool(ue.agent.is_grasping(ue.mug)[0].item()),
        )

    def step_with_target(target: np.ndarray, gripper: float, max_delta_m: float) -> bool:
        nonlocal video_rgb, control_step, contact, contact_rgb, touch_step, mug_z_at_touch, mug_z_max
        nonlocal contacts_by_pad, audio_play_start_times

        if control_step >= args.max_steps:
            return False

        t_start = time.time()

        tcp = _tcp_world(ue)
        pos_err = target - tcp
        ee_delta = np.clip(pos_err, -max_delta_m, max_delta_m)
        arm_action = _arm_delta_to_action(ue, ee_delta)
        action = _to_action(env, arm_command=arm_action, gripper=gripper)

        _, _, terminated, truncated, _ = env.step(action)
        if show_native_viewer:
            env.render()
        control_step += 1

        mug_z_max = max(mug_z_max, float(ue.mug.pose.p[0, 2].item()))
        video_rgb = _render_video_rgb(ue, settle_frames=args.render_settle_frames)

        _drain_worker_responses()

        current_contacts = _extract_mug_contacts(ue, min_force_n=0.0)
        pad_force_norms = _pad_contact_force_norms(ue)
        active_contacts = {
            pad_id: current_contacts.get(pad_id)
            if current_contacts.get(pad_id) is not None and pad_force_norms[pad_id] > 1e-6
            else None
            for pad_id in PAD_IDS
        }
        contacts_by_pad = active_contacts

        if any(active_contacts.values()):
            if touch_step is None:
                touch_step = control_step
                mug_z_at_touch = float(ue.mug.pose.p[0, 2].item())
                contact = next(c for c in active_contacts.values() if c is not None)
                contact_rgb = video_rgb.copy()
                active_pad_names = ", ".join(pad_id for pad_id, pad_contact in active_contacts.items() if pad_contact is not None)
                print(f"Contact detected at step {control_step} on {active_pad_names}.")

            for pad_id, pad_contact in active_contacts.items():
                if pad_contact is None:
                    tactile_rgbs[pad_id] = neutral_tactile
                    tactile_request_pending[pad_id] = False
                    was_contact_active[pad_id] = False
                    continue

                force_norm = float(np.linalg.norm(pad_contact.force_world))
                press_depth = _press_depth_from_force(force_norm)

                if worker is not None and not was_contact_active[pad_id]:
                    print(f"{pad_id} pad contact onset at step {control_step}. Dispatching impact sound query...")
                    request_queue.put(
                        {
                            "type": "audio",
                            "pad_id": pad_id,
                            "local_point": pad_contact.local_point,
                            "press_depth": press_depth,
                        }
                    )

                if worker is not None and not tactile_request_pending[pad_id]:
                    request_queue.put(
                        {
                            "type": "tactile",
                            "pad_id": pad_id,
                            "local_point": pad_contact.local_point,
                            "press_depth": press_depth,
                        }
                    )
                    tactile_request_pending[pad_id] = True

                was_contact_active[pad_id] = True
        else:
            for pad_id in PAD_IDS:
                tactile_rgbs[pad_id] = neutral_tactile
                tactile_request_pending[pad_id] = False
                was_contact_active[pad_id] = False

        if dashboard.enabled:
            dashboard_payload = _make_dashboard_payload(active_contacts)
            keep_running, updated_audio_start = dashboard.render(dashboard_payload)
            if not keep_running:
                if worker is not None:
                    request_queue.put(None)
                    worker.join()
                dashboard.close()
                env.close()
                sys.exit(0)
            if updated_audio_start is not None:
                audio_play_start_times = updated_audio_start

        # Cap frame rate at 30 FPS for smooth rendering.
        t_elapsed = time.time() - t_start
        target_dt = 0.033
        if t_elapsed < target_dt:
            time.sleep(target_dt - t_elapsed)

        return not (bool(terminated[0]) or bool(truncated[0]))

    def move_to_target(
        target: np.ndarray,
        gripper: float,
        max_steps: int,
        pos_tol_m: float,
        max_delta_m: float,
        settle_steps: int = 0,
    ) -> bool:
        stable = 0
        for _ in range(max_steps):
            tcp = _tcp_world(ue)
            if np.linalg.norm(target - tcp) <= pos_tol_m:
                stable += 1
            else:
                stable = 0

            if stable >= max(1, settle_steps):
                return True

            if not step_with_target(target, gripper=gripper, max_delta_m=max_delta_m):
                return False
        return False

    # Let physics settle briefly before executing the scripted trajectory.
    hold_target = _tcp_world(ue)
    for _ in range(10):
        if not step_with_target(hold_target, gripper=1.0, max_delta_m=0.004):
            terminated_early = True
            break

    grasp_center = _grasp_center_world(ue)
    pregrasp_target = grasp_center + np.array([0.0, 0.0, 0.10], dtype=np.float64)
    grasp_target = grasp_center.copy()
    lift_target = grasp_center + np.array([0.0, 0.0, 0.14], dtype=np.float64)

    if not terminated_early:
        phase_ok = move_to_target(
            pregrasp_target, gripper=1.0, max_steps=90, pos_tol_m=0.003, max_delta_m=0.02, settle_steps=6
        )
        if not phase_ok:
            terminated_early = True

    if not terminated_early:
        phase_ok = move_to_target(
            grasp_target, gripper=1.0, max_steps=90, pos_tol_m=0.0015, max_delta_m=0.009, settle_steps=8
        )
        if not phase_ok:
            terminated_early = True

    if not terminated_early:
        for _ in range(16):
            if not step_with_target(grasp_target, gripper=1.0, max_delta_m=0.004):
                terminated_early = True
                break

    if not terminated_early:
        for _ in range(56):
            if not step_with_target(grasp_target, gripper=-1.0, max_delta_m=0.003):
                terminated_early = True
                break

    if not terminated_early:
        for _ in range(20):
            if not step_with_target(grasp_target, gripper=-1.0, max_delta_m=0.003):
                terminated_early = True
                break

    if not terminated_early:
        phase_ok = move_to_target(
            lift_target, gripper=-1.0, max_steps=110, pos_tol_m=0.003, max_delta_m=0.012, settle_steps=8
        )
        if not phase_ok:
            terminated_early = True

    if not terminated_early:
        for _ in range(30):
            if not step_with_target(lift_target, gripper=-1.0, max_delta_m=0.004):
                terminated_early = True
                break

    rgb_dir = root / "runs" / "mug_contact_probe"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    rgb_first_contact_path = rgb_dir / "probe_rgb_first_contact.png"
    rgb_final_path = rgb_dir / "probe_rgb_final.png"

    iio.imwrite(rgb_final_path, video_rgb)
    if contact_rgb is not None:
        iio.imwrite(rgb_first_contact_path, contact_rgb)

    if contact is None:
        print("No mug contact point found. Try adjusting approach targets.")
        print(f"Saved final camera RGB: {rgb_final_path}")
        dashboard.close()
        env.close()
        raise SystemExit(1)

    print("Contact point (world):", np.array2string(contact.world_point, precision=6))
    print("Contact point (mug local):", np.array2string(contact.local_point, precision=6))
    print("Contact force (world, N):", np.array2string(contact.force_world, precision=6))
    for pad_id in PAD_IDS:
        pad_contact = contacts_by_pad.get(pad_id)
        if pad_contact is None:
            print(f"{pad_id.capitalize()} pad final contact: none")
            continue
        print(f"{pad_id.capitalize()} pad contact point (world):", np.array2string(pad_contact.world_point, precision=6))
        print(f"{pad_id.capitalize()} pad contact point (mug local):", np.array2string(pad_contact.local_point, precision=6))
        print(f"{pad_id.capitalize()} pad contact force (world, N):", np.array2string(pad_contact.force_world, precision=6))

    mug_z_final = float(ue.mug.pose.p[0, 2].item())
    if mug_z_at_touch is not None:
        print(f"Mug z at touch: {mug_z_at_touch:.6f}")
        print(f"Mug z max after touch: {mug_z_max:.6f}")
        print(f"Mug z final: {mug_z_final:.6f}")
        print(f"Mug lift delta after touch: {mug_z_final - mug_z_at_touch:.6f} m")

    print(f"Grasping at end: {bool(ue.agent.is_grasping(ue.mug)[0].item())}")
    print(f"Saved first-contact RGB: {rgb_first_contact_path}")
    print(f"Saved final RGB: {rgb_final_path}")

    if dashboard.enabled:
        print("\nTrajectory complete. Press 'q' or ESC in the dashboard window to exit.")
        current_contacts = _extract_mug_contacts(ue, min_force_n=0.0)
        pad_force_norms = _pad_contact_force_norms(ue)
        contacts_by_pad = {
            pad_id: current_contacts.get(pad_id)
            if current_contacts.get(pad_id) is not None and pad_force_norms[pad_id] > 1e-6
            else None
            for pad_id in PAD_IDS
        }
        while True:
            _drain_worker_responses()

            dashboard_payload = _make_dashboard_payload(contacts_by_pad)
            keep_running, updated_audio_start = dashboard.render(dashboard_payload)
            if not keep_running:
                break
            if updated_audio_start is not None:
                audio_play_start_times = updated_audio_start
            time.sleep(0.03)

    if worker is not None:
        request_queue.put(None)
        worker.join()

    dashboard.close()
    env.close()


if __name__ == "__main__":
    main()
