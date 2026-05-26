import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import custom_envs.mass_memory_bin_sort  # noqa: F401
import gymnasium as gym
import numpy as np
import sapien
import tyro


from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers import RecordEpisode

# Monkeypatching BaseMotionPlanningSolver before importing anything else
import mplib
from mplib.pymp import Pose as MPlibPose
from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import BaseMotionPlanningSolver
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import TwoFingerGripperMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb, compute_grasp_info_by_obb
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

FOLLOW_PATH_STEP_STRIDE = 1
FOLLOW_PATH_SETTLE_STEPS = 30

def patched_setup_planner(self):
    move_group = self.MOVE_GROUP if hasattr(self, "MOVE_GROUP") else "eef"
    link_names = [link.get_name() for link in self.robot.get_links()]
    joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
    planner = mplib.Planner(
        urdf=self.env_agent.urdf_path,
        srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group=move_group,
    )
    # Convert self.base_pose to mplib.pymp.Pose
    mplib_pose = MPlibPose(p=self.base_pose.p, q=self.base_pose.q)
    planner.set_base_pose(mplib_pose)
    
    planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
    planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
    return planner

def patched_move_to_pose_with_RRTConnect(self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0):
    pose = to_sapien_pose(pose)
    self._update_grasp_visual(pose)
    pose = self._transform_pose_for_planning(pose)
    
    mplib_pose = MPlibPose(p=pose.p, q=pose.q)
    result = self.planner.plan_pose(
        mplib_pose,
        self.robot.get_qpos().cpu().numpy()[0],
        time_step=self.base_env.control_timestep,
        wrt_world=True,
    )
    if result["status"] != "Success":
        print(f"RRTConnect planning failed: {result['status']}")
        self.render_wait()
        return -1
    self.render_wait()
    if dry_run:
        return result
    return self.follow_path(result, refine_steps=refine_steps)

def patched_move_to_pose_with_screw(self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0):
    pose = to_sapien_pose(pose)
    self._update_grasp_visual(pose)
    pose = self._transform_pose_for_planning(pose)
    
    mplib_pose = MPlibPose(p=pose.p, q=pose.q)
    result = self.planner.plan_screw(
        mplib_pose,
        self.robot.get_qpos().cpu().numpy()[0],
        time_step=self.base_env.control_timestep,
    )
    if result["status"] != "Success":
        # Fall back to RRTConnect
        result = self.planner.plan_pose(
            mplib_pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(f"Screw+RRTConnect fallback failed: {result['status']}")
            self.render_wait()
            return -1
    self.render_wait()
    if dry_run:
        return result
    return self.follow_path(result, refine_steps=refine_steps)

def patched_follow_path(self, result, refine_steps: int = 0):
    n_step = result["position"].shape[0]
    obs = None
    reward = 0.0
    terminated = False
    truncated = False
    info = {}
    arm_joints_len = len(self.planner.joint_vel_limits)
    for i in range(0, n_step + refine_steps, FOLLOW_PATH_STEP_STRIDE):
        qpos = result["position"][min(i, n_step - 1)]
        if self.control_mode == "pd_joint_pos_vel":
            qvel = result["velocity"][min(i, n_step - 1)]
            action = np.hstack([qpos, qvel, self.gripper_state])
        else:
            action = np.hstack([qpos, self.gripper_state])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        if self.print_env_info:
            print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
        if self.vis:
            self.base_env.render_human()
    # Settling loop: keep commanding the final joint target until tracking error < 0.01 rad
    target_qpos = result["position"][-1]
    for _ in range(FOLLOW_PATH_SETTLE_STEPS):
        actual_qpos = self.robot.get_qpos().cpu().numpy()[0][:arm_joints_len]
        if np.linalg.norm(actual_qpos - target_qpos[:arm_joints_len]) < 0.01:
            break
        if self.control_mode == "pd_joint_pos_vel":
            action = np.hstack([target_qpos, np.zeros_like(target_qpos), self.gripper_state])
        else:
            action = np.hstack([target_qpos, self.gripper_state])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        if self.vis:
            self.base_env.render_human()
    return obs, reward, terminated, truncated, info

TwoFingerGripperMotionPlanningSolver.follow_path = patched_follow_path
BaseMotionPlanningSolver.setup_planner = patched_setup_planner
BaseMotionPlanningSolver.move_to_pose_with_RRTConnect = patched_move_to_pose_with_RRTConnect
BaseMotionPlanningSolver.move_to_pose_with_screw = patched_move_to_pose_with_screw


@dataclass
class Args:
    episodes: int = 1
    """Number of oracle episodes to run."""

    seed: int = 0
    """Base reset/action seed."""

    output_dir: str = "runs/mass_memory_oracle"
    """Directory for compact oracle event logs."""

    record_video: bool = False
    """Also save a video of the scripted rollout."""

    obs_mode: str = "state"
    """Observation mode for environment creation."""

    sim_backend: str = "cpu"
    """ManiSkill simulation backend."""

    render_backend: str = "cpu"
    """ManiSkill render backend."""

    num_objects: int = 6
    """N objects per episode."""

    num_ambiguous: int = 2
    """D ambiguous near-threshold objects per episode."""

    lift_hold_steps: int = 40
    """Number of oracle steps to hold each object at lift height."""

    approach_height: float = 0.23
    """TCP height for moving above objects/bins."""

    place_height: float = 0.20
    """TCP height while carrying objects over bins."""

    release_height: float = 0.085
    """TCP height before opening the gripper over a bin."""

    bin_clearance_margin: float = 0.022
    """Extra vertical clearance above bin wall plus object half-height during transport."""

    initial_open_steps: int = 4
    """Steps for initial gripper-open command before object loop."""

    grasp_close_steps: int = 130
    """Steps for primary close-gripper command during grasp."""

    recovery_grasp_close_steps: int = 130
    """Steps for close-gripper command during grasp recovery."""

    grasp_settle_steps: int = 20
    """Additional hold-closed steps before checking grasp state."""

    release_open_steps: int = 40
    """Steps to open gripper for release over the bin."""

    release_settle_steps: int = 24
    """Open-gripper settle steps while staying at release pose before retreat."""

    post_release_open_steps: int = 40
    """Post-release open-gripper settle steps before next object."""

    completion_wait_steps: int = 24
    """Extra settle steps to allow object-complete to trigger before recovery/failure."""

    post_release_eval_wait_steps: int = 24
    """Extra settle steps after release/retreat before final completion check."""

    max_move_steps: int = 90
    """Maximum controller steps for a point-to-point TCP move."""

    pos_tolerance: float = 0.012
    """TCP position tolerance for scripted moves."""

    robot_uid: str = "panda"
    """Robot to use for oracle rollouts. Plain Panda avoids wrist-camera side-grasp collisions."""

    grasp_height_offset: float = 0.008
    """Meters above the object's settled center for the side-pinch TCP target."""

    gripper_force_limit: float = 340.0
    """Oracle gripper drive force limit. Higher force prevents slip while staying physical."""

    adaptive_gripper_force: bool = True
    """Scale gripper force by object mass to reduce launch failures on light objects."""

    adaptive_force_min: float = 240.0
    """Lower bound for adaptive gripper force."""

    adaptive_force_max: float = 360.0
    """Upper bound for adaptive gripper force."""

    grasp_depth: float = 0.012
    """Depth (m) from top object surface towards center for top-down grasp target."""

    pregrasp_offset: float = 0.08
    """Meters to stay above grasp pose before descending."""

    safe_hover_height: float = 0.26
    """Minimum Z for orientation alignment before descent to avoid side collisions."""

    quiet: bool = False
    """Suppress stdout event summaries."""

    video_width: int = 800
    """Recorded video width when --record-video is enabled."""

    video_height: int = 600
    """Recorded video height when --record-video is enabled."""

    video_fps: int = 24
    """Recorded video FPS when --record-video is enabled."""

    video_capture_stride: int = 4
    """Record one frame every N environment steps (native frame reduction)."""

    video_shader_pack: str = "default"
    """Shader pack for recorded camera render quality (e.g. minimal/default)."""

    enable_shadow: bool = False
    """Whether to enable shadows during rendering."""

    path_step_stride: int = 1
    """Execute one control action every N planner waypoints (reduces frames)."""

    settle_steps: int = 30
    """Maximum final-target settle steps after each planned move."""

    grasp_retry_count: int = 0
    """Per-attempt close-retry cycles after the first close."""

    grasp_retry_retract: float = 0.02
    """Meters to retract above grasp pose before retrying close."""

    grasp_retry_close_steps: int = 130
    """Close-gripper steps for retry grasp attempts."""

    grasp_retry_open_steps: int = 20
    """Open-gripper steps before a retry attempt."""

    object_pickup_retry_count: int = 2
    """Number of full pickup retries per object before aborting episode."""

    pre_pick_static_wait_steps: int = 14
    """Wait steps to let target object settle before pickup motions."""



class OracleLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")

    def event(self, **payload):
        self._f.write(json.dumps(payload, sort_keys=True) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


class SparseRecordEpisode(RecordEpisode):
    """Record videos while capturing one frame every N env steps (native sparse rendering)."""

    def __init__(self, *args, capture_stride: int = 1, **kwargs):
        self.capture_stride = max(1, int(capture_stride))
        super().__init__(*args, **kwargs)

    def step(self, action):
        # This script uses save_trajectory=False, so we can keep this fast sparse path.
        if self.save_trajectory:
            return super().step(action)

        capture_now = (self._elapsed_record_steps % self.capture_stride) == 0
        if self.save_video and capture_now and self._video_steps == 0:
            # First frame is state before applying this action.
            self.render_images.append(self.capture_image())

        obs, rew, terminated, truncated, info = self.env.step(action)

        if self.save_video and capture_now:
            self._video_steps += 1
            self.render_images.append(self.capture_image())
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()
        self._elapsed_record_steps += 1
        return obs, rew, terminated, truncated, info


def find_grasp_pose_candidates(
    planner_solver,
    obj,
    n_yaws: int = 24,
    grasp_depth: float = 0.012,
    pregrasp_offset: float = 0.08,
):
    """Search yaw angles for collision-free top-down grasp pose candidates.

    Uses the object's oriented bounding box to compute the grasp center, which
    properly accounts for object size and shape.

    Returns a list sorted by low joint-motion score:
    [(grasp_pose, pregrasp_pose, grasp_center, score), ...].
    """
    current_qpos = planner_solver.robot.get_qpos().cpu().numpy()[0]
    mplib_base = planner_solver.planner.robot.get_base_pose()
    base_pose_sapien = sapien.Pose(p=mplib_base.p, q=mplib_base.q)

    candidates = []

    # Panda finger length ~2.5cm — use OBB to compute correct grasp center
    GRASP_DEPTH = float(grasp_depth)
    approaching = np.array([0.0, 0.0, -1.0])
    target_closing = planner_solver.robot.get_links()[-1].pose.to_transformation_matrix()[:3, 1]
    preferred_yaw = None
    try:
        obb = get_actor_obb(obj)
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=GRASP_DEPTH,
        )
        obb_center = grasp_info["center"].copy()
        closing_hint = np.array(grasp_info["closing"], dtype=np.float64)
        closing_hint[2] = 0.0
        if np.linalg.norm(closing_hint) > 1e-6:
            closing_hint = closing_hint / np.linalg.norm(closing_hint)
            preferred_yaw = float(np.arctan2(closing_hint[1], closing_hint[0]))
    except Exception:
        # Fallback: object centroid
        obj_xyz = obj.pose.p[0].detach().cpu().numpy().astype(np.float64)
        obb_center = np.array([obj_xyz[0], obj_xyz[1], max(obj_xyz[2] + 0.008, 0.005)])

    def _wrap_pi(theta: float) -> float:
        return float(np.mod(theta, np.pi))

    def _circular_pi_diff(a: float, b: float) -> float:
        raw = (a - b + 0.5 * np.pi) % np.pi - 0.5 * np.pi
        return float(abs(raw))

    if preferred_yaw is None:
        yaws = np.linspace(0, np.pi, n_yaws, endpoint=False)
    else:
        span = np.deg2rad(85.0)
        raw_offsets = np.linspace(-span, span, n_yaws, endpoint=True)
        raw_offsets = raw_offsets[np.argsort(np.abs(raw_offsets))]
        yaws = []
        used = set()
        for off in raw_offsets:
            yaw = _wrap_pi(preferred_yaw + float(off))
            key = int(round(yaw * 10000))
            if key in used:
                continue
            used.add(key)
            yaws.append(yaw)
        yaws = np.array(yaws, dtype=np.float64)

    for yaw in yaws:
        closing = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = obb_center
        grasp_pose_world = sapien.Pose(T)
        pregrasp_pose_world = grasp_pose_world * sapien.Pose([0, 0, -float(pregrasp_offset)])

        # Transform world->base frame (what plan_pose does internally)
        pregrasp_base = base_pose_sapien.inv() * pregrasp_pose_world
        grasp_base = base_pose_sapien.inv() * grasp_pose_world

        # Test IK for pregrasp pose
        status_pre, q_goals_pre = planner_solver.planner.IK(
            MPlibPose(p=pregrasp_base.p, q=pregrasp_base.q),
            current_qpos,
            return_closest=True,
        )
        if status_pre != "Success" or q_goals_pre is None:
            continue

        # Test IK for grasp pose, starting from pregrasp joint pos
        status_g, q_goals_g = planner_solver.planner.IK(
            MPlibPose(p=grasp_base.p, q=grasp_base.q),
            q_goals_pre,
            return_closest=True,
        )
        if status_g != "Success" or q_goals_g is None:
            continue

        score = np.linalg.norm(q_goals_pre[:7] - current_qpos[:7])
        if preferred_yaw is not None:
            score += 0.15 * _circular_pi_diff(float(yaw), preferred_yaw)
        candidates.append((grasp_pose_world, pregrasp_pose_world, obb_center.copy(), score))

    candidates.sort(key=lambda x: x[3])
    return candidates


def find_best_grasp_pose(
    planner_solver,
    obj,
    n_yaws: int = 24,
    grasp_depth: float = 0.012,
    pregrasp_offset: float = 0.08,
):
    """Backwards-compatible best-candidate accessor."""
    candidates = find_grasp_pose_candidates(
        planner_solver,
        obj,
        n_yaws=n_yaws,
        grasp_depth=grasp_depth,
        pregrasp_offset=pregrasp_offset,
    )
    if len(candidates) == 0:
        return None, None, None
    grasp_pose, pregrasp_pose, grasp_center, _ = candidates[0]
    return grasp_pose, pregrasp_pose, grasp_center


def _object_identity(base_env, obj_idx):
    model_ids = getattr(base_env, "sampled_model_ids", None)
    if model_ids is None:
        return {"slot_index": obj_idx, "model_id": None}
    return {"slot_index": obj_idx, "model_id": model_ids[obj_idx][0]}


def _bin_assignment(base_env, obj_idx):
    mass = float(base_env.object_masses[0, obj_idx].detach().cpu())
    is_heavy = bool(base_env.object_is_heavy[0, obj_idx].detach().cpu())
    light_bin = int(base_env.light_bin_index[0].detach().cpu())
    assigned_bin = 1 - light_bin if is_heavy else light_bin
    return {
        "mass_kg": mass,
        "mass_threshold_kg": float(base_env.mass_threshold),
        "mass_class": "heavy" if is_heavy else "light",
        "assigned_bin_index": int(assigned_bin),
        "assigned_bin_role": "heavy" if is_heavy else "light",
        "is_ambiguous": bool(base_env.object_is_ambiguous[0, obj_idx].detach().cpu()),
    }


def _finger_force_reading(base_env, obj):
    l_force_vec = base_env.scene.get_pairwise_contact_forces(base_env.agent.finger1_link, obj)
    r_force_vec = base_env.scene.get_pairwise_contact_forces(base_env.agent.finger2_link, obj)
    l_force = float(np.linalg.norm(l_force_vec.detach().cpu().numpy()[0]))
    r_force = float(np.linalg.norm(r_force_vec.detach().cpu().numpy()[0]))
    return {
        "left_finger_force_n": round(l_force, 6),
        "right_finger_force_n": round(r_force, 6),
        "total_finger_force_n": round(l_force + r_force, 6),
        "source": "sim_pairwise_contact_forces",
    }


def _log_action(logger, episode_idx, planner, name, object_identity, fn):
    start = planner.elapsed_steps
    logger.event(
        event="action_start",
        action=name,
        episode_index=episode_idx,
        step_index=start,
        object=object_identity,
    )
    res = fn()
    success = (res != -1)
    logger.event(
        event="action_end",
        action=name,
        episode_index=episode_idx,
        step_index=planner.elapsed_steps,
        object=object_identity,
        controller_success=success,
    )
    return success


def _log_assignment_events(logger, episode_idx, step_idx, identity, assignment):
    logger.event(
        event="mass_ground_truth",
        episode_index=episode_idx,
        step_index=step_idx,
        object=identity,
        mass_kg=assignment["mass_kg"],
    )
    logger.event(
        event="bin_assignment",
        episode_index=episode_idx,
        step_index=step_idx,
        object=identity,
        assignment={
            k: v
            for k, v in assignment.items()
            if k
            in {
                "mass_class",
                "assigned_bin_index",
                "assigned_bin_role",
                "mass_threshold_kg",
                "is_ambiguous",
            }
        },
    )


def _configure_oracle_gripper(base_env, force_limit: float):
    for joint in base_env.agent.robot.get_active_joints():
        if "finger" in joint.get_name():
            joint.set_drive_properties(
                stiffness=float(base_env.agent.gripper_stiffness),
                damping=float(base_env.agent.gripper_damping),
                force_limit=float(force_limit),
            )


def _object_gripper_force(args: Args, assignment: dict) -> float:
    if not args.adaptive_gripper_force:
        return float(args.gripper_force_limit)
    mass = float(assignment["mass_kg"])
    # ~240N for very light objects up to ~360N for heavier ones.
    force = 220.0 + 350.0 * mass
    return float(np.clip(force, args.adaptive_force_min, args.adaptive_force_max))


def _object_grasp_policy(identity: dict, args: Args):
    model_id = str((identity or {}).get("model_id") or "")

    # Privileged-but-physical policy: tune grasp generation by known object type.
    if any(k in model_id for k in ("gelatin_box", "cracker_box", "sugar_box", "pudding_box")):
        return dict(
            n_yaws=56,
            grasp_depth=max(0.006, args.grasp_depth - 0.004),
            pregrasp_offset=max(0.09, args.pregrasp_offset),
            close_steps=args.grasp_close_steps + 10,
            retry_close_steps=args.grasp_retry_close_steps + 10,
            recovery_close_steps=args.recovery_grasp_close_steps + 10,
        )
    if any(k in model_id for k in ("can", "marker", "mug", "bowl")):
        return dict(
            n_yaws=56,
            grasp_depth=min(0.018, args.grasp_depth + 0.002),
            pregrasp_offset=args.pregrasp_offset,
            close_steps=args.grasp_close_steps,
            retry_close_steps=args.grasp_retry_close_steps,
            recovery_close_steps=args.recovery_grasp_close_steps,
        )
    if any(k in model_id for k in ("hammer", "wrench", "screwdriver", "scissors", "drill")):
        return dict(
            n_yaws=64,
            grasp_depth=max(0.007, args.grasp_depth - 0.002),
            pregrasp_offset=max(0.09, args.pregrasp_offset),
            close_steps=args.grasp_close_steps + 6,
            retry_close_steps=args.grasp_retry_close_steps + 6,
            recovery_close_steps=args.recovery_grasp_close_steps + 6,
        )
    return dict(
        n_yaws=40,
        grasp_depth=args.grasp_depth,
        pregrasp_offset=args.pregrasp_offset,
        close_steps=args.grasp_close_steps,
        retry_close_steps=args.grasp_retry_close_steps,
        recovery_close_steps=args.recovery_grasp_close_steps,
    )


def _is_grasping(base_env, obj) -> bool:
    return bool(base_env.agent.is_grasping(obj)[0].detach().cpu())


def _wait_for_completion(base_env, planner, obj_idx: int, wait_steps: int):
    info = base_env.evaluate()
    if bool(info["obj_completed"][0, obj_idx].detach().cpu()):
        return True, info
    for _ in range(max(0, int(wait_steps))):
        # One control step with gripper-open command to let physics settle.
        planner.open_gripper(1)
        info = base_env.evaluate()
        if bool(info["obj_completed"][0, obj_idx].detach().cpu()):
            return True, info
    return False, info


def _wait_object_static(base_env, planner, obj, wait_steps: int):
    for _ in range(max(0, int(wait_steps))):
        is_static = bool(obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)[0].detach().cpu())
        if is_static:
            return True
        planner.open_gripper(1)
    return bool(obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)[0].detach().cpu())


def _approach_align_descend(
    logger,
    episode_idx: int,
    planner,
    base_env,
    obj,
    identity,
    pregrasp_pose: sapien.Pose,
    grasp_pose: sapien.Pose,
    safe_hover_height: float,
    static_wait_steps: int,
    action_prefix: str = "",
):
    _wait_object_static(base_env, planner, obj=obj, wait_steps=static_wait_steps)
    tcp_q = base_env.agent.tcp.pose.q[0].detach().cpu().numpy().astype(np.float64)
    hover_z = max(float(pregrasp_pose.p[2]), float(safe_hover_height))
    pregrasp_pos_only = sapien.Pose(
        p=np.array([float(pregrasp_pose.p[0]), float(pregrasp_pose.p[1]), hover_z], dtype=np.float64),
        q=tcp_q,
    )
    hover_aligned_pose = sapien.Pose(
        p=np.array(pregrasp_pos_only.p, dtype=np.float64),
        q=np.array(pregrasp_pose.q, dtype=np.float64),
    )

    if not _log_action(
        logger,
        episode_idx,
        planner,
        f"{action_prefix}approach_object",
        identity,
        lambda: planner.move_to_pose_with_RRTConnect(pregrasp_pos_only),
    ):
        return False, "failed_to_reach_pregrasp_position"

    if not _log_action(
        logger,
        episode_idx,
        planner,
        f"{action_prefix}align_grasp_orientation",
        identity,
        lambda: planner.move_to_pose_with_screw(hover_aligned_pose),
    ):
        return False, "failed_to_align_grasp_orientation"

    if not _log_action(
        logger,
        episode_idx,
        planner,
        f"{action_prefix}lower_to_pregrasp",
        identity,
        lambda: planner.move_to_pose_with_screw(pregrasp_pose),
    ):
        return False, "failed_to_reach_pregrasp_pose"

    if not _log_action(
        logger,
        episode_idx,
        planner,
        f"{action_prefix}lower_to_grasp",
        identity,
        lambda: planner.move_to_pose_with_screw(grasp_pose),
    ):
        return False, "failed_to_reach_grasp"

    return True, None


def _transport_with_fallback(planner, target_pose: sapien.Pose):
    # Prefer screw interpolation for smoother carry motion; fall back to RRT if needed.
    result = planner.move_to_pose_with_screw(target_pose)
    if result == -1:
        return planner.move_to_pose_with_RRTConnect(target_pose)
    return result


def _recover_grasp_and_lift(
    logger,
    episode_idx: int,
    planner,
    base_env,
    obj,
    identity,
    assignment,
    args: Args,
):
    _configure_oracle_gripper(base_env, _object_gripper_force(args, assignment))
    policy = _object_grasp_policy(identity, args)
    grasp_candidates = find_grasp_pose_candidates(
        planner,
        obj,
        n_yaws=int(policy["n_yaws"]),
        grasp_depth=float(policy["grasp_depth"]),
        pregrasp_offset=float(policy["pregrasp_offset"]),
    )
    if len(grasp_candidates) == 0:
        logger.event(
            event="controller_warning",
            episode_index=episode_idx,
            step_index=planner.elapsed_steps,
            object=identity,
            warning="recovery_no_valid_grasp_orientation",
        )
        return False, None

    recovered = False
    active_grasp_pose = grasp_candidates[0][0]
    max_recovery_attempts = min(len(grasp_candidates), args.grasp_retry_count + 2)
    for attempt_idx in range(max_recovery_attempts):
        grasp_pose, pregrasp_pose, _, _ = grasp_candidates[attempt_idx]
        active_grasp_pose = grasp_pose
        suffix = "" if attempt_idx == 0 else f"_{attempt_idx}"

        recovery_ok, _ = _approach_align_descend(
            logger=logger,
            episode_idx=episode_idx,
            planner=planner,
            base_env=base_env,
            obj=obj,
            identity=identity,
            pregrasp_pose=pregrasp_pose,
            grasp_pose=grasp_pose,
            safe_hover_height=args.safe_hover_height,
            static_wait_steps=max(4, args.pre_pick_static_wait_steps // 2),
            action_prefix=f"recover{suffix}_",
        )
        if not recovery_ok:
            continue

        _log_action(
            logger,
            episode_idx,
            planner,
            f"recover_close_gripper{suffix}",
            identity,
            lambda close_steps=int(policy["recovery_close_steps"]): planner.close_gripper(close_steps),
        )
        if args.grasp_settle_steps > 0:
            _log_action(
                logger,
                episode_idx,
                planner,
                f"recover_hold_closed_gripper{suffix}",
                identity,
                lambda settle_steps=args.grasp_settle_steps: planner.close_gripper(settle_steps),
            )
        if _is_grasping(base_env, obj):
            recovered = True
            break

        logger.event(
            event="controller_warning",
            episode_index=episode_idx,
            step_index=planner.elapsed_steps,
            object=identity,
            warning="recovery_grasp_retry",
            retry_index=int(attempt_idx),
        )

    if not recovered:
        logger.event(
            event="controller_warning",
            episode_index=episode_idx,
            step_index=planner.elapsed_steps,
            object=identity,
            warning="recovery_grasp_not_detected_by_sim",
        )
        return False, None

    tcp_pos = base_env.agent.tcp.pose.p[0].detach().cpu().numpy().astype(np.float64)
    lift_pose = sapien.Pose(
        p=[tcp_pos[0], tcp_pos[1], args.place_height],
        q=active_grasp_pose.q,
    )
    if not _log_action(
        logger,
        episode_idx,
        planner,
        "recover_lift_object",
        identity,
        lambda: planner.move_to_pose_with_screw(lift_pose),
    ):
        return False, None

    _log_action(
        logger,
        episode_idx,
        planner,
        "recover_hold_lifted_object",
        identity,
        lambda: planner.close_gripper(max(20, args.lift_hold_steps // 2)),
    )
    return _is_grasping(base_env, obj), active_grasp_pose.q


def _object_half_height(base_env, obj_idx: int) -> float:
    raw_obj = base_env._raw_objects_by_slot[obj_idx][0]
    mesh = raw_obj.get_first_collision_mesh()
    z_bounds = mesh.bounding_box.bounds[:, 2]
    return float(0.5 * (z_bounds[1] - z_bounds[0]))


def run_episode(env, episode_idx: int, logger: OracleLogger, args: Args):
    base = env.unwrapped
    seed = args.seed + episode_idx
    env.reset(seed=seed, options=dict(reconfigure=True))
    _configure_oracle_gripper(base, args.gripper_force_limit)

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )

    # Load bin walls (not floor/ground) as collision objects in the planning world.
    # We skip the ground and bin floors so the gripper can reach thin objects at z≈0.
    # Physical simulation handles the actual ground collision.
    from mplib.sapien_utils.conversion import SapienPlanningWorld
    scene = base.scene.sub_scenes[0]
    _SKIP_KEYWORDS = ("ground", "floor")
    for actor in scene.get_all_actors():
        name = actor.get_name().lower()
        if any(kw in name for kw in _SKIP_KEYWORDS):
            continue
        comp = actor.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
        if comp is not None and isinstance(comp, sapien.physx.PhysxRigidStaticComponent):
            fcl_obj = SapienPlanningWorld.convert_physx_component(comp)
            if fcl_obj is not None:
                planner.planner.planning_world.add_object(fcl_obj)

    logger.event(
        event="episode_start",
        episode_index=episode_idx,
        step_index=0,
        seed=seed,
        num_objects=int(base.num_objects),
        num_ambiguous=int(base.num_ambiguous),
        mass_threshold_kg=float(base.mass_threshold),
        controller="pd_joint_pos",
        robot_uid=args.robot_uid,
        gripper_force_limit=float(args.gripper_force_limit),
    )

    # Start from a safe height with the gripper open.
    planner.open_gripper(args.initial_open_steps)

    for obj_idx, obj in enumerate(base.objects):
        sync_info = base.evaluate()
        active_idx = int(sync_info["current_object_idx"][0].detach().cpu())
        if active_idx < obj_idx:
            logger.event(
                event="controller_warning",
                episode_index=episode_idx,
                step_index=planner.elapsed_steps,
                warning="blocked_by_uncompleted_prior_object",
                blocked_object_index=active_idx,
                attempted_object_index=obj_idx,
            )
            break
        if active_idx > obj_idx:
            # Already completed by prior dynamics; skip redundant manipulation.
            continue

        identity = _object_identity(base, obj_idx)
        assignment = _bin_assignment(base, obj_idx)
        policy = _object_grasp_policy(identity, args)
        object_force_limit = _object_gripper_force(args, assignment)
        _configure_oracle_gripper(base, object_force_limit)
        logger.event(
            event="object_start",
            episode_index=episode_idx,
            step_index=planner.elapsed_steps,
            object=identity,
            mass_ground_truth_kg=assignment["mass_kg"],
            is_ambiguous=assignment["is_ambiguous"],
            object_gripper_force_limit_n=round(float(object_force_limit), 3),
            grasp_policy=policy,
        )

        grasp_pose = None
        pregrasp_pose = None
        grasped = False
        active_grasp_pose = None
        grasp_candidates = []

        for pickup_try in range(args.object_pickup_retry_count + 1):
            if pickup_try > 0:
                logger.event(
                    event="controller_warning",
                    episode_index=episode_idx,
                    step_index=planner.elapsed_steps,
                    object=identity,
                    warning="pickup_retrying_full_sequence",
                    retry_index=int(pickup_try),
                )
                planner.open_gripper(max(args.grasp_retry_open_steps, 20))

            grasp_candidates = find_grasp_pose_candidates(
                planner,
                obj,
                n_yaws=int(policy["n_yaws"]),
                grasp_depth=float(policy["grasp_depth"]),
                pregrasp_offset=float(policy["pregrasp_offset"]),
            )
            if len(grasp_candidates) == 0:
                continue

            # Keep retry orientation stable to avoid edge-on shove failures after reopen.
            candidate_base_i = 0
            grasp_pose, pregrasp_pose, _, _ = grasp_candidates[candidate_base_i]
            active_grasp_pose = grasp_pose
            action_suffix = "" if pickup_try == 0 else f"_pickup_retry_{pickup_try}"

            pickup_ok, _ = _approach_align_descend(
                logger=logger,
                episode_idx=episode_idx,
                planner=planner,
                base_env=base,
                obj=obj,
                identity=identity,
                pregrasp_pose=pregrasp_pose,
                grasp_pose=grasp_pose,
                safe_hover_height=args.safe_hover_height,
                static_wait_steps=args.pre_pick_static_wait_steps,
                action_prefix=f"{action_suffix[1:]}_" if action_suffix else "",
            )
            if not pickup_ok:
                continue

            for attempt_idx in range(args.grasp_retry_count + 1):
                close_steps = (
                    int(policy["close_steps"])
                    if attempt_idx == 0
                    else int(policy["retry_close_steps"])
                )
                if attempt_idx > 0:
                    candidate_i = min(candidate_base_i + attempt_idx, len(grasp_candidates) - 1)
                    retry_grasp_pose, retry_pregrasp_pose, _, _ = grasp_candidates[candidate_i]
                    active_grasp_pose = retry_grasp_pose
                    logger.event(
                        event="controller_warning",
                        episode_index=episode_idx,
                        step_index=planner.elapsed_steps,
                        object=identity,
                        warning="initial_grasp_not_detected_retrying",
                        retry_index=int(attempt_idx),
                        retry_candidate_index=int(candidate_i),
                    )
                    _log_action(
                        logger,
                        episode_idx,
                        planner,
                        f"retry_open_gripper_{attempt_idx}",
                        identity,
                        lambda: planner.open_gripper(args.grasp_retry_open_steps),
                    )
                    if not _log_action(
                        logger,
                        episode_idx,
                        planner,
                        f"retry_retract_from_grasp_{attempt_idx}",
                        identity,
                        lambda retry_pregrasp_pose=retry_pregrasp_pose: planner.move_to_pose_with_screw(retry_pregrasp_pose),
                    ):
                        continue
                    if not _log_action(
                        logger,
                        episode_idx,
                        planner,
                        f"retry_align_grasp_orientation_{attempt_idx}",
                        identity,
                        lambda retry_pregrasp_pose=retry_pregrasp_pose: planner.move_to_pose_with_screw(retry_pregrasp_pose),
                    ):
                        continue
                    if not _log_action(
                        logger,
                        episode_idx,
                        planner,
                        f"retry_lower_to_grasp_{attempt_idx}",
                        identity,
                        lambda retry_grasp_pose=retry_grasp_pose: planner.move_to_pose_with_screw(retry_grasp_pose),
                    ):
                        continue

                close_action = "close_gripper" if attempt_idx == 0 else f"retry_close_gripper_{attempt_idx}"
                _log_action(
                    logger,
                    episode_idx,
                    planner,
                    close_action,
                    identity,
                    lambda close_steps=close_steps: planner.close_gripper(close_steps),
                )
                if args.grasp_settle_steps > 0:
                    _log_action(
                        logger,
                        episode_idx,
                        planner,
                        "hold_closed_gripper" if attempt_idx == 0 else f"retry_hold_closed_gripper_{attempt_idx}",
                        identity,
                        lambda settle_steps=args.grasp_settle_steps: planner.close_gripper(settle_steps),
                    )
                if _is_grasping(base, obj):
                    grasped = True
                    break

            if grasped:
                break

        if not grasped:
            logger.event(
                event="object_failed",
                episode_index=episode_idx,
                step_index=planner.elapsed_steps,
                object=identity,
                reason="pickup_failed_after_retries",
                retry_budget=int(args.object_pickup_retry_count),
            )
            _log_assignment_events(logger, episode_idx, planner.elapsed_steps, identity, assignment)
            planner.open_gripper(max(args.grasp_retry_open_steps, 20))
            logger.event(
                event="episode_aborted",
                episode_index=episode_idx,
                step_index=planner.elapsed_steps,
                reason="pickup_failed_after_retries",
                object=identity,
            )
            if not args.quiet:
                print(
                    "Episode aborted: "
                    f"pickup failed after {int(args.object_pickup_retry_count)} retries "
                    f"for object slot={identity.get('slot_index')} model={identity.get('model_id')}"
                )
            info = base.evaluate()
            logger.event(
                event="episode_complete",
                episode_index=episode_idx,
                step_index=planner.elapsed_steps,
                success=False,
                sorted_count=float(info["sorted_count"][0].detach().cpu()),
                aborted=True,
            )
            return False

        bin_xy = base.bin_centers[assignment["assigned_bin_index"], :2].detach().cpu().numpy()
        obj_half_height = _object_half_height(base, obj_idx)
        carry_height = max(
            args.place_height,
            float(base.bin_wall_height) + obj_half_height + args.bin_clearance_margin,
        )
        bin_high_pose = sapien.Pose(
            p=[float(bin_xy[0]), float(bin_xy[1]), carry_height],
            q=grasp_pose.q,
        )
        bin_release_pose = sapien.Pose(
            p=[float(bin_xy[0]), float(bin_xy[1]), args.release_height],
            q=grasp_pose.q,
        )
        logger.event(event="contact_onset", episode_index=episode_idx, step_index=planner.elapsed_steps, object=identity, contact="gripper_object")

        tcp_pos = base.agent.tcp.pose.p[0].detach().cpu().numpy().astype(np.float64)
        lift_pose = sapien.Pose(
            p=[tcp_pos[0], tcp_pos[1], args.place_height],
            q=active_grasp_pose.q,
        )
        if not _log_action(
            logger,
            episode_idx,
            planner,
            "lift_object",
            identity,
            lambda: planner.move_to_pose_with_screw(lift_pose),
        ):
            logger.event(event="object_failed", episode_index=episode_idx, step_index=planner.elapsed_steps, object=identity, reason="failed_to_lift")
            planner.open_gripper(max(args.grasp_retry_open_steps, 20))
            continue

        logger.event(
            event="ft_reading_at_lift",
            episode_index=episode_idx,
            step_index=planner.elapsed_steps,
            object=identity,
            object_xyz=[round(float(x), 5) for x in obj.pose.p[0].detach().cpu().numpy().astype(np.float64)],
            ft=_finger_force_reading(base, obj),
        )
        _log_assignment_events(logger, episode_idx, planner.elapsed_steps, identity, assignment)

        _log_action(
            logger,
            episode_idx,
            planner,
            "hold_lifted_object",
            identity,
            lambda: planner.close_gripper(args.lift_hold_steps),
        )

        reached_bin_tcp = _log_action(
            logger,
            episode_idx,
            planner,
            "transport_to_assigned_bin",
            identity,
            lambda: _transport_with_fallback(planner, bin_high_pose),
        )
        if not reached_bin_tcp:
            logger.event(
                event="controller_warning",
                episode_index=episode_idx,
                step_index=planner.elapsed_steps,
                object=identity,
                warning="tcp_did_not_reach_bin_tolerance",
            )
        logger.event(
            event="object_state_after_transport",
            episode_index=episode_idx,
            step_index=planner.elapsed_steps,
            object=identity,
            object_xyz=[round(float(x), 5) for x in obj.pose.p[0].detach().cpu().numpy().astype(np.float64)],
            is_grasping=_is_grasping(base, obj),
        )

        if not _is_grasping(base, obj):
            settled_complete, info_after_drop = _wait_for_completion(
                base, planner, obj_idx, args.completion_wait_steps
            )
            if settled_complete:
                final_obj_xyz = obj.pose.p[0].detach().cpu().numpy().astype(np.float64)
                logger.event(
                    event="object_complete",
                    episode_index=episode_idx,
                    step_index=planner.elapsed_steps,
                    object=identity,
                    assigned_bin_index=assignment["assigned_bin_index"],
                    assigned_bin_role=assignment["assigned_bin_role"],
                    final_object_xyz=[round(float(x), 5) for x in final_obj_xyz],
                    target_dist=float(info_after_drop["obj_target_dists"][0, obj_idx].detach().cpu()),
                    reason=None,
                )
                continue
            logger.event(
                event="controller_warning",
                episode_index=episode_idx,
                step_index=planner.elapsed_steps,
                object=identity,
                warning="object_lost_during_transport_attempting_recovery",
            )
            recovered, recovered_q = _recover_grasp_and_lift(
                logger=logger,
                episode_idx=episode_idx,
                planner=planner,
                base_env=base,
                obj=obj,
                identity=identity,
                assignment=assignment,
                args=args,
            )
            if not recovered:
                logger.event(
                    event="object_failed",
                    episode_index=episode_idx,
                    step_index=planner.elapsed_steps,
                    object=identity,
                    reason="lost_grasp_during_transport_unrecovered",
                )
                planner.open_gripper(max(args.grasp_retry_open_steps, 20))
                continue

            bin_high_pose = sapien.Pose(
                p=[float(bin_xy[0]), float(bin_xy[1]), carry_height],
                q=recovered_q,
            )
            bin_release_pose = sapien.Pose(
                p=[float(bin_xy[0]), float(bin_xy[1]), args.release_height],
                q=recovered_q,
            )
            recovered_transport = _log_action(
                logger,
                episode_idx,
                planner,
                "retry_transport_to_assigned_bin",
                identity,
                lambda: _transport_with_fallback(planner, bin_high_pose),
            )
            if (not recovered_transport) or (not _is_grasping(base, obj)):
                logger.event(
                    event="object_failed",
                    episode_index=episode_idx,
                    step_index=planner.elapsed_steps,
                    object=identity,
                    reason="lost_grasp_after_transport_retry",
                )
                continue

        _log_action(
            logger,
            episode_idx,
            planner,
            "lower_into_bin",
            identity,
            lambda: planner.move_to_pose_with_screw(bin_release_pose),
        )
        _log_action(
            logger,
            episode_idx,
            planner,
            "open_gripper_release",
            identity,
            lambda: planner.open_gripper(args.release_open_steps),
        )
        logger.event(event="contact_release", episode_index=episode_idx, step_index=planner.elapsed_steps, object=identity, contact="gripper_object")

        if args.release_settle_steps > 0:
            _log_action(
                logger,
                episode_idx,
                planner,
                "settle_after_release",
                identity,
                lambda: planner.open_gripper(args.release_settle_steps),
            )
        
        retreat_p = np.array(bin_high_pose.p, dtype=np.float64) + np.array([0.0, 0.0, 0.06])
        retreat_pose = sapien.Pose(retreat_p, bin_high_pose.q)
        _log_action(
            logger,
            episode_idx,
            planner,
            "retreat_from_bin",
            identity,
            lambda: planner.move_to_pose_with_screw(retreat_pose),
        )
        planner.open_gripper(args.post_release_open_steps)

        physically_completed, info = _wait_for_completion(
            base, planner, obj_idx, args.post_release_eval_wait_steps
        )
        final_obj_xyz = obj.pose.p[0].detach().cpu().numpy().astype(np.float64)
        logger.event(
            event="object_complete" if physically_completed else "object_failed",
            episode_index=episode_idx,
            step_index=planner.elapsed_steps,
            object=identity,
            assigned_bin_index=assignment["assigned_bin_index"],
            assigned_bin_role=assignment["assigned_bin_role"],
            final_object_xyz=[round(float(x), 5) for x in final_obj_xyz],
            target_dist=float(info["obj_target_dists"][0, obj_idx].detach().cpu()),
            reason=None if physically_completed else "sim_did_not_report_object_completed",
        )

    info = base.evaluate()
    success = bool(info["success"][0].detach().cpu())
    logger.event(
        event="episode_complete",
        episode_index=episode_idx,
        step_index=planner.elapsed_steps,
        success=success,
        sorted_count=float(info["sorted_count"][0].detach().cpu()),
    )
    return True


def main():
    global FOLLOW_PATH_STEP_STRIDE, FOLLOW_PATH_SETTLE_STEPS
    args = tyro.cli(Args)
    FOLLOW_PATH_STEP_STRIDE = max(1, int(args.path_step_stride))
    FOLLOW_PATH_SETTLE_STEPS = max(1, int(args.settle_steps))
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    video_dir = out_dir / "videos"
    existing_videos = set(video_dir.rglob("*.mp4")) if video_dir.exists() else set()

    render_mode = "rgb_array" if args.record_video else None
    env = gym.make(
        "MassMemoryBinSort-v1",
        obs_mode=args.obs_mode,
        render_mode=render_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        control_mode="pd_joint_pos",
        robot_uids=args.robot_uid,
        human_render_camera_configs=dict(
            pose=sapien_utils.look_at([-0.30, 1.08, 1.02], [0.12, 0.0, 0.12]),
            width=args.video_width,
            height=args.video_height,
            fov=0.72,
            shader_pack=args.video_shader_pack,
        ),
        enable_shadow=args.enable_shadow,
        num_envs=1,
        num_objects=args.num_objects,
        num_ambiguous=args.num_ambiguous,
    )
    if args.record_video:
        capture_stride = max(1, int(args.video_capture_stride))
        env = SparseRecordEpisode(
            env,
            output_dir=str(out_dir / "videos"),
            save_trajectory=False,
            info_on_video=False,
            video_fps=args.video_fps,
            capture_stride=capture_stride,
            avoid_overwriting_video=True,
            max_steps_per_video=20_000,
        )

    log_path = out_dir / "oracle_events.jsonl"
    logger = OracleLogger(log_path)
    try:
        for episode_idx in range(args.episodes):
            completed = run_episode(env, episode_idx, logger, args)
            if not completed:
                break
    finally:
        logger.close()
        env.close()

    if not args.quiet:
        print(f"Wrote oracle event log: {log_path.resolve()}")
        if args.record_video:
            videos = sorted(video_dir.rglob("*.mp4"))
            new_videos = [v for v in videos if v not in existing_videos]
            if len(new_videos) == 0:
                print(f"Warning: no new MP4 files were produced in {video_dir.resolve()}")
            for video in new_videos:
                print(f"Video: {video.resolve()} size={video.stat().st_size}")


if __name__ == "__main__":
    main()
