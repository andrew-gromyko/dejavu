from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("MassMemoryBinSort-v1", max_episode_steps=1200, asset_download_ids=["ycb"])
class MassMemoryBinSortEnv(BaseEnv):
    """
    Sequential YCB sorting by mass threshold with ambiguous distractors.

    Episode structure:
    1. Spawn N YCB objects on a table.
    2. Agent must process objects sequentially.
    3. For each object, grasp and lift, hold for 1-2 seconds, then place into the
       correct bin according to mass threshold (light/heavy).

    Key controls:
    - num_objects: N
    - num_ambiguous: D objects with masses near threshold
    - randomize_bin_roles: if True, light/heavy bin side is randomized per episode
    - bin_color_mode: "identical" or "random"
    """

    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    YCB_MODEL_POOL = [
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "010_potted_meat_can",
        "011_banana",
        "013_apple",
        "014_lemon",
        "015_peach",
        "016_pear",
        "017_orange",
        "018_plum",
        "021_bleach_cleanser",
        "024_bowl",
        "025_mug",
        "035_power_drill",
        "037_scissors",
        "040_large_marker",
        "042_adjustable_wrench",
        "043_phillips_screwdriver",
        "044_flat_screwdriver",
        "048_hammer",
    ]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=1,
        num_objects=6,
        num_ambiguous=2,
        mass_threshold=0.25,
        ambiguity_delta=0.04,
        light_mass_range=(0.05, 0.18),
        heavy_mass_range=(0.32, 0.60),
        hold_seconds_range=(1.0, 2.0),
        min_lift_height=0.09,
        randomize_bin_roles=False,
        bin_color_mode="identical",
        include_privileged_mass_in_state=False,
        **kwargs,
    ):
        if num_objects < 1:
            raise ValueError("num_objects must be >= 1")
        if num_ambiguous < 0 or num_ambiguous > num_objects:
            raise ValueError("num_ambiguous must be in [0, num_objects]")
        if ambiguity_delta < 0:
            raise ValueError("ambiguity_delta must be >= 0")
        if bin_color_mode not in {"identical", "random"}:
            raise ValueError("bin_color_mode must be 'identical' or 'random'")

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.num_objects = int(num_objects)
        self.num_ambiguous = int(num_ambiguous)
        self.mass_threshold = float(mass_threshold)
        self.ambiguity_delta = float(ambiguity_delta)
        self.light_mass_range = (float(light_mass_range[0]), float(light_mass_range[1]))
        self.heavy_mass_range = (float(heavy_mass_range[0]), float(heavy_mass_range[1]))
        self.hold_seconds_range = (
            float(hold_seconds_range[0]),
            float(hold_seconds_range[1]),
        )
        self.min_lift_height = float(min_lift_height)
        self.randomize_bin_roles = bool(randomize_bin_roles)
        self.bin_color_mode = str(bin_color_mode)
        self.include_privileged_mass_in_state = bool(include_privileged_mass_in_state)

        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22,
                max_rigid_patch_count=2**20,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.45, -0.55, 0.62], target=[0.03, 0, 0.12])
        return [CameraConfig("base_camera", pose, 384, 384, np.pi / 3, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.72, -0.78, 0.68], [0.04, 0.0, 0.16])
        return CameraConfig("render_camera", pose, 960, 720, 0.78, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _sample_mass_spec(self):
        masses = np.zeros((self.num_envs, self.num_objects), dtype=np.float32)
        ambiguous = np.zeros((self.num_envs, self.num_objects), dtype=bool)
        env_seeds = self._batched_episode_rng.randint(
            0, np.iinfo(np.int32).max, size=self.num_envs
        )
        for env_i in range(self.num_envs):
            rng = np.random.default_rng(int(env_seeds[env_i]))
            if self.num_ambiguous > 0:
                amb_idxs = rng.choice(
                    self.num_objects, size=self.num_ambiguous, replace=False
                )
                ambiguous[env_i, amb_idxs] = True
            for obj_i in range(self.num_objects):
                if ambiguous[env_i, obj_i]:
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    delta = rng.uniform(0.0, self.ambiguity_delta)
                    masses[env_i, obj_i] = self.mass_threshold + sign * delta
                else:
                    if rng.random() < 0.5:
                        masses[env_i, obj_i] = rng.uniform(*self.light_mass_range)
                    else:
                        masses[env_i, obj_i] = rng.uniform(*self.heavy_mass_range)
        return masses, ambiguous

    def _sample_bin_colors(self):
        if self.bin_color_mode == "identical":
            return [[0.67, 0.67, 0.67, 1.0], [0.67, 0.67, 0.67, 1.0]]
        colors = []
        for _ in range(2):
            rgb = np.random.uniform(0.2, 0.85, size=3).tolist()
            colors.append([rgb[0], rgb[1], rgb[2], 1.0])
        return colors

    def _sample_object_spawn_xy(self, batch_size: int) -> torch.Tensor:
        """Return separated tabletop spawn positions.

        The YCB collision geometries have varied origins and footprints. A shuffled
        grid is less pretty than rejection sampling, but it avoids accidental initial
        overlaps that inject large object velocities before the task starts.
        """
        x_values = np.array([-0.38, -0.24, -0.10], dtype=np.float32)
        rows = max(2, int(np.ceil(self.num_objects / len(x_values))))
        y_values = np.linspace(-0.24, 0.24, rows, dtype=np.float32)
        grid = np.array([[x, y] for y in y_values for x in x_values], dtype=np.float32)
        if len(grid) < self.num_objects:
            raise RuntimeError("spawn grid construction bug")

        env_seeds = self._batched_episode_rng.randint(
            0, np.iinfo(np.int32).max, size=batch_size
        )
        out = np.zeros((batch_size, self.num_objects, 2), dtype=np.float32)
        for env_i in range(batch_size):
            rng = np.random.default_rng(int(env_seeds[env_i]))
            perm = rng.permutation(len(grid))[: self.num_objects]
            jitter = rng.uniform(-0.012, 0.012, size=(self.num_objects, 2)).astype(np.float32)
            out[env_i] = grid[perm] + jitter
        return common.to_tensor(out, device=self.device)

    def _build_bins(self):
        self.bin_inner_half_xy = common.to_tensor([0.16, 0.13], device=self.device)
        self.bin_wall_thickness = 0.006
        self.bin_wall_height = 0.16
        self.bin_centers = common.to_tensor(
            [
                [0.12, -0.24, 0.0],  # left
                [0.12, 0.24, 0.0],  # right
            ],
            device=self.device,
        )
        colors = self._sample_bin_colors()
        hx, hy = self.bin_inner_half_xy.tolist()
        t = self.bin_wall_thickness
        hh = self.bin_wall_height / 2
        self.bin_parts = []
        for i in range(2):
            cx, cy = self.bin_centers[i, :2].tolist()
            color = colors[i]
            self.bin_parts.append(
                actors.build_box(
                    self.scene,
                    half_sizes=[hx, hy, t],
                    color=color,
                    name=f"bin{i}_floor",
                    body_type="static",
                    initial_pose=sapien.Pose(p=[cx, cy, t]),
                )
            )
            self.bin_parts.append(
                actors.build_box(
                    self.scene,
                    half_sizes=[hx + t, t, hh],
                    color=color,
                    name=f"bin{i}_wall_pos_y",
                    body_type="static",
                    initial_pose=sapien.Pose(p=[cx, cy + hy + t, hh]),
                )
            )
            self.bin_parts.append(
                actors.build_box(
                    self.scene,
                    half_sizes=[hx + t, t, hh],
                    color=color,
                    name=f"bin{i}_wall_neg_y",
                    body_type="static",
                    initial_pose=sapien.Pose(p=[cx, cy - hy - t, hh]),
                )
            )
            self.bin_parts.append(
                actors.build_box(
                    self.scene,
                    half_sizes=[t, hy + t, hh],
                    color=color,
                    name=f"bin{i}_wall_pos_x",
                    body_type="static",
                    initial_pose=sapien.Pose(p=[cx + hx + t, cy, hh]),
                )
            )
            self.bin_parts.append(
                actors.build_box(
                    self.scene,
                    half_sizes=[t, hy + t, hh],
                    color=color,
                    name=f"bin{i}_wall_neg_x",
                    body_type="static",
                    initial_pose=sapien.Pose(p=[cx - hx - t, cy, hh]),
                )
            )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._build_bins()

        self._raw_objects_by_slot: list[list[Actor]] = []
        self.objects: list[Actor] = []
        self.sampled_model_ids: list[list[str]] = []
        self.object_masses_np, self.object_is_ambiguous_np = self._sample_mass_spec()

        def _normalize_model_id(raw_model_id):
            if isinstance(raw_model_id, np.ndarray):
                if raw_model_id.size != 1:
                    raise ValueError(
                        f"Expected singleton sampled model id, got shape={raw_model_id.shape}"
                    )
                raw_model_id = raw_model_id.reshape(()).item()
            if isinstance(raw_model_id, (list, tuple)):
                if len(raw_model_id) != 1:
                    raise ValueError(
                        f"Expected singleton sampled model id list/tuple, got len={len(raw_model_id)}"
                    )
                raw_model_id = raw_model_id[0]
            return str(raw_model_id)

        for slot_i in range(self.num_objects):
            sampled = self._batched_episode_rng.choice(self.YCB_MODEL_POOL, replace=True)
            raw_objs: list[Actor] = []
            slot_model_ids: list[str] = []
            for env_i, model_id in enumerate(sampled):
                model_id = _normalize_model_id(model_id)
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.25 + 0.04 * slot_i])
                builder.set_scene_idxs([env_i])
                obj = builder.build(name=f"obj{slot_i}_{model_id}_{env_i}")
                obj.mass = float(self.object_masses_np[env_i, slot_i])
                self.remove_from_state_dict_registry(obj)
                raw_objs.append(obj)
                slot_model_ids.append(model_id)
            merged = Actor.merge(raw_objs, name=f"ycb_slot_{slot_i}")
            self.add_to_state_dict_registry(merged)
            self._raw_objects_by_slot.append(raw_objs)
            self.objects.append(merged)
            self.sampled_model_ids.append(slot_model_ids)

    def _after_reconfigure(self, options: dict):
        object_zs = []
        for raw_objs in self._raw_objects_by_slot:
            slot_zs = []
            for obj in raw_objs:
                mesh = obj.get_first_collision_mesh()
                slot_zs.append(-mesh.bounding_box.bounds[0, 2])
            object_zs.append(slot_zs)
        self.object_zs = common.to_tensor(object_zs, device=self.device)

        self.object_masses = common.to_tensor(self.object_masses_np, device=self.device)
        self.object_is_ambiguous = common.to_tensor(
            self.object_is_ambiguous_np.astype(np.float32), device=self.device
        ).bool()
        self.object_is_heavy = self.object_masses >= self.mass_threshold

        self.object_hold_counters = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.long, device=self.device
        )
        self.object_hold_done = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.bool, device=self.device
        )
        self.object_completed = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.bool, device=self.device
        )
        self.required_hold_steps = torch.zeros(
            (self.num_envs, self.num_objects), dtype=torch.long, device=self.device
        )
        self.current_object_idx = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self.light_bin_index = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            spawn_xy = self._sample_object_spawn_xy(b)
            for obj_i, obj in enumerate(self.objects):
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, :2] = spawn_xy[:, obj_i]
                xyz[:, 2] = 0.12 + 0.01 * (obj_i % 3)
                qs = randomization.random_quaternions(
                    b, device=self.device, lock_x=True, lock_y=True
                )
                obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            self.object_hold_counters[env_idx] = 0
            self.object_hold_done[env_idx] = False
            self.object_completed[env_idx] = False
            self.current_object_idx[env_idx] = 0

            if self.randomize_bin_roles:
                rand_roles = common.to_tensor(
                    self._batched_episode_rng.randint(0, 2, size=b), device=self.device
                ).to(torch.long)
                self.light_bin_index[env_idx] = rand_roles
            else:
                self.light_bin_index[env_idx] = 0

            min_steps = max(1, int(round(self.hold_seconds_range[0] * self.control_freq)))
            max_steps = max(min_steps, int(round(self.hold_seconds_range[1] * self.control_freq)))
            hold_steps = common.to_tensor(
                self._batched_episode_rng.randint(
                    min_steps, max_steps + 1, size=(b, self.num_objects)
                ),
                device=self.device,
            ).to(torch.long)
            self.required_hold_steps[env_idx] = hold_steps

    def evaluate(self):
        contact_force_pairs = []
        grasp_flags = []
        target_dists = []

        light_bin_centers = self.bin_centers[self.light_bin_index]
        heavy_bin_centers = self.bin_centers[1 - self.light_bin_index]

        for obj_i, obj in enumerate(self.objects):
            obj_pos = obj.pose.p
            is_grasped = self.agent.is_grasping(obj)
            grasp_flags.append(is_grasped)

            lifted = obj_pos[:, 2] >= self.min_lift_height
            hold_cond = is_grasped & lifted
            prev_counter = self.object_hold_counters[:, obj_i]
            new_counter = torch.where(
                hold_cond,
                prev_counter + 1,
                torch.where(
                    self.object_hold_done[:, obj_i], prev_counter, torch.zeros_like(prev_counter)
                ),
            )
            self.object_hold_counters[:, obj_i] = new_counter
            self.object_hold_done[:, obj_i] = self.object_hold_done[:, obj_i] | (
                new_counter >= self.required_hold_steps[:, obj_i]
            )

            target_xy = torch.where(
                self.object_is_heavy[:, obj_i][:, None],
                heavy_bin_centers[:, :2],
                light_bin_centers[:, :2],
            )
            xy_delta = obj_pos[:, :2] - target_xy
            target_dists.append(torch.linalg.norm(xy_delta, dim=1))
            inside_x = torch.abs(xy_delta[:, 0]) <= (self.bin_inner_half_xy[0] - 0.005)
            inside_y = torch.abs(xy_delta[:, 1]) <= (self.bin_inner_half_xy[1] - 0.005)
            in_target_bin = torch.logical_and(inside_x, inside_y)
            is_static = obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)

            can_complete = (
                self.object_hold_done[:, obj_i] & in_target_bin & is_static & (~is_grasped)
            )
            active_mask = self.current_object_idx == obj_i
            self.object_completed[:, obj_i] = self.object_completed[:, obj_i] | (
                can_complete & active_mask
            )

            l_force_vec = self.scene.get_pairwise_contact_forces(
                self.agent.finger1_link, obj
            )
            r_force_vec = self.scene.get_pairwise_contact_forces(
                self.agent.finger2_link, obj
            )
            l_force = torch.linalg.norm(l_force_vec, dim=1)
            r_force = torch.linalg.norm(r_force_vec, dim=1)
            contact_force_pairs.append(torch.stack([l_force, r_force], dim=1))

        done_idx = torch.minimum(
            self.current_object_idx,
            torch.full_like(self.current_object_idx, self.num_objects - 1),
        )
        row_idx = torch.arange(self.num_envs, device=self.device)
        active_completed = self.object_completed[row_idx, done_idx]
        can_advance = active_completed & (self.current_object_idx < self.num_objects)
        self.current_object_idx = torch.where(
            can_advance, self.current_object_idx + 1, self.current_object_idx
        )
        self.current_object_idx = torch.clamp(self.current_object_idx, max=self.num_objects)

        obj_grasped = torch.stack(grasp_flags, dim=1)
        obj_target_dists = torch.stack(target_dists, dim=1)
        obj_contact_forces = torch.stack(contact_force_pairs, dim=1)
        obj_hold_progress = torch.clamp(
            self.object_hold_counters.float() / self.required_hold_steps.float(), 0.0, 1.0
        )

        sorted_count = self.object_completed.float().sum(dim=1)
        success = sorted_count == float(self.num_objects)
        return dict(
            obj_completed=self.object_completed,
            obj_grasped=obj_grasped,
            obj_target_dists=obj_target_dists,
            obj_contact_forces=obj_contact_forces,
            obj_hold_done=self.object_hold_done,
            obj_hold_progress=obj_hold_progress,
            obj_is_ambiguous=self.object_is_ambiguous,
            sorted_count=sorted_count,
            current_object_idx=self.current_object_idx,
            success=success,
            fail=torch.zeros_like(success),
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            sorted_count=info["sorted_count"],
            current_object_idx=info["current_object_idx"].float(),
        )
        if "state" in self.obs_mode:
            obj_poses = torch.stack([obj.pose.raw_pose for obj in self.objects], dim=1)
            obj_pos = torch.stack([obj.pose.p for obj in self.objects], dim=1)
            tcp_to_obj = obj_pos - self.agent.tcp.pose.p[:, None, :]
            target_bin_centers = self.bin_centers[None, :, :].repeat(self.num_envs, 1, 1)
            light_bin_one_hot = torch.nn.functional.one_hot(
                self.light_bin_index, num_classes=2
            ).float()

            def _flatten_env_batch(x: torch.Tensor) -> torch.Tensor:
                if x.ndim <= 2:
                    return x
                return x.reshape(self.num_envs, -1)

            obs.update(
                obj_poses=_flatten_env_batch(obj_poses),
                tcp_to_obj=_flatten_env_batch(tcp_to_obj),
                target_bin_centers=_flatten_env_batch(target_bin_centers),
                light_bin_one_hot=light_bin_one_hot,
                obj_contact_forces=_flatten_env_batch(info["obj_contact_forces"]),
                obj_hold_progress=info["obj_hold_progress"],
                obj_hold_done=info["obj_hold_done"].float(),
                obj_completed=info["obj_completed"].float(),
                obj_target_dists=info["obj_target_dists"],
                obj_grasped=info["obj_grasped"].float(),
                obj_is_ambiguous=info["obj_is_ambiguous"].float(),
            )
            if self.include_privileged_mass_in_state:
                obs.update(
                    obj_masses=self.object_masses,
                    obj_is_heavy=self.object_is_heavy.float(),
                )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = 2.0 * info["sorted_count"]

        active_idx = torch.minimum(
            info["current_object_idx"],
            torch.full_like(info["current_object_idx"], self.num_objects - 1),
        )
        row_idx = torch.arange(self.num_envs, device=self.device)
        active_dist = info["obj_target_dists"][row_idx, active_idx]
        active_hold = info["obj_hold_progress"][row_idx, active_idx]
        active_grasp = info["obj_grasped"][row_idx, active_idx].float()

        reward += 0.6 * (1.0 - torch.tanh(4.0 * active_dist))
        reward += 0.6 * active_hold
        reward += 0.1 * active_grasp

        contact_mag = info["obj_contact_forces"].sum(dim=(1, 2))
        reward += 0.1 * torch.clamp(contact_mag / 20.0, 0.0, 1.0)

        reward[info["success"]] = 12.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 12.0
