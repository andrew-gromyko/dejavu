from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.fetch.fetch import Fetch
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


@register_env("SortYCBIntoBins-v1", max_episode_steps=120, asset_download_ids=["ycb"])
class SortYCBIntoBinsEnv(BaseEnv):
    """
    Pick-and-sort task with multiple YCB objects and three bins.

    Each episode spawns three objects (one per semantic category) and the agent must
    place each object into its corresponding bin:
    - Bin 0 (red): pantry items (boxes/cans)
    - Bin 1 (green): fruits
    - Bin 2 (blue): tools

    Observation can include:
    - visual data (rgb / rgb+depth via obs_mode)
    - state data (object poses, target bin centers, contact-force magnitudes)
    """

    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    agent: Union[Panda, PandaWristCam, Fetch]

    # Curated YCB subsets that are generally graspable in ManiSkill.
    CATEGORY_MODEL_IDS = [
        [
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "006_mustard_bottle",
            "007_tuna_fish_can",
            "008_pudding_box",
            "009_gelatin_box",
            "010_potted_meat_can",
        ],
        [
            "011_banana",
            "012_strawberry",
            "013_apple",
            "014_lemon",
            "015_peach",
            "016_pear",
            "017_orange",
            "018_plum",
        ],
        [
            "035_power_drill",
            "037_scissors",
            "040_large_marker",
            "042_adjustable_wrench",
            "043_phillips_screwdriver",
            "044_flat_screwdriver",
            "048_hammer",
        ],
    ]

    BIN_COLORS = [
        [0.92, 0.25, 0.25, 1.0],  # red
        [0.25, 0.78, 0.33, 1.0],  # green
        [0.25, 0.45, 0.92, 1.0],  # blue
    ]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.num_categories = 3
        if reconfiguration_freq is None:
            # Reconfigure every reset for singleton envs so object identities change each episode.
            reconfiguration_freq = 1 if num_envs == 1 else 0
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
                max_rigid_contact_count=2**21,
                max_rigid_patch_count=2**19,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.65, 0.75, 0.62], [0.05, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _build_bins(self):
        # Bin geometry in table frame.
        self.bin_inner_half_xy = common.to_tensor([0.055, 0.045], device=self.device)
        self.bin_wall_thickness = 0.004
        self.bin_wall_height = 0.05
        # One target bin per category.
        self.bin_centers = common.to_tensor(
            [
                [0.16, -0.18, 0.0],
                [0.16, 0.00, 0.0],
                [0.16, 0.18, 0.0],
            ],
            device=self.device,
        )
        hx, hy = self.bin_inner_half_xy.tolist()
        t = self.bin_wall_thickness
        hh = self.bin_wall_height / 2
        self.bin_parts = []
        for i in range(self.num_categories):
            cx, cy = self.bin_centers[i, :2].tolist()
            color = self.BIN_COLORS[i]
            # Floor.
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
            # +Y / -Y walls.
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
            # +X / -X walls.
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

        self._raw_objects_by_category: list[list[Actor]] = []
        self.objects: list[Actor] = []
        self.sampled_model_ids: list[list[str]] = []

        def _normalize_model_id(raw_model_id):
            # Batched RNG can sometimes return nested singleton containers.
            if isinstance(raw_model_id, np.ndarray):
                if raw_model_id.size != 1:
                    raise ValueError(
                        f"Expected singleton sampled model id, got array shape={raw_model_id.shape}"
                    )
                raw_model_id = raw_model_id.reshape(()).item()
            if isinstance(raw_model_id, (list, tuple)):
                if len(raw_model_id) != 1:
                    raise ValueError(
                        f"Expected singleton sampled model id list/tuple, got len={len(raw_model_id)}"
                    )
                raw_model_id = raw_model_id[0]
            return str(raw_model_id)

        for cat_idx, model_pool in enumerate(self.CATEGORY_MODEL_IDS):
            # _batched_episode_rng is already vectorized over envs.
            sampled = self._batched_episode_rng.choice(model_pool, replace=True)
            cat_raw_objs: list[Actor] = []
            cat_model_ids: list[str] = []
            for env_i, model_id in enumerate(sampled):
                model_id = _normalize_model_id(model_id)
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.2 + 0.1 * cat_idx])
                builder.set_scene_idxs([env_i])
                obj = builder.build(name=f"cat{cat_idx}_{model_id}_{env_i}")
                self.remove_from_state_dict_registry(obj)
                cat_raw_objs.append(obj)
                cat_model_ids.append(model_id)
            merged = Actor.merge(cat_raw_objs, name=f"ycb_cat_{cat_idx}")
            self.add_to_state_dict_registry(merged)
            self._raw_objects_by_category.append(cat_raw_objs)
            self.objects.append(merged)
            self.sampled_model_ids.append(cat_model_ids)

    def _after_reconfigure(self, options: dict):
        # Per-object z offsets so each object sits on the table when pose.z = object_z.
        object_zs = []
        for cat_raw_objs in self._raw_objects_by_category:
            cat_zs = []
            for obj in cat_raw_objs:
                mesh = obj.get_first_collision_mesh()
                cat_zs.append(-mesh.bounding_box.bounds[0, 2])
            object_zs.append(cat_zs)
        self.object_zs = common.to_tensor(object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Spawn objects away from bins so sorting requires actual transport.
            sampler = randomization.UniformPlacementSampler(
                bounds=[[-0.18, -0.22], [0.04, 0.22]],
                batch_size=b,
                device=self.device,
            )
            for cat_idx, obj in enumerate(self.objects):
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, :2] = sampler.sample(radius=0.06, max_trials=200, verbose=False)
                xyz[:, 2] = self.object_zs[cat_idx, env_idx]
                qs = randomization.random_quaternions(
                    b, device=self.device, lock_x=True, lock_y=True
                )
                obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        sorted_flags = []
        target_dists = []
        grasp_flags = []
        contact_force_pairs = []
        for cat_idx, obj in enumerate(self.objects):
            obj_pos = obj.pose.p
            target_xy = self.bin_centers[cat_idx, :2]
            xy_delta = obj_pos[:, :2] - target_xy
            target_dists.append(torch.linalg.norm(xy_delta, dim=1))

            inside_x = torch.abs(xy_delta[:, 0]) <= (self.bin_inner_half_xy[0] - 0.005)
            inside_y = torch.abs(xy_delta[:, 1]) <= (self.bin_inner_half_xy[1] - 0.005)
            in_target_bin = torch.logical_and(inside_x, inside_y)

            is_static = obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_grasped = self.agent.is_grasping(obj)
            sorted_flag = in_target_bin & is_static & (~is_grasped)

            # Tactile-like features via contact force magnitudes on each finger.
            l_force_vec = self.scene.get_pairwise_contact_forces(self.agent.finger1_link, obj)
            r_force_vec = self.scene.get_pairwise_contact_forces(self.agent.finger2_link, obj)
            l_force = torch.linalg.norm(l_force_vec, dim=1)
            r_force = torch.linalg.norm(r_force_vec, dim=1)

            sorted_flags.append(sorted_flag)
            grasp_flags.append(is_grasped)
            contact_force_pairs.append(torch.stack([l_force, r_force], dim=1))

        obj_sorted = torch.stack(sorted_flags, dim=1)  # (N, 3)
        obj_target_dists = torch.stack(target_dists, dim=1)  # (N, 3)
        obj_grasped = torch.stack(grasp_flags, dim=1)  # (N, 3)
        obj_contact_forces = torch.stack(contact_force_pairs, dim=1)  # (N, 3, 2)
        success = torch.all(obj_sorted, dim=1)
        sorted_count = obj_sorted.float().sum(dim=1)
        return dict(
            obj_sorted=obj_sorted,
            obj_grasped=obj_grasped,
            obj_target_dists=obj_target_dists,
            obj_contact_forces=obj_contact_forces,
            sorted_count=sorted_count,
            success=success,
            fail=torch.zeros_like(success),
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            sorted_count=info["sorted_count"],
        )
        if "state" in self.obs_mode:
            obj_poses = torch.stack([obj.pose.raw_pose for obj in self.objects], dim=1)
            obj_pos = torch.stack([obj.pose.p for obj in self.objects], dim=1)
            tcp_to_obj = obj_pos - self.agent.tcp.pose.p[:, None, :]
            target_bin_centers = self.bin_centers[None, :, :].repeat(self.num_envs, 1, 1)
            category_one_hot = torch.eye(self.num_categories, device=self.device)[None, :, :].repeat(
                self.num_envs, 1, 1
            )

            def _flatten_env_batch(x: torch.Tensor) -> torch.Tensor:
                # ManiSkill's flatten_state_dict expects tensors with ndim <= 2.
                if x.ndim <= 2:
                    return x
                return x.reshape(self.num_envs, -1)

            obs.update(
                obj_poses=_flatten_env_batch(obj_poses),
                tcp_to_obj=_flatten_env_batch(tcp_to_obj),
                target_bin_centers=_flatten_env_batch(target_bin_centers),
                category_one_hot=_flatten_env_batch(category_one_hot),
                obj_contact_forces=_flatten_env_batch(info["obj_contact_forces"]),
                obj_sorted=info["obj_sorted"].float(),
                obj_grasped=info["obj_grasped"].float(),
                obj_target_dists=info["obj_target_dists"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Per-object sort completion.
        reward = 2.0 * info["sorted_count"]

        # Progress term: encourage moving each object toward its assigned bin.
        progress = 1 - torch.tanh(4.0 * info["obj_target_dists"])
        reward += 0.35 * progress.sum(dim=1)

        # Mild shaping for making contact (tactile signal usage).
        contact_mag = info["obj_contact_forces"].sum(dim=(1, 2))
        reward += 0.1 * torch.clamp(contact_mag / 20.0, 0.0, 1.0)

        # Sparse completion bonus.
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10.0
