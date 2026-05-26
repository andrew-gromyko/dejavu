"""Mug contact probing environment."""

from __future__ import annotations

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("MugContactProbe-v1", max_episode_steps=400, asset_download_ids=["ycb"])
class MugContactProbeEnv(BaseEnv):
    """Single-scene contact probing env with Panda + one YCB mug."""

    SUPPORTED_ROBOTS = ["panda"]
    SUPPORTED_REWARD_MODES = ["none"]
    agent: Panda

    def __init__(self, *args, robot_uids: str = "panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.35, -0.55, 0.42], target=[0.06, 0.0, 0.10])
        return [CameraConfig("probe_camera", pose, 640, 480, np.pi / 3, 0.01, 10, shader_pack="default")]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.35, -0.55, 0.42], target=[0.06, 0.0, 0.10])
        return CameraConfig("render_camera", pose, 640, 480, np.pi / 3, 0.01, 20, shader_pack="default")

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=False
        )
        self.scene.add_directional_light([0, 0, -1], [0.5, 0.5, 0.5])

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        for obj in self._hidden_objects:
            obj.hide_visual()
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.capture_sensor_data()
        
        sensor_obs = dict()
        for name, sensor in self.scene.sensors.items():
            # Only fetch RGB/Color to save GPU-to-CPU transfer bandwidth and avoid lag
            sensor_obs[name] = sensor.get_obs(
                rgb=True,
                depth=False,
                position=False,
                segmentation=False,
                normal=False,
                albedo=False,
                apply_texture_transforms=apply_texture_transforms
            )
        return sensor_obs

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=0.0)
        self.table_scene.build()

        builder = actors.get_actor_builder(self.scene, id="ycb:025_mug")
        builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 0.0]))
        if self.num_envs > 1:
            builder.set_scene_idxs(torch.arange(self.num_envs, device=self.device))
        self.mug = builder.build(name="ycb_025_mug")

    def _after_reconfigure(self, options: dict):
        # Place mug exactly on the table surface in a deterministic orientation.
        mesh = self.mug.get_first_collision_mesh()
        self.mug_z = float(-mesh.bounding_box.bounds[0, 2])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            b = len(env_idx)

            init_qpos = torch.tensor(
                [0.0, 0.15, 0.0, -2.10, 0.0, 2.30, 0.78, 0.04, 0.04],
                dtype=torch.float32,
                device=self.device,
            )
            self.agent.reset(init_qpos.repeat(b, 1))

            # Start separated from the gripper and already resting on the table.
            mug_pos = torch.tensor([0.055, 0.0, self.mug_z], device=self.device).repeat(b, 1)
            # 90deg yaw gives a stable side grasp for this scripted probe.
            mug_pose = Pose.create_from_pq(
                p=mug_pos, q=[0.70710678, 0.0, 0.0, 0.70710678]
            )
            self.mug.set_pose(mug_pose)

    def evaluate(self):
        return {}
