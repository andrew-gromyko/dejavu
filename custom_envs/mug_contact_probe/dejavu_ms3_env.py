"""ManiSkill3 visualization environment for the DejaVu memory benchmark."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

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

from custom_envs.mug_contact_probe.dejavu_env import DejaVuEnv


ROOT = Path(__file__).resolve().parents[2]
OBJECTFOLDER_ROOT = ROOT / "assets" / "objectfolder2" / "ObjectFolder1-100"
OBJECTFOLDER_TAR = ROOT / "assets" / "objectfolder2" / "ObjectFolder1-100.tar.gz"
OBJECTFOLDER_SCALE = 0.48


def _local_objectfolder_ids() -> list[int]:
    if not OBJECTFOLDER_ROOT.exists():
        return []
    ids = []
    for p in OBJECTFOLDER_ROOT.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / "model.obj").exists() and (p / "ObjectFile.pth").exists():
            bounds_min, bounds_max = _read_obj_bounds(p / "model.obj")
            scaled_size = (bounds_max - bounds_min) * OBJECTFOLDER_SCALE
            horizontal = scaled_size[:2]
            height = float(scaled_size[2])
            # Very tiny objects are visually valid ObjectFolder assets but unreliable
            # for this simple Panda parallel-jaw grasp script.
            if 0.035 <= float(np.max(horizontal)) <= 0.070:
                ids.append(int(p.name))
    return sorted(ids)


def _tar_objectfolder_ids() -> list[int]:
    if not OBJECTFOLDER_TAR.exists():
        return []
    ids = set()
    with tarfile.open(OBJECTFOLDER_TAR, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) >= 3 and parts[0] == "ObjectFolder1-100" and parts[1].isdigit() and parts[2] == "model.obj":
                ids.add(int(parts[1]))
    return sorted(ids)


def _safe_member_for_object(member_name: str, object_id: int) -> bool:
    prefix = f"ObjectFolder1-100/{object_id}/"
    path = Path(member_name)
    return member_name.startswith(prefix) and ".." not in path.parts and path.is_absolute() is False


def _ensure_objectfolder_asset(object_id: int) -> Path:
    object_dir = OBJECTFOLDER_ROOT / str(object_id)
    if (object_dir / "model.obj").exists() and (object_dir / "ObjectFile.pth").exists():
        _sanitize_objectfolder_material(object_dir)
        return object_dir
    if not OBJECTFOLDER_TAR.exists():
        raise FileNotFoundError(f"Missing {OBJECTFOLDER_TAR}; cannot extract ObjectFolder object {object_id}")

    OBJECTFOLDER_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OBJECTFOLDER_TAR, "r:gz") as tar:
        members = [m for m in tar.getmembers() if _safe_member_for_object(m.name, object_id)]
        if not members:
            raise FileNotFoundError(f"ObjectFolder object {object_id} not found in {OBJECTFOLDER_TAR}")
        tar.extractall(path=OBJECTFOLDER_ROOT.parent, members=members)
    if not (object_dir / "model.obj").exists():
        raise FileNotFoundError(f"Extracted ObjectFolder object {object_id}, but model.obj is missing")
    _sanitize_objectfolder_material(object_dir)
    return object_dir


def _sanitize_objectfolder_material(object_dir: Path) -> None:
    """Trim malformed trailing whitespace from ObjectFolder texture paths."""

    mtl_path = object_dir / "model.mtl"
    if not mtl_path.exists():
        return
    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    fixed = []
    changed = False
    for line in lines:
        if line.lstrip().startswith("map_Kd "):
            prefix, texture_path = line.split("map_Kd", 1)
            new_line = f"{prefix}map_Kd {texture_path.strip()}"
            changed = changed or new_line != line
            fixed.append(new_line)
        else:
            fixed.append(line.rstrip())
    if changed:
        mtl_path.write_text("\n".join(fixed) + "\n", encoding="utf-8")


def _read_obj_bounds(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise ValueError(f"No vertices found in {obj_path}")
    verts = np.asarray(vertices, dtype=np.float32)
    return verts.min(axis=0), verts.max(axis=0)


def _select_objectfolder_ids(count: int, seed: int) -> list[int]:
    ids = _local_objectfolder_ids()
    if not ids:
        tar_ids = _tar_objectfolder_ids()
        for object_id in tar_ids:
            _ensure_objectfolder_asset(object_id)
            ids.append(object_id)
            if ids:
                break
    ids = sorted(set(ids))
    if not ids:
        raise RuntimeError(f"No ObjectFolder objects found in {OBJECTFOLDER_ROOT}")
    rng = np.random.default_rng(seed)
    replace = len(ids) < count
    return [int(x) for x in rng.choice(ids, size=count, replace=replace).tolist()]


@register_env("DejaVuMemory-v1", max_episode_steps=5000)
class DejaVuMemoryEnv(BaseEnv):
    """View a DejaVu memory episode in ManiSkill3.

    The environment is intentionally a visualization harness: each generated
    encounter becomes one tabletop primitive. External code can drive the Panda
    with normal ManiSkill actions and call :meth:`advance_encounter` after a
    probe is complete.
    """

    SUPPORTED_ROBOTS = ["panda"]
    SUPPORTED_REWARD_MODES = ["none"]
    agent: Panda

    object_half_size = 0.035
    inactive_xyz = np.array([0.0, 0.0, -2.0], dtype=np.float32)
    grid_origin_xy = np.array([0.05, -0.18], dtype=np.float32)
    grid_spacing_xy = np.array([0.11, 0.12], dtype=np.float32)
    grid_cols = 4

    def __init__(
        self,
        *args,
        task: str = "mass",
        H: int = 3,
        D: int = 1,
        benchmark_seed: int = 0,
        robot_uids: str = "panda",
        **kwargs,
    ):
        self.task = task
        self.H = int(H)
        self.D = int(D)
        self.benchmark_seed = int(benchmark_seed)
        self.benchmark = DejaVuEnv(task=self.task, H=self.H, D=self.D, seed=self.benchmark_seed)
        self.encounter_steps = [s for s in self.benchmark.episode if s["type"] == "encounter"]
        self.query_step = self.benchmark.episode[-1]
        self.objectfolder_ids = _select_objectfolder_ids(len(self.encounter_steps), self.benchmark_seed)
        self.asset_infos = []
        for step, objectfolder_id in zip(self.encounter_steps, self.objectfolder_ids, strict=True):
            object_dir = _ensure_objectfolder_asset(objectfolder_id)
            obj_path = object_dir / "model.obj"
            objectfile_path = object_dir / "ObjectFile.pth"
            bounds_min, bounds_max = _read_obj_bounds(obj_path)
            asset_info = {
                "objectfolder_id": objectfolder_id,
                "object_dir": object_dir,
                "model_path": obj_path,
                "objectfile_path": objectfile_path,
                "bounds_min": bounds_min,
                "bounds_max": bounds_max,
            }
            self.asset_infos.append(asset_info)
            step["observation"]["objectfolder_id"] = objectfolder_id
            step["observation"]["objectfile_path"] = str(objectfile_path)
            step["observation"]["model_path"] = str(obj_path)
        self.active_encounter_index = 0
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.45, -0.65, 0.48], target=[0.06, 0.0, 0.08])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 3, 0.01, 10, shader_pack="default")]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.45, -0.65, 0.48], target=[0.06, 0.0, 0.08])
        return CameraConfig("render_camera", pose, 768, 576, np.pi / 3, 0.01, 20, shader_pack="default")

    @property
    def active_step(self) -> dict[str, Any] | None:
        if self.active_encounter_index >= len(self.encounter_steps):
            return None
        return self.encounter_steps[self.active_encounter_index]

    @property
    def active_object(self):
        if self.active_encounter_index >= len(self.objects):
            return None
        return self.objects[self.active_encounter_index]

    def encounter_position(self, encounter_index: int) -> np.ndarray:
        """Return the table position for an encounter object."""

        row = encounter_index // self.grid_cols
        col = encounter_index % self.grid_cols
        xy = self.grid_origin_xy + np.array(
            [row * self.grid_spacing_xy[0], col * self.grid_spacing_xy[1]],
            dtype=np.float32,
        )
        return np.array([xy[0], xy[1], self.object_half_size], dtype=np.float32)

    def asset_pose_position(self, encounter_index: int, table_xy: np.ndarray) -> np.ndarray:
        """Actor pose that puts an ObjectFolder mesh on the table at table_xy."""

        bounds_min = self.asset_infos[encounter_index]["bounds_min"]
        bounds_max = self.asset_infos[encounter_index]["bounds_max"]
        center_xy = 0.5 * (bounds_min[:2] + bounds_max[:2]) * OBJECTFOLDER_SCALE
        return np.array(
            [
                table_xy[0] - center_xy[0],
                table_xy[1] - center_xy[1],
                -bounds_min[2] * OBJECTFOLDER_SCALE,
            ],
            dtype=np.float32,
        )

    def grasp_position(self, encounter_index: int) -> np.ndarray:
        """Approximate mesh center used as the scripted grasp target."""

        base = self.encounter_position(encounter_index)
        bounds_min = self.asset_infos[encounter_index]["bounds_min"]
        bounds_max = self.asset_infos[encounter_index]["bounds_max"]
        height = max(0.04, float((bounds_max[2] - bounds_min[2]) * OBJECTFOLDER_SCALE))
        grasp_z = min(max(0.04, height) * 0.55, 0.055)
        return np.array([base[0], base[1], grasp_z], dtype=np.float32)

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.35, 0.35, 0.35])
        self.scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=False)
        self.scene.add_directional_light([0, 0, -1], [0.6, 0.6, 0.6])

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=0.0)
        self.table_scene.build()

        self.objects = []
        for i, step in enumerate(self.encounter_steps):
            obs = step["observation"]
            asset = self.asset_infos[i]
            initial_pose = sapien.Pose(p=self.inactive_xyz)
            name = f"dejavu_{i}_of{asset['objectfolder_id']}_{obs['object_id']}"
            builder = self.scene.create_actor_builder()
            mesh_path = str(asset["model_path"])
            bounds_min = asset["bounds_min"]
            bounds_max = asset["bounds_max"]
            bbox_center = 0.5 * (bounds_min + bounds_max) * OBJECTFOLDER_SCALE
            bbox_half_size = 0.5 * (bounds_max - bounds_min) * OBJECTFOLDER_SCALE
            builder.add_box_collision(
                pose=sapien.Pose(p=bbox_center),
                half_size=np.maximum(bbox_half_size, 0.01),
                density=300,
            )
            builder.add_visual_from_file(filename=mesh_path, scale=[OBJECTFOLDER_SCALE] * 3)
            builder.set_initial_pose(initial_pose)
            obj = builder.build(name=name)
            self.objects.append(obj)

        self.contact_site = actors.build_sphere(
            self.scene,
            radius=0.008,
            color=[1.0, 0.15, 0.05, 1.0],
            name="target_contact_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=self.inactive_xyz),
        )
        self.target_site = actors.build_sphere(
            self.scene,
            radius=0.012,
            color=[0.1, 1.0, 0.15, 1.0],
            name="memory_target_marker",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=self.inactive_xyz),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self.active_encounter_index = 0
            self._reset_robot(len(env_idx))
            self._place_all_objects()
            self._update_markers()

    def _get_obs_extra(self, info: dict):
        active_step = self.active_step
        if active_step is None:
            active_idx = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.int32)
            target_idx = torch.full((self.num_envs,), self.query_step["target_encounter_index"], device=self.device, dtype=torch.int32)
            return {"active_encounter_index": active_idx, "target_encounter_index": target_idx}

        obs = active_step["observation"]
        props = obs["properties"]
        active_idx = torch.full((self.num_envs,), active_step["encounter_index"], device=self.device, dtype=torch.int32)
        target_idx = torch.full((self.num_envs,), self.query_step["target_encounter_index"], device=self.device, dtype=torch.int32)
        return {
            "active_encounter_index": active_idx,
            "target_encounter_index": target_idx,
            "is_target": torch.full((self.num_envs,), bool(active_step["is_target"]), device=self.device),
            "is_distractor": torch.full((self.num_envs,), bool(active_step["is_distractor"]), device=self.device),
            "mass": torch.full((self.num_envs,), float(props["mass"]), device=self.device),
            "hardness": torch.full((self.num_envs,), float(props["hardness"]), device=self.device),
            "contact_point": torch.tensor(obs["contact_point"], device=self.device).repeat(self.num_envs, 1),
        }

    def evaluate(self):
        return {}

    def advance_encounter(self) -> bool:
        """Move to the next generated encounter. Returns False at query time."""

        if self.active_encounter_index + 1 >= len(self.encounter_steps):
            self.active_encounter_index = len(self.encounter_steps)
            self._hide_markers()
            return False
        self.active_encounter_index += 1
        self._update_markers()
        return True

    def print_current_step(self) -> None:
        active_step = self.active_step
        if active_step is None:
            print("QUERY:", self.query_step["query_text"])
            print("ANSWER:", repr(self.query_step["ground_truth_answer"]))
            return

        obs = active_step["observation"]
        props = obs["properties"]
        flags = []
        if active_step["is_target"]:
            flags.append("TARGET")
        if active_step["is_distractor"]:
            flags.append("DISTRACTOR")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        print(
            f"Encounter {active_step['encounter_index']}: {obs['object_id']} "
            f"(ObjectFolder {obs['objectfolder_id']}){suffix} | "
            f"mass={props['mass']:.3f}kg, material={props['material']}, "
            f"hardness={props['hardness']:.3f}, texture={props['texture']}"
        )

    def _reset_robot(self, batch_size: int) -> None:
        init_qpos = torch.tensor(
            [0.0, 0.15, 0.0, -2.10, 0.0, 2.30, 0.78, 0.04, 0.04],
            dtype=torch.float32,
            device=self.device,
        )
        self.agent.reset(init_qpos.repeat(batch_size, 1))

    def _hide_markers(self) -> None:
        hidden_pose = Pose.create_from_pq(
            p=torch.tensor(self.inactive_xyz, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        )
        self.contact_site.set_pose(hidden_pose)
        self.target_site.set_pose(hidden_pose)

    def _place_all_objects(self) -> None:
        for i, obj in enumerate(self.objects):
            table_xyz = self.encounter_position(i)
            xyz = torch.tensor(
                self.asset_pose_position(i, table_xyz[:2]),
                dtype=torch.float32,
                device=self.device,
            ).repeat(self.num_envs, 1)
            pose = Pose.create_from_pq(p=xyz, q=[1.0, 0.0, 0.0, 0.0])
            obj.set_pose(pose)

    def _update_markers(self) -> None:
        self._hide_markers()
        active_step = self.active_step
        if active_step is None:
            return

        object_xyz = self.grasp_position(self.active_encounter_index)
        contact = np.asarray(active_step["observation"]["contact_point"], dtype=np.float32)
        surface_contact = np.clip(contact * OBJECTFOLDER_SCALE, -self.object_half_size, self.object_half_size)
        if np.linalg.norm(surface_contact) < 1e-6:
            surface_contact = np.array([self.object_half_size, 0.0, 0.0], dtype=np.float32)
        marker_xyz = object_xyz + surface_contact
        marker_pose = Pose.create_from_pq(
            p=torch.tensor(marker_xyz, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        )
        self.contact_site.set_pose(marker_pose)

        if active_step["is_target"]:
            target_pose = Pose.create_from_pq(
                p=torch.tensor(object_xyz + np.array([0.0, 0.0, 0.09], dtype=np.float32), dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
            )
        else:
            target_pose = Pose.create_from_pq(
                p=torch.tensor(self.inactive_xyz, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
            )
        self.target_site.set_pose(target_pose)
