"""DejaVu episodic memory benchmark scaffold for ManiSkill-style probes.

This module intentionally does not integrate real sensors, encoders, or physics.
It generates the encounter/query sequence and ground-truth logs that will later
be backed by ManiSkill3 contact probing and tactile perception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


Task = Literal["mass", "material", "hardness", "texture"]


@dataclass(frozen=True)
class ObjectSpec:
    """Synthetic object metadata used until real assets/perception are wired in."""

    object_id: str
    mass: float
    material: str
    hardness: float
    texture: str

    def properties(self) -> dict[str, Any]:
        return {
            "mass": self.mass,
            "material": self.material,
            "hardness": self.hardness,
            "texture": self.texture,
        }


class DejaVuEnv:
    """Generate one DejaVu memory-benchmark episode.

    Args:
        task: Property queried at the end of the episode.
        H: Number of encounter timesteps between the target encounter and query.
        D: Number of intervening encounters similar to the target in ``task``.
        seed: Seed controlling all object properties, ordering, contacts, and embeddings.
    """

    SUPPORTED_TASKS: tuple[Task, ...] = ("mass", "material", "hardness", "texture")
    EMBEDDING_DIM = 2048

    _MATERIALS = ("ceramic", "metal", "plastic", "wood", "rubber", "glass")
    _TEXTURES = ("smooth", "ribbed", "matte", "glossy", "rough", "woven")
    _NUMERIC_SIMILARITY = {"mass": 0.04, "hardness": 0.05}

    def __init__(self, task: str, H: int, D: int, seed: int):
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"task must be one of {self.SUPPORTED_TASKS}, got {task!r}")
        if H < 0:
            raise ValueError(f"H must be non-negative, got {H}")
        if D < 0:
            raise ValueError(f"D must be non-negative, got {D}")
        if D > H:
            raise ValueError(f"D must be <= H because distractors are intervening encounters, got D={D}, H={H}")

        self.task: Task = task  # type: ignore[assignment]
        self.H = int(H)
        self.D = int(D)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.target_encounter_index = int(self.rng.integers(0, 3))
        self.episode = self.generate_episode()
        self.validate_episode(self.episode)

    def generate_episode(self) -> list[dict[str, Any]]:
        """Return an episode as encounters followed by one final query timestep."""

        pre_target_count = self.target_encounter_index
        target = self._sample_object("target")

        prefix = [self._sample_non_distractor_object(target, f"prefix_{i}") for i in range(pre_target_count)]
        intervening = self._sample_intervening_objects(target)
        encounters = prefix + [target] + intervening

        timesteps = [
            {
                "type": "encounter",
                "encounter_index": i,
                "observation": self._make_observation(obj),
                "is_target": i == self.target_encounter_index,
                "is_distractor": self._is_distractor(obj, target) and i > self.target_encounter_index,
            }
            for i, obj in enumerate(encounters)
        ]

        timesteps.append(self._make_query_timestep(target))
        return timesteps

    def print_episode(self, episode: list[dict[str, Any]] | None = None) -> None:
        """Print a compact, readable view of an episode."""

        episode = self.episode if episode is None else episode
        query = episode[-1]
        print(f"DejaVuEnv(task={self.task!r}, H={self.H}, D={self.D}, seed={self.seed})")
        for step in episode[:-1]:
            obs = step["observation"]
            props = obs["properties"]
            flags = []
            if step["is_target"]:
                flags.append("TARGET")
            if step["is_distractor"]:
                flags.append("DISTRACTOR")
            suffix = f" [{' '.join(flags)}]" if flags else ""
            asset_suffix = ""
            if "objectfolder_id" in obs:
                asset_suffix = f" (ObjectFolder {obs['objectfolder_id']})"
            print(
                f"  encounter {step['encounter_index']}: {obs['object_id']}{asset_suffix}{suffix} | "
                f"mass={props['mass']:.3f}kg, material={props['material']}, "
                f"hardness={props['hardness']:.3f}, texture={props['texture']}, "
                f"contact={np.array2string(obs['contact_point'], precision=3)}, "
                f"embedding_shape={obs['tactile_embedding'].shape}"
            )
        print(
            f"  query: {query['query_text']}\n"
            f"    target_encounter_index={query['target_encounter_index']}, "
            f"ground_truth_answer={query['ground_truth_answer']!r}"
        )

    def validate_episode(self, episode: list[dict[str, Any]] | None = None) -> None:
        """Assert the H/D controls and ground-truth answer are internally correct."""

        episode = self.episode if episode is None else episode
        assert episode[-1]["type"] == "query", "final timestep must be the query"

        query_index = len(episode) - 1
        target_index = int(episode[-1]["target_encounter_index"])
        assert query_index - target_index - 1 == self.H, (
            f"expected H={self.H} steps between target and query, "
            f"got {query_index - target_index - 1}"
        )

        target_obs = episode[target_index]["observation"]
        target_props = target_obs["properties"]
        intervening = episode[target_index + 1 : query_index]
        distractor_count = sum(
            self._properties_are_similar(step["observation"]["properties"], target_props)
            for step in intervening
        )
        assert distractor_count == self.D, f"expected D={self.D} distractors, got {distractor_count}"

        expected_answer = target_props[self.task]
        assert episode[-1]["ground_truth_answer"] == expected_answer, (
            f"ground-truth answer mismatch: expected {expected_answer!r}, "
            f"got {episode[-1]['ground_truth_answer']!r}"
        )

    def _sample_intervening_objects(self, target: ObjectSpec) -> list[ObjectSpec]:
        distractor_positions = set(self.rng.choice(self.H, size=self.D, replace=False).tolist())
        objects = []
        for i in range(self.H):
            if i in distractor_positions:
                objects.append(self._sample_distractor_object(target, f"distractor_{i}"))
            else:
                objects.append(self._sample_non_distractor_object(target, f"filler_{i}"))
        return objects

    def _sample_object(self, label: str) -> ObjectSpec:
        return ObjectSpec(
            object_id=self._object_id(label),
            mass=float(self.rng.uniform(0.10, 2.00)),
            material=str(self.rng.choice(self._MATERIALS)),
            hardness=float(self.rng.uniform(0.05, 0.95)),
            texture=str(self.rng.choice(self._TEXTURES)),
        )

    def _sample_distractor_object(self, target: ObjectSpec, label: str) -> ObjectSpec:
        obj = self._sample_object(label)
        props = obj.properties()
        target_props = target.properties()
        if self.task in self._NUMERIC_SIMILARITY:
            tol = self._NUMERIC_SIMILARITY[self.task]
            props[self.task] = self._near(float(target_props[self.task]), tol)
        else:
            props[self.task] = target_props[self.task]
        return ObjectSpec(object_id=obj.object_id, **props)

    def _sample_non_distractor_object(self, target: ObjectSpec, label: str) -> ObjectSpec:
        target_props = target.properties()
        for _ in range(100):
            obj = self._sample_object(label)
            if not self._properties_are_similar(obj.properties(), target_props):
                return obj
        raise RuntimeError("failed to sample a non-distractor object")

    def _make_observation(self, obj: ObjectSpec) -> dict[str, Any]:
        return {
            "object_id": obj.object_id,
            "properties": obj.properties(),
            "contact_point": self.rng.uniform(-0.05, 0.05, size=3).astype(np.float32),
            "tactile_embedding": self.rng.normal(0.0, 1.0, size=self.EMBEDDING_DIM).astype(np.float32),
        }

    def _make_query_timestep(self, target: ObjectSpec) -> dict[str, Any]:
        answer = target.properties()[self.task]
        return {
            "type": "query",
            "query_text": f"What was the {self.task} of the object you touched at encounter {self.target_encounter_index}?",
            "target_encounter_index": self.target_encounter_index,
            "ground_truth_answer": answer,
        }

    def _is_distractor(self, obj: ObjectSpec, target: ObjectSpec) -> bool:
        return self._properties_are_similar(obj.properties(), target.properties())

    def _properties_are_similar(self, props: dict[str, Any], target_props: dict[str, Any]) -> bool:
        if self.task in self._NUMERIC_SIMILARITY:
            tol = self._NUMERIC_SIMILARITY[self.task]
            return abs(float(props[self.task]) - float(target_props[self.task])) <= tol
        return props[self.task] == target_props[self.task]

    def _near(self, value: float, tolerance: float) -> float:
        low = max(0.01, value - tolerance * 0.8)
        high = value + tolerance * 0.8
        return float(self.rng.uniform(low, high))

    def _object_id(self, label: str) -> str:
        return f"obj_{label}_{int(self.rng.integers(10_000, 99_999))}"


def print_episode(env_or_episode: DejaVuEnv | list[dict[str, Any]]) -> None:
    """Convenience function for readable episode printing."""

    if isinstance(env_or_episode, DejaVuEnv):
        env_or_episode.print_episode()
        return

    for step in env_or_episode:
        print(step)


def _demo() -> None:
    examples = [
        DejaVuEnv(task="mass", H=2, D=0, seed=11),
        DejaVuEnv(task="material", H=4, D=2, seed=22),
        DejaVuEnv(task="texture", H=6, D=3, seed=33),
    ]
    for i, env in enumerate(examples, start=1):
        print(f"\n=== Example {i} ===")
        env.print_episode()
        env.validate_episode()


if __name__ == "__main__":
    _demo()
