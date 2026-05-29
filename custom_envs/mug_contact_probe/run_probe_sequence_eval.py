#!/usr/bin/env python3
"""Probe-level tactile evaluation with UniTouch embeddings.

Generates 8 encounters per object, 12 tactile frames per encounter, then evaluates:
1) Probe representation = mean pooled embedding.
2) Probe representation = concat(mean, std) embedding.
Also reruns single-frame evaluation with high-curvature contact sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from custom_envs.mug_contact_probe.objectfolder_query import (
    TOUCH_DEPTH_MAX_M,
    TOUCH_DEPTH_MIN_M,
    TOUCH_THETA_MAX_RAD,
    ObjectFolderModalityRenderer,
)
from custom_envs.mug_contact_probe.run_tactile_unitouch_tsne import (
    OBJECTS_DIR,
    _ensure_object_folder,
    _load_frozen_unitouch,
    _preprocess_tactile_image,
)

try:
    import umap
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "Missing dependency `umap-learn`. Install with:\n"
        "  .venv-ms3-of2/bin/pip install umap-learn"
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
OBJECTS_CSV = ROOT / "third_party" / "ObjectFolder" / "objects.csv"
DEFAULT_OBJECT_IDS = (21, 23, 29, 30, 36)


@dataclass(frozen=True)
class ObjectMeta:
    object_id: int
    name: str
    material: str


@dataclass(frozen=True)
class FrameSample:
    object_id: int
    encounter_id: int
    frame_id: int
    contact_xyz_local: np.ndarray
    orientation_theta_phi: np.ndarray
    press_depth: float
    tactile_rgb: np.ndarray


@dataclass
class MeshData:
    vertices: np.ndarray
    high_curvature_indices: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probe-level tactile sequence evaluation.")
    parser.add_argument(
        "--object-ids",
        type=str,
        default=",".join(str(x) for x in DEFAULT_OBJECT_IDS),
        help="Comma-separated ObjectFolder ids.",
    )
    parser.add_argument("--encounters-per-object", type=int, default=8, help="Encounter count per object.")
    parser.add_argument("--frames-per-encounter", type=int, default=12, help="Frames per encounter.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--knn-k", type=int, default=3, help="k for kNN.")
    parser.add_argument(
        "--curvature-quantile",
        type=float,
        default=0.80,
        help="Top quantile of curvature scores used for edge/high-curvature sampling.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=20.0,
        help="t-SNE perplexity for visualization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for UniTouch embedding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "mug_contact_probe" / "probe_sequence_eval_8x12",
        help="Output directory.",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=ROOT / "runs" / "mug_contact_probe" / "tactile_unitouch_tsne_64" / "results.json",
        help="Path to prior single-frame baseline results.json (for comparison).",
    )
    return parser.parse_args()


def _load_object_metadata() -> dict[int, ObjectMeta]:
    metas: dict[int, ObjectMeta] = {}
    with OBJECTS_CSV.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            oid = int(row[0])
            metas[oid] = ObjectMeta(object_id=oid, name=row[1], material=row[3])
    return metas


def _parse_obj_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []
                for p in parts:
                    idx = int(p.split("/")[0]) - 1
                    face.append(idx)
                if len(face) >= 3:
                    faces.append(face)
    if not vertices:
        raise RuntimeError(f"No vertices found in {obj_path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=object)


def _curvature_indices(vertices: np.ndarray, faces_obj: np.ndarray, quantile: float) -> np.ndarray:
    n = len(vertices)
    neighbors: list[set[int]] = [set() for _ in range(n)]
    vert_normal_sum = np.zeros((n, 3), dtype=np.float64)

    # Triangulate polygon faces by fan and aggregate normals.
    for face in faces_obj:
        face = list(face)
        for i in range(1, len(face) - 1):
            tri = (face[0], face[i], face[i + 1])
            a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            nrm = np.cross(b - a, c - a).astype(np.float64)
            norm = float(np.linalg.norm(nrm))
            if norm < 1e-10:
                continue
            nrm /= norm
            for v in tri:
                vert_normal_sum[v] += nrm
            u, v, w = tri
            neighbors[u].update((v, w))
            neighbors[v].update((u, w))
            neighbors[w].update((u, v))

    norms = np.linalg.norm(vert_normal_sum, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    vert_normals = vert_normal_sum / norms

    # Curvature proxy: mean angular deviation of normal from neighbor normals.
    scores = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        if not neighbors[i]:
            continue
        nbr = np.fromiter(neighbors[i], dtype=np.int64)
        dots = np.einsum("ij,j->i", vert_normals[nbr], vert_normals[i])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        scores[i] = float(np.mean(angles))

    th = float(np.quantile(scores, quantile))
    idx = np.where(scores >= th)[0]
    if len(idx) == 0:
        return np.arange(n, dtype=np.int64)
    return idx.astype(np.int64)


def _prepare_mesh_data(object_ids: Sequence[int], quantile: float) -> dict[int, MeshData]:
    mesh_data: dict[int, MeshData] = {}
    for oid in object_ids:
        object_dir = _ensure_object_folder(oid)
        verts, faces = _parse_obj_mesh(object_dir / "model.obj")
        high_idx = _curvature_indices(verts, faces, quantile=quantile)
        mesh_data[oid] = MeshData(vertices=verts, high_curvature_indices=high_idx)
    return mesh_data


def _sample_indices(
    rng: np.random.Generator,
    n: int,
    n_vertices: int,
    mode: str,
    high_curvature_indices: np.ndarray,
) -> np.ndarray:
    if mode == "uniform":
        pool = np.arange(n_vertices, dtype=np.int64)
    elif mode == "high_curvature":
        pool = high_curvature_indices
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

    replace = len(pool) < n
    return rng.choice(pool, size=n, replace=replace)


def _collect_frames(
    object_ids: Sequence[int],
    encounters_per_object: int,
    frames_per_encounter: int,
    sampling_mode: str,
    mesh_data: dict[int, MeshData],
    seed: int,
) -> list[FrameSample]:
    rng = np.random.default_rng(seed)
    frames: list[FrameSample] = []

    for oid in object_ids:
        object_dir = OBJECTS_DIR / str(oid)
        renderer = ObjectFolderModalityRenderer(object_dir / "ObjectFile.pth")
        mesh = mesh_data[oid]

        for enc in range(encounters_per_object):
            idxs = _sample_indices(
                rng,
                n=frames_per_encounter,
                n_vertices=len(mesh.vertices),
                mode=sampling_mode,
                high_curvature_indices=mesh.high_curvature_indices,
            )
            for frame_id, vidx in enumerate(idxs):
                contact = mesh.vertices[int(vidx)]
                theta = float(rng.uniform(0.0, TOUCH_THETA_MAX_RAD))
                phi = float(rng.uniform(0.0, 2.0 * np.pi))
                depth = float(rng.uniform(TOUCH_DEPTH_MIN_M, TOUCH_DEPTH_MAX_M))
                touch = renderer.make_touch_spec(
                    contact_xyz_local=contact,
                    orientation=np.array([theta, phi], dtype=np.float32),
                    press_depth=depth,
                )
                tactile = renderer.render_tactile(touch)
                frames.append(
                    FrameSample(
                        object_id=oid,
                        encounter_id=enc,
                        frame_id=frame_id,
                        contact_xyz_local=contact.copy(),
                        orientation_theta_phi=np.array([theta, phi], dtype=np.float32),
                        press_depth=depth,
                        tactile_rgb=tactile,
                    )
                )
    return frames


def _embed_frames(frames: Sequence[FrameSample], device: torch.device, batch_size: int) -> np.ndarray:
    model, touch_key = _load_frozen_unitouch(device=device)
    tensors = [_preprocess_tactile_image(f.tactile_rgb) for f in frames]

    outs = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size], dim=0).to(device)
            emb = model({touch_key: batch})[touch_key].detach().cpu().numpy()
            outs.append(emb)
    return np.concatenate(outs, axis=0).astype(np.float32)


def _eval_knn_and_silhouette(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    sil = float(silhouette_score(X, y))
    return {
        "knn_accuracy_mean": float(np.mean(scores)),
        "knn_accuracy_std": float(np.std(scores)),
        "silhouette": sil,
    }


def _make_probe_representations(
    embeddings: np.ndarray,
    frames: Sequence[FrameSample],
    object_ids: Sequence[int],
    encounters_per_object: int,
    frames_per_encounter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_list = []
    mean_std_list = []
    labels = []

    # Frames are generated in deterministic order: object -> encounter -> frame.
    cursor = 0
    for oid in object_ids:
        for _enc in range(encounters_per_object):
            probe_emb = embeddings[cursor : cursor + frames_per_encounter]
            cursor += frames_per_encounter
            mu = probe_emb.mean(axis=0)
            sd = probe_emb.std(axis=0)
            mean_list.append(mu)
            mean_std_list.append(np.concatenate([mu, sd], axis=0))
            labels.append(oid)

    return (
        np.asarray(mean_list, dtype=np.float32),
        np.asarray(mean_std_list, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def _plot_tsne_umap_2d(
    X: np.ndarray,
    y: np.ndarray,
    class_names: dict[int, str],
    out_path: Path,
    seed: int,
    perplexity: float,
    title_prefix: str,
) -> None:
    n = len(X)
    tsne_perp = float(min(perplexity, max(2.0, n - 1.0)))
    tsne2 = TSNE(
        n_components=2,
        perplexity=tsne_perp,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    ).fit_transform(X)
    umap2 = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(5, n // 4)),
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    ).fit_transform(X)

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
    cmap = plt.colormaps.get_cmap("tab10")
    object_ids = sorted(class_names.keys())

    for ax, points, title in [
        (axs[0], tsne2, f"{title_prefix}: t-SNE (2D)"),
        (axs[1], umap2, f"{title_prefix}: UMAP (2D)"),
    ]:
        for i, oid in enumerate(object_ids):
            m = y == oid
            ax.scatter(
                points[m, 0],
                points[m, 1],
                s=16,
                alpha=0.68,
                color=cmap(i / max(1, len(object_ids) - 1)),
                edgecolors="black",
                linewidths=0.20,
                label=f"{oid}: {class_names[oid]}",
            )
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.grid(alpha=0.2)

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True, fontsize=8)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_path, dpi=360)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    object_ids = [int(x.strip()) for x in args.object_ids.split(",") if x.strip()]
    if len(object_ids) != 5:
        raise ValueError(f"Expected exactly 5 object ids, got {len(object_ids)}: {object_ids}")

    metas = _load_object_metadata()
    class_names = {oid: metas[oid].name for oid in object_ids}
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_data = _prepare_mesh_data(object_ids, quantile=args.curvature_quantile)

    # Uniform sampling for probe representations: 5 * 8 * 12 = 480 frames.
    uniform_frames = _collect_frames(
        object_ids=object_ids,
        encounters_per_object=args.encounters_per_object,
        frames_per_encounter=args.frames_per_encounter,
        sampling_mode="uniform",
        mesh_data=mesh_data,
        seed=args.seed,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but CUDA is unavailable.")
    uniform_emb = _embed_frames(uniform_frames, device=device, batch_size=args.batch_size)
    uniform_labels = np.asarray([f.object_id for f in uniform_frames], dtype=np.int32)

    probe_mean, probe_meanstd, probe_labels = _make_probe_representations(
        uniform_emb,
        uniform_frames,
        object_ids=object_ids,
        encounters_per_object=args.encounters_per_object,
        frames_per_encounter=args.frames_per_encounter,
    )

    probe_mean_metrics = _eval_knn_and_silhouette(probe_mean, probe_labels, k=args.knn_k, seed=args.seed)
    probe_meanstd_metrics = _eval_knn_and_silhouette(probe_meanstd, probe_labels, k=args.knn_k, seed=args.seed)

    # High-curvature-only sampling for single-frame rerun.
    edge_frames = _collect_frames(
        object_ids=object_ids,
        encounters_per_object=args.encounters_per_object,
        frames_per_encounter=args.frames_per_encounter,
        sampling_mode="high_curvature",
        mesh_data=mesh_data,
        seed=args.seed + 1000,
    )
    edge_emb = _embed_frames(edge_frames, device=device, batch_size=args.batch_size)
    edge_labels = np.asarray([f.object_id for f in edge_frames], dtype=np.int32)
    edge_single_metrics = _eval_knn_and_silhouette(edge_emb, edge_labels, k=args.knn_k, seed=args.seed)

    baseline = {}
    if args.baseline_results.exists():
        baseline_json = json.loads(args.baseline_results.read_text())
        baseline = {
            "path": str(args.baseline_results),
            "single_frame_uniform_accuracy_mean": float(baseline_json["knn"]["accuracy_mean"]),
            "single_frame_uniform_accuracy_std": float(baseline_json["knn"]["accuracy_std"]),
            "single_frame_uniform_silhouette": float(baseline_json["silhouette_score"]),
        }

    _plot_tsne_umap_2d(
        probe_mean,
        probe_labels,
        class_names=class_names,
        out_path=out_dir / "probe_mean_tsne_umap_2d.png",
        seed=args.seed,
        perplexity=args.tsne_perplexity,
        title_prefix="Probe Mean",
    )
    _plot_tsne_umap_2d(
        probe_meanstd,
        probe_labels,
        class_names=class_names,
        out_path=out_dir / "probe_meanstd_tsne_umap_2d.png",
        seed=args.seed,
        perplexity=args.tsne_perplexity,
        title_prefix="Probe Mean+Std",
    )
    _plot_tsne_umap_2d(
        edge_emb,
        edge_labels,
        class_names=class_names,
        out_path=out_dir / "single_frame_high_curvature_tsne_umap_2d.png",
        seed=args.seed,
        perplexity=args.tsne_perplexity,
        title_prefix="Single-Frame High-Curvature",
    )

    report = {
        "config": {
            "object_ids": object_ids,
            "encounters_per_object": int(args.encounters_per_object),
            "frames_per_encounter": int(args.frames_per_encounter),
            "total_frames_uniform": int(len(uniform_frames)),
            "total_frames_edge": int(len(edge_frames)),
            "knn_k": int(args.knn_k),
            "cv_folds": 5,
            "curvature_quantile": float(args.curvature_quantile),
        },
        "probe_eval_mean": probe_mean_metrics,
        "probe_eval_mean_std": probe_meanstd_metrics,
        "single_frame_high_curvature": edge_single_metrics,
        "baseline_single_frame_uniform": baseline,
        "did_high_curvature_improve_vs_baseline": (
            None
            if not baseline
            else bool(edge_single_metrics["knn_accuracy_mean"] > baseline["single_frame_uniform_accuracy_mean"])
        ),
        "artifacts": {
            "probe_mean_plot": str(out_dir / "probe_mean_tsne_umap_2d.png"),
            "probe_meanstd_plot": str(out_dir / "probe_meanstd_tsne_umap_2d.png"),
            "single_frame_high_curvature_plot": str(out_dir / "single_frame_high_curvature_tsne_umap_2d.png"),
        },
    }

    np.save(out_dir / "uniform_frame_embeddings.npy", uniform_emb)
    np.save(out_dir / "uniform_frame_labels.npy", uniform_labels)
    np.save(out_dir / "probe_mean_embeddings.npy", probe_mean)
    np.save(out_dir / "probe_meanstd_embeddings.npy", probe_meanstd)
    np.save(out_dir / "probe_labels.npy", probe_labels)
    np.save(out_dir / "edge_frame_embeddings.npy", edge_emb)
    np.save(out_dir / "edge_frame_labels.npy", edge_labels)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("=== Probe Sequence Evaluation ===")
    print(f"Frames generated (uniform): {len(uniform_frames)}")
    print(f"Frames generated (high-curvature): {len(edge_frames)}")
    print(
        f"Probe mean: kNN={probe_mean_metrics['knn_accuracy_mean']:.4f} "
        f"+/- {probe_mean_metrics['knn_accuracy_std']:.4f}, "
        f"silhouette={probe_mean_metrics['silhouette']:.4f}"
    )
    print(
        f"Probe mean+std: kNN={probe_meanstd_metrics['knn_accuracy_mean']:.4f} "
        f"+/- {probe_meanstd_metrics['knn_accuracy_std']:.4f}, "
        f"silhouette={probe_meanstd_metrics['silhouette']:.4f}"
    )
    print(
        f"Single-frame high-curvature: kNN={edge_single_metrics['knn_accuracy_mean']:.4f} "
        f"+/- {edge_single_metrics['knn_accuracy_std']:.4f}, "
        f"silhouette={edge_single_metrics['silhouette']:.4f}"
    )
    if baseline:
        print(
            "Baseline single-frame uniform: "
            f"{baseline['single_frame_uniform_accuracy_mean']:.4f} +/- "
            f"{baseline['single_frame_uniform_accuracy_std']:.4f}"
        )
        delta = edge_single_metrics["knn_accuracy_mean"] - baseline["single_frame_uniform_accuracy_mean"]
        print(f"High-curvature improvement vs baseline: {delta:+.4f}")
    print(f"Saved report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
