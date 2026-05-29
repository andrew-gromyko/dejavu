#!/usr/bin/env python3
"""ObjectFolder tactile clustering experiment with frozen UniTouch embeddings.

Experiment:
1) Select 5 YCB objects from ObjectFolder 2.0 (first-100 archive).
2) Render tactile images with TouchNet at varied contact points.
3) Embed tactile images using frozen UniTouch touch encoder.
4) Plot t-SNE and report kNN classification accuracy across objects.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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


ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = ROOT / "assets" / "objectfolder2"
OBJECTS_CSV = ROOT / "third_party" / "ObjectFolder" / "objects.csv"
UNITOUCH_ROOT = ROOT / "third_party" / "UniTouch"
UNITOUCH_CKPT = ROOT / "assets" / "unitouch" / "last_new.ckpt"
OBJECTS_ARCHIVE = ASSETS_ROOT / "ObjectFolder1-100.tar.gz"
OBJECTS_DIR = ASSETS_ROOT / "ObjectFolder1-100"

# Five YCB objects with visibly different surfaces/material metadata.
DEFAULT_OBJECT_IDS = (21, 23, 29, 30, 36)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class ObjectMeta:
    object_id: int
    name: str
    material: str
    url: str


@dataclass(frozen=True)
class TactileSample:
    object_id: int
    object_name: str
    material: str
    contact_xyz_local: np.ndarray
    orientation_theta_phi: np.ndarray
    press_depth: float
    tactile_rgb: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tactile object clustering with TouchNet + UniTouch.")
    parser.add_argument(
        "--object-ids",
        type=str,
        default=",".join(str(x) for x in DEFAULT_OBJECT_IDS),
        help="Comma-separated ObjectFolder ids (must exist in ObjectFolder1-100 archive).",
    )
    parser.add_argument(
        "--samples-per-object",
        type=int,
        default=10,
        help="Number of tactile renders per object.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--knn-k",
        type=int,
        default=3,
        help="k for kNN classifier.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=12.0,
        help="t-SNE perplexity (must be < number of samples).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for UniTouch embedding inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Torch device for UniTouch embedding inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "mug_contact_probe" / "tactile_unitouch_tsne",
        help="Output directory for plots/results.",
    )
    return parser.parse_args()


def _load_object_metadata() -> dict[int, ObjectMeta]:
    metas: dict[int, ObjectMeta] = {}
    with OBJECTS_CSV.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            oid = int(row[0])
            metas[oid] = ObjectMeta(object_id=oid, name=row[1], material=row[3], url=row[4])
    return metas


def _safe_member_for_object(member_name: str, object_id: int) -> bool:
    prefix = f"ObjectFolder1-100/{object_id}/"
    if not member_name.startswith(prefix):
        return False
    if ".." in member_name or member_name.startswith("/"):
        return False
    return True


def _ensure_object_folder(object_id: int) -> Path:
    object_dir = OBJECTS_DIR / str(object_id)
    object_file = object_dir / "ObjectFile.pth"
    model_obj = object_dir / "model.obj"
    if object_file.exists() and model_obj.exists():
        return object_dir

    if not OBJECTS_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Missing object archive: {OBJECTS_ARCHIVE}. "
            "Expected ObjectFolder1-100.tar.gz in assets/objectfolder2."
        )

    print(f"[info] Extracting ObjectFolder object {object_id} from archive...")
    with tarfile.open(OBJECTS_ARCHIVE, "r:gz") as tar:
        members = [m for m in tar.getmembers() if _safe_member_for_object(m.name, object_id)]
        if not members:
            raise RuntimeError(f"Object {object_id} not found in {OBJECTS_ARCHIVE.name}")
        tar.extractall(path=ASSETS_ROOT, members=members)

    if not object_file.exists() or not model_obj.exists():
        raise RuntimeError(f"Failed to extract required files for object {object_id}")
    return object_dir


def _load_obj_vertices(obj_path: Path) -> np.ndarray:
    vertices = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise RuntimeError(f"No vertices found in {obj_path}")
    return np.asarray(vertices, dtype=np.float32)


def _collect_tactile_samples(
    object_ids: Sequence[int],
    object_metas: dict[int, ObjectMeta],
    samples_per_object: int,
    seed: int,
) -> list[TactileSample]:
    rng = np.random.default_rng(seed)
    samples: list[TactileSample] = []

    for object_id in object_ids:
        meta = object_metas[object_id]
        object_dir = _ensure_object_folder(object_id)
        object_file = object_dir / "ObjectFile.pth"
        vertices = _load_obj_vertices(object_dir / "model.obj")
        renderer = ObjectFolderModalityRenderer(object_file)

        replace = len(vertices) < samples_per_object
        contact_indices = rng.choice(len(vertices), size=samples_per_object, replace=replace)
        for idx in contact_indices:
            contact_xyz_local = vertices[idx]
            theta = float(rng.uniform(0.0, TOUCH_THETA_MAX_RAD))
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            depth = float(rng.uniform(TOUCH_DEPTH_MIN_M, TOUCH_DEPTH_MAX_M))

            touch = renderer.make_touch_spec(
                contact_xyz_local=contact_xyz_local,
                orientation=np.array([theta, phi], dtype=np.float32),
                press_depth=depth,
            )
            tactile_rgb = renderer.render_tactile(touch)
            samples.append(
                TactileSample(
                    object_id=object_id,
                    object_name=meta.name,
                    material=meta.material,
                    contact_xyz_local=contact_xyz_local.copy(),
                    orientation_theta_phi=np.array([theta, phi], dtype=np.float32),
                    press_depth=depth,
                    tactile_rgb=tactile_rgb,
                )
            )
        print(
            f"[info] Rendered {samples_per_object} tactile samples for {object_id}: "
            f"{meta.name} ({meta.material})"
        )

    return samples


def _preprocess_tactile_image(img: np.ndarray) -> torch.Tensor:
    pil_img = Image.fromarray(img.astype(np.uint8), mode="RGB")
    width, height = pil_img.size
    scale = 224.0 / float(min(width, height))
    resized_size = (int(width * scale), int(height * scale))
    pil_img = pil_img.resize(resized_size, resample=Image.Resampling.BICUBIC)

    left = (pil_img.size[0] - 224) // 2
    top = (pil_img.size[1] - 224) // 2
    pil_img = pil_img.crop((left, top, left + 224, top + 224))

    tensor = torch.from_numpy(np.asarray(pil_img, dtype=np.float32)).permute(2, 0, 1) / 255.0
    tensor = (tensor - CLIP_MEAN) / CLIP_STD
    return tensor


def _load_frozen_unitouch(device: torch.device) -> tuple[torch.nn.Module, str]:
    if str(UNITOUCH_ROOT) not in sys.path:
        sys.path.insert(0, str(UNITOUCH_ROOT))

    from ImageBind.models.x2touch_model_part import ModalityType, x2touch

    model = x2touch(pretrained=False)
    ckpt = torch.load(UNITOUCH_CKPT, map_location="cpu", weights_only=False)
    state_dict_raw = ckpt["state_dict"]

    state_dict = {}
    for key, value in state_dict_raw.items():
        if key.startswith("model."):
            state_dict[key[len("model.") :]] = value
        elif key.startswith("module.model."):
            state_dict[key[len("module.model.") :]] = value
        elif key.startswith("module."):
            state_dict[key[len("module.") :]] = value
        else:
            state_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        raise RuntimeError(f"UniTouch checkpoint load failed, missing keys: {missing_keys[:8]}")

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model, ModalityType.TOUCH


def _embed_tactile_samples(
    samples: Sequence[TactileSample],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model, touch_key = _load_frozen_unitouch(device=device)
    tensors = [_preprocess_tactile_image(s.tactile_rgb) for s in samples]

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size], dim=0).to(device)
            outputs = model({touch_key: batch})
            emb = outputs[touch_key].detach().cpu().numpy()
            embeddings.append(emb)
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def _save_tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
    perplexity: float,
    seed: int,
    out_path: Path,
) -> np.ndarray:
    max_perplexity = max(2.0, float(len(embeddings) - 1))
    eff_perplexity = float(min(perplexity, max_perplexity))

    tsne = TSNE(
        n_components=2,
        perplexity=eff_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    points_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    object_ids = sorted(class_names.keys())
    cmap = plt.colormaps.get_cmap("tab10")
    for i, oid in enumerate(object_ids):
        mask = labels == oid
        plt.scatter(
            points_2d[mask, 0],
            points_2d[mask, 1],
            s=55,
            alpha=0.85,
            color=cmap(i / max(1, len(object_ids) - 1)),
            label=f"{oid}: {class_names[oid]}",
            edgecolors="black",
            linewidths=0.4,
        )
    plt.title("UniTouch Embeddings of TouchNet Tactile Samples (t-SNE)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.grid(alpha=0.22)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return points_2d


def _compute_knn_accuracy(embeddings: np.ndarray, labels: np.ndarray, k: int, seed: int) -> tuple[float, float, int]:
    estimator = KNeighborsClassifier(n_neighbors=k, weights="distance")
    _, counts = np.unique(labels, return_counts=True)
    n_splits = int(min(5, np.min(counts)))
    if n_splits < 2:
        raise ValueError("Need at least 2 samples per class for kNN cross-validation.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, embeddings, labels, cv=cv, scoring="accuracy")
    return float(np.mean(scores)), float(np.std(scores)), n_splits


def _separability_verdict(knn_acc: float, silhouette: float) -> str:
    if knn_acc >= 0.85 and silhouette >= 0.22:
        return "Yes: clusters are reasonably separable."
    if knn_acc >= 0.70 and silhouette >= 0.10:
        return "Partially: clusters show overlap but are still distinguishable."
    return "No: clusters are not cleanly separable."


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    object_ids = [int(x.strip()) for x in args.object_ids.split(",") if x.strip()]
    if len(object_ids) != 5:
        raise ValueError(f"Expected exactly 5 object ids, got {len(object_ids)}: {object_ids}")
    if args.samples_per_object < 2:
        raise ValueError("--samples-per-object should be >= 2.")

    object_metas = _load_object_metadata()
    missing_meta = [oid for oid in object_ids if oid not in object_metas]
    if missing_meta:
        raise ValueError(f"Object ids missing from metadata csv: {missing_meta}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Collecting tactile renders from TouchNet...")
    samples = _collect_tactile_samples(
        object_ids=object_ids,
        object_metas=object_metas,
        samples_per_object=args.samples_per_object,
        seed=args.seed,
    )
    labels = np.asarray([s.object_id for s in samples], dtype=np.int32)
    class_names = {oid: object_metas[oid].name for oid in object_ids}

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    print(f"[info] Embedding tactile samples with frozen UniTouch on {device}...")
    embeddings = _embed_tactile_samples(samples=samples, device=device, batch_size=args.batch_size)

    print("[info] Computing t-SNE and kNN metrics...")
    tsne_png = out_dir / "unitouch_touch_tsne.png"
    tsne_points = _save_tsne_plot(
        embeddings=embeddings,
        labels=labels,
        class_names=class_names,
        perplexity=args.tsne_perplexity,
        seed=args.seed,
        out_path=tsne_png,
    )

    knn_acc_mean, knn_acc_std, cv_folds = _compute_knn_accuracy(
        embeddings=embeddings,
        labels=labels,
        k=args.knn_k,
        seed=args.seed,
    )
    sil = float(silhouette_score(embeddings, labels))
    verdict = _separability_verdict(knn_acc_mean, sil)

    samples_json = []
    for s in samples:
        samples_json.append(
            {
                "object_id": s.object_id,
                "object_name": s.object_name,
                "material": s.material,
                "contact_xyz_local": [float(x) for x in s.contact_xyz_local.tolist()],
                "orientation_theta_phi": [float(x) for x in s.orientation_theta_phi.tolist()],
                "press_depth_m": float(s.press_depth),
            }
        )

    results = {
        "object_ids": object_ids,
        "objects": {
            str(oid): {
                "name": object_metas[oid].name,
                "material": object_metas[oid].material,
                "url": object_metas[oid].url,
            }
            for oid in object_ids
        },
        "n_samples_total": int(len(samples)),
        "samples_per_object": int(args.samples_per_object),
        "embedding_dim": int(embeddings.shape[1]),
        "knn": {
            "k": int(args.knn_k),
            "cv_folds": int(cv_folds),
            "accuracy_mean": knn_acc_mean,
            "accuracy_std": knn_acc_std,
        },
        "silhouette_score": sil,
        "separability_verdict": verdict,
        "artifacts": {
            "tsne_plot": str(tsne_png),
        },
        "samples": samples_json,
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    np.save(out_dir / "unitouch_embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "tsne_points.npy", tsne_points)

    print("\n=== Experiment Summary ===")
    print(f"Objects: {object_ids}")
    print(f"Total tactile samples: {len(samples)}")
    print(f"kNN accuracy (k={args.knn_k}, {cv_folds}-fold): {knn_acc_mean:.4f} ± {knn_acc_std:.4f}")
    print(f"Silhouette score: {sil:.4f}")
    print(f"Clusters separable? {verdict}")
    print(f"Saved t-SNE plot: {tsne_png}")
    print(f"Saved metrics: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
