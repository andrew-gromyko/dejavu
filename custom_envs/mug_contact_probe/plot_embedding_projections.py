#!/usr/bin/env python3
"""Plot PCA, t-SNE, and UMAP projections for saved UniTouch embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

try:
    import umap
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "Missing dependency `umap-learn`. Install with:\n"
        "  .venv-ms3-of2/bin/pip install umap-learn"
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PCA/t-SNE/UMAP projection plots from saved embeddings.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/Users/andrew/Documents/git/maniskill/runs/mug_contact_probe/tactile_unitouch_tsne"),
        help="Directory with unitouch_embeddings.npy, labels.npy, results.json.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for stochastic projections.")
    parser.add_argument("--tsne-perplexity", type=float, default=12.0, help="t-SNE perplexity.")
    parser.add_argument("--umap-neighbors", type=int, default=10, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--point-size-2d", type=float, default=18.0, help="Marker size for 2D plots.")
    parser.add_argument("--point-alpha-2d", type=float, default=0.72, help="Marker alpha for 2D plots.")
    parser.add_argument("--point-size-3d", type=float, default=14.0, help="Marker size for 3D plots.")
    parser.add_argument("--point-alpha-3d", type=float, default=0.80, help="Marker alpha for 3D plots.")
    parser.add_argument("--dpi", type=int, default=320, help="Output image DPI.")
    return parser.parse_args()


def _load_data(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    embeddings = np.load(run_dir / "unitouch_embeddings.npy")
    labels = np.load(run_dir / "labels.npy")
    results = json.loads((run_dir / "results.json").read_text())
    class_names = {int(k): v["name"] for k, v in results["objects"].items()}
    return embeddings, labels, class_names


def _distance_preservation_metrics(high_dim: np.ndarray, low_dim: np.ndarray) -> dict[str, float]:
    hd = pdist(high_dim, metric="euclidean")
    ld = pdist(low_dim, metric="euclidean")
    if np.std(ld) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(hd, ld)[0, 1])
    spear = float(spearmanr(hd, ld).correlation)
    return {
        "distance_pearson": pearson,
        "distance_spearman": spear,
    }


def _scatter_2d(
    ax,
    points: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
    title: str,
    point_size: float,
    point_alpha: float,
) -> None:
    object_ids = sorted(class_names.keys())
    cmap = plt.colormaps.get_cmap("tab10")
    for i, oid in enumerate(object_ids):
        mask = labels == oid
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=point_size,
            alpha=point_alpha,
            color=cmap(i / max(1, len(object_ids) - 1)),
            label=f"{oid}: {class_names[oid]}",
            edgecolors="black",
            linewidths=0.35,
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(alpha=0.2)


def _scatter_3d(
    ax,
    points: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
    title: str,
    point_size: float,
    point_alpha: float,
) -> None:
    object_ids = sorted(class_names.keys())
    cmap = plt.colormaps.get_cmap("tab10")
    for i, oid in enumerate(object_ids):
        mask = labels == oid
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            points[mask, 2],
            s=point_size,
            alpha=point_alpha,
            color=cmap(i / max(1, len(object_ids) - 1)),
            label=f"{oid}: {class_names[oid]}",
            depthshade=True,
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    embeddings, labels, class_names = _load_data(run_dir)

    n_samples = len(embeddings)
    tsne_perplexity = min(args.tsne_perplexity, max(2.0, float(n_samples - 1)))

    pca2 = PCA(n_components=2, random_state=args.seed).fit_transform(embeddings)
    tsne2 = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
    ).fit_transform(embeddings)
    umap2 = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
    ).fit_transform(embeddings)
    mds2_model = MDS(
        n_components=2,
        metric=True,
        normalized_stress="auto",
        n_init=8,
        max_iter=1000,
        random_state=args.seed,
    )
    mds2 = mds2_model.fit_transform(embeddings)

    pca3 = PCA(n_components=3, random_state=args.seed).fit_transform(embeddings)
    tsne3 = TSNE(
        n_components=3,
        perplexity=tsne_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
    ).fit_transform(embeddings)
    umap3 = umap.UMAP(
        n_components=3,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
    ).fit_transform(embeddings)
    mds3_model = MDS(
        n_components=3,
        metric=True,
        normalized_stress="auto",
        n_init=8,
        max_iter=1000,
        random_state=args.seed,
    )
    mds3 = mds3_model.fit_transform(embeddings)

    fig2d, axs2d = plt.subplots(1, 4, figsize=(23, 5.5))
    _scatter_2d(axs2d[0], pca2, labels, class_names, "PCA (2D)", args.point_size_2d, args.point_alpha_2d)
    _scatter_2d(axs2d[1], tsne2, labels, class_names, "t-SNE (2D)", args.point_size_2d, args.point_alpha_2d)
    _scatter_2d(axs2d[2], umap2, labels, class_names, "UMAP (2D)", args.point_size_2d, args.point_alpha_2d)
    _scatter_2d(axs2d[3], mds2, labels, class_names, "MDS (2D)", args.point_size_2d, args.point_alpha_2d)
    handles, labels_legend = axs2d[3].get_legend_handles_labels()
    fig2d.legend(handles, labels_legend, loc="lower center", ncol=5, frameon=True, fontsize=8)
    fig2d.tight_layout(rect=[0, 0.09, 1, 1])
    out2d = run_dir / "unitouch_projection_panel_2d.png"
    fig2d.savefig(out2d, dpi=args.dpi)
    plt.close(fig2d)

    fig3d = plt.figure(figsize=(23, 5.6))
    ax1 = fig3d.add_subplot(141, projection="3d")
    ax2 = fig3d.add_subplot(142, projection="3d")
    ax3 = fig3d.add_subplot(143, projection="3d")
    ax4 = fig3d.add_subplot(144, projection="3d")
    _scatter_3d(ax1, pca3, labels, class_names, "PCA (3D)", args.point_size_3d, args.point_alpha_3d)
    _scatter_3d(ax2, tsne3, labels, class_names, "t-SNE (3D)", args.point_size_3d, args.point_alpha_3d)
    _scatter_3d(ax3, umap3, labels, class_names, "UMAP (3D)", args.point_size_3d, args.point_alpha_3d)
    _scatter_3d(ax4, mds3, labels, class_names, "MDS (3D)", args.point_size_3d, args.point_alpha_3d)
    handles3d, labels3d = ax4.get_legend_handles_labels()
    fig3d.legend(handles3d, labels3d, loc="lower center", ncol=5, frameon=True, fontsize=8)
    fig3d.tight_layout(rect=[0, 0.09, 1, 1])
    out3d = run_dir / "unitouch_projection_panel_3d.png"
    fig3d.savefig(out3d, dpi=args.dpi)
    plt.close(fig3d)

    metrics = {
        "2d": {
            "pca": _distance_preservation_metrics(embeddings, pca2),
            "tsne": _distance_preservation_metrics(embeddings, tsne2),
            "umap": _distance_preservation_metrics(embeddings, umap2),
            "mds": {
                **_distance_preservation_metrics(embeddings, mds2),
                "normalized_stress": float(mds2_model.stress_),
            },
        },
        "3d": {
            "pca": _distance_preservation_metrics(embeddings, pca3),
            "tsne": _distance_preservation_metrics(embeddings, tsne3),
            "umap": _distance_preservation_metrics(embeddings, umap3),
            "mds": {
                **_distance_preservation_metrics(embeddings, mds3),
                "normalized_stress": float(mds3_model.stress_),
            },
        },
    }
    metrics_path = run_dir / "projection_quality_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    np.save(run_dir / "projection_pca_2d.npy", pca2)
    np.save(run_dir / "projection_tsne_2d.npy", tsne2)
    np.save(run_dir / "projection_umap_2d.npy", umap2)
    np.save(run_dir / "projection_mds_2d.npy", mds2)
    np.save(run_dir / "projection_pca_3d.npy", pca3)
    np.save(run_dir / "projection_tsne_3d.npy", tsne3)
    np.save(run_dir / "projection_umap_3d.npy", umap3)
    np.save(run_dir / "projection_mds_3d.npy", mds3)

    print(f"Saved 2D panel: {out2d}")
    print(f"Saved 3D panel: {out3d}")
    print(f"Saved projection metrics: {metrics_path}")
    print("Saved projection arrays: projection_{pca,tsne,umap,mds}_{2d,3d}.npy")


if __name__ == "__main__":
    main()
