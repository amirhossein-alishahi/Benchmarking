#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scVI batch correction on GSE194122 (OpenProblems NeurIPS 2021 CITE-seq BMMC).

Bioinformatics rationale:
- We use scVI to learn a latent space that explains gene expression variation while
  regressing out technical batch structure (batch_key='batch').
- A standard qualitative QC is UMAP before vs after integration, colored by:
    - cell_type (biological structure)
    - batch (technical mixing)
- A standard quantitative QC is clustering in latent space and measuring agreement
  with cell_type labels (ARI/AMI) and cluster purity proxies (majority-vote mapping).

Key assumptions:
- Raw counts are stored in adata.layers['counts'] (true for many OpenProblems artifacts).
  If that layer is missing, this script fails fast (do not silently train on log data).
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scvi

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    accuracy_score,
    f1_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/"
    "GSE194122%5Fopenproblems%5Fneurips2021%5Fcite%5FBMMC%5Fprocessed.h5ad.gz"
)


# -----------------------------
# I/O helpers
# -----------------------------
def download_if_needed(url: str, out_path: Path, overwrite: bool = False) -> None:
    """Download via stdlib; avoids notebook shell magics."""
    if out_path.exists() and not overwrite:
        print(f"[download] Exists, skipping: {out_path}")
        return

    print(f"[download] Fetching: {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from urllib.request import urlretrieve
    urlretrieve(url, out_path)

    print(f"[download] Saved: {out_path}")


def gunzip_if_needed(gz_path: Path, out_path: Optional[Path] = None, overwrite: bool = False) -> Path:
    """Decompress .gz -> output path using stream copy (safe for large files)."""
    if out_path is None:
        out_path = gz_path.with_suffix("")  # drop ".gz"

    if out_path.exists() and not overwrite:
        print(f"[gunzip] Exists, skipping: {out_path}")
        return out_path

    print(f"[gunzip] Decompressing: {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_path


# -----------------------------
# Validation helpers
# -----------------------------
def require_obs_keys(adata: sc.AnnData, keys: List[str]) -> None:
    missing = [k for k in keys if k not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"Missing required obs keys: {missing}. "
            f"Available columns (first 30): {list(adata.obs.columns)[:30]}"
        )


def require_layer(adata: sc.AnnData, layer: str) -> None:
    layers = list(adata.layers.keys()) if adata.layers is not None else []
    if layer not in layers:
        raise ValueError(
            f"Required layer '{layer}' not found. Available layers: {layers}. "
            "scVI expects counts; do not silently train on normalized/log values."
        )


def ensure_pre_umap(adata: sc.AnnData, prefer_modality: str = "GEX", n_neighbors: int = 15) -> None:
    """
    The OpenProblems multi-omic objects often include precomputed embeddings:
      - 'GEX_X_umap' or 'ADT_X_umap'
    We alias one of them to 'X_umap_pre' for consistent downstream plotting.

    If none exist, we build a baseline UMAP from PCA if available; otherwise from .X.
    """
    if "X_umap_pre" in adata.obsm:
        return

    candidates = (
        ["GEX_X_umap", "X_umap", "ADT_X_umap"]
        if prefer_modality.upper() == "GEX"
        else ["ADT_X_umap", "X_umap", "GEX_X_umap"]
    )
    for k in candidates:
        if k in adata.obsm:
            adata.obsm["X_umap_pre"] = adata.obsm[k].copy()
            print(f"[umap] Aliased '{k}' -> 'X_umap_pre'")
            return

    # fallback: compute baseline UMAP
    pca_key = f"{prefer_modality.upper()}_X_pca"
    if pca_key in adata.obsm:
        sc.pp.neighbors(adata, use_rep=pca_key, key_added="neighbors_pre", n_neighbors=n_neighbors)
        sc.tl.umap(adata, neighbors_key="neighbors_pre")
        adata.obsm["X_umap_pre"] = adata.obsm["X_umap"].copy()
        print(f"[umap] Built 'X_umap_pre' from '{pca_key}'")
        return

    sc.pp.neighbors(adata, key_added="neighbors_pre", n_neighbors=n_neighbors)
    sc.tl.umap(adata, neighbors_key="neighbors_pre")
    adata.obsm["X_umap_pre"] = adata.obsm["X_umap"].copy()
    print("[umap] Built 'X_umap_pre' from .X (fallback)")


def plot_embedding(adata: sc.AnnData, obsm_key: str, colors: List[str], title: str, s: int = 5) -> None:
    """
    Scanpy expects basis names with 'X_' prefix. We temporarily alias the embedding
    to avoid polluting AnnData with dozens of basis keys.
    """
    if obsm_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm['{obsm_key}'] for plotting.")
    tmp = "X_tmp_plot"
    adata.obsm[tmp] = adata.obsm[obsm_key]
    sc.pl.embedding(adata, basis=tmp, color=colors, s=s, frameon=False, ncols=2, title=title)
    del adata.obsm[tmp]


# -----------------------------
# Metrics / clustering helpers
# -----------------------------
def cluster_leiden(adata: sc.AnnData, use_rep: str, resolution: float = 1.0, n_neighbors: int = 15) -> str:
    """Build graph on use_rep and run Leiden; returns the cluster key used."""
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    sc.tl.leiden(adata, key_added="leiden", resolution=resolution)
    return "leiden"


def majority_vote_labels(adata: sc.AnnData, cluster_key: str, true_label_key: str, out_key: str = "pred_major") -> None:
    """Map each cluster -> most frequent true label (cluster purity proxy)."""
    major = (
        adata.obs.groupby(cluster_key)[true_label_key]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    adata.obs[out_key] = adata.obs[cluster_key].map(major).astype("category")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    These behave like classification metrics but here they are *cluster purity proxies*.
    Interpret with caution: no held-out test set, and mapping uses true labels.
    """
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)

    eps = 1e-12
    gmean = float(np.exp(np.mean(np.log(recalls + eps))))

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_macro": float(f1_macro),
        "F1_weighted": float(f1_weighted),
        "G-Mean": float(gmean),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def asw_on_embedding(X: np.ndarray, labels: np.ndarray, sample_size: int = 10_000, seed: int = 0) -> float:
    """
    Silhouette score (ASW proxy). We subsample for speed and memory stability.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    n = X.shape[0]
    m = min(n, sample_size)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)

    le = LabelEncoder().fit(labels)
    y = le.transform(labels[idx])

    try:
        return float(silhouette_score(X[idx], y, metric="euclidean"))
    except Exception:
        return float("nan")


def build_sankey_true_to_pred(
    df: pd.DataFrame,
    highlight_label: str,
    out_html: Path,
) -> None:
    """
    Writes an interactive Sankey diagram: True cell types -> predicted (majority-vote) labels.
    We highlight links that touch a chosen label (often a rare cell type).

    Requires plotly. If not installed, we skip gracefully.
    """
    if not _HAS_PLOTLY:
        print("[sankey] plotly not installed; skipping sankey export.")
        return

    counts = df.groupby(["True", "PredMajor"]).size().reset_index(name="count")
    trues = counts["True"].unique().tolist()
    preds = counts["PredMajor"].unique().tolist()

    # Put the highlight label first if present (makes the plot stable across runs)
    if highlight_label in trues:
        trues = [highlight_label] + [x for x in trues if x != highlight_label]
    if highlight_label in preds:
        preds = [highlight_label] + [x for x in preds if x != highlight_label]

    nodes = trues + preds
    true_to_idx = {v: i for i, v in enumerate(trues)}
    pred_to_idx = {v: i + len(trues) for i, v in enumerate(preds)}

    source = counts["True"].map(true_to_idx).tolist()
    target = counts["PredMajor"].map(pred_to_idx).tolist()
    value = counts["count"].tolist()

    link_colors = []
    for t, p in counts[["True", "PredMajor"]].itertuples(index=False):
        if t == highlight_label and p == highlight_label:
            link_colors.append("green")
        elif t == highlight_label and p != highlight_label:
            link_colors.append("red")
        elif t != highlight_label and p == highlight_label:
            link_colors.append("blue")
        else:
            link_colors.append("lightgray")

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=15, thickness=18),
        link=dict(source=source, target=target, value=value, color=link_colors),
    ))
    fig.update_layout(
        title_text=f"Sankey: True → PredMajor (highlight: {highlight_label})",
        font_size=11,
        width=1100,
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.write_html(str(out_html))
    print(f"[sankey] Wrote: {out_html}")


# -----------------------------
# scVI pipeline
# -----------------------------
def run_scvi(
    adata: sc.AnnData,
    counts_layer: str,
    batch_key: str,
    n_latent: int,
    max_epochs: int,
    early_stopping: bool,
    seed: int = 0,
) -> None:
    """
    Train scVI and store:
      - adata.obsm['X_scVI']
    """
    scvi.settings.seed = seed
    sc.settings.verbosity = 2

    # Ensure correct dtype for categorical covariates
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")

    scvi.model.SCVI.setup_anndata(
        adata,
        layer=counts_layer,
        batch_key=batch_key,
    )

    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(max_epochs=max_epochs, early_stopping=early_stopping)

    adata.obsm["X_scVI"] = model.get_latent_representation()


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scVI on GSE194122 processed BMMC AnnData.")
    p.add_argument("--workdir", type=Path, default=Path("./data"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--h5ad", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("./outputs/GSE194122_scvi"))

    p.add_argument("--counts_layer", type=str, default="counts")
    p.add_argument("--batch_key", type=str, default="batch")
    p.add_argument("--celltype_key", type=str, default="cell_type")
    p.add_argument("--prefer_modality", type=str, default="GEX", choices=["GEX", "ADT"])

    p.add_argument("--n_latent", type=int, default=30)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--early_stopping", action="store_true")

    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--leiden_resolution", type=float, default=1.0)

    p.add_argument("--export_sankey", action="store_true")
    p.add_argument("--rare_threshold", type=float, default=0.01, help="Rare type threshold (fraction).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    # --- resolve input file ---
    if args.h5ad is not None:
        h5ad_path = args.h5ad
        if not h5ad_path.exists():
            raise FileNotFoundError(f"--h5ad not found: {h5ad_path}")
    else:
        if not args.download:
            raise ValueError("Provide --h5ad or pass --download.")
        gz_path = args.workdir / "BMMC_processed.h5ad.gz"
        h5ad_path = args.workdir / "BMMC_processed.h5ad"
        download_if_needed(GEO_URL, gz_path, overwrite=args.overwrite)
        h5ad_path = gunzip_if_needed(gz_path, out_path=h5ad_path, overwrite=args.overwrite)

    print(f"[io] Reading: {h5ad_path}")
    adata = sc.read_h5ad(str(h5ad_path))
    require_obs_keys(adata, [args.batch_key, args.celltype_key])
    require_layer(adata, args.counts_layer)

    # categorical labels are helpful for plotting / groupby
    adata.obs[args.celltype_key] = adata.obs[args.celltype_key].astype("category")

    # --- pre-UMAP baseline ---
    ensure_pre_umap(adata, prefer_modality=args.prefer_modality, n_neighbors=args.neighbors)

    # --- scVI training ---
    print("[scvi] Training scVI")
    run_scvi(
        adata=adata,
        counts_layer=args.counts_layer,
        batch_key=args.batch_key,
        n_latent=args.n_latent,
        max_epochs=args.epochs,
        early_stopping=args.early_stopping,
        seed=0,
    )

    # --- post-UMAP on scVI latent ---
    print("[umap] Computing UMAP on X_scVI")
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=args.neighbors)
    sc.tl.umap(adata)
    adata.obsm["X_umap_scvi"] = adata.obsm["X_umap"].copy()

    # --- plots (interactive display; safe to comment out on headless HPC) ---
    plot_embedding(adata, "X_umap_pre", [args.celltype_key, args.batch_key], "Before scVI")
    plot_embedding(adata, "X_umap_scvi", [args.celltype_key, args.batch_key], "After scVI")

    # --- clustering + majority vote labels ---
    cluster_key = cluster_leiden(adata, use_rep="X_scVI", resolution=args.leiden_resolution, n_neighbors=args.neighbors)
    majority_vote_labels(adata, cluster_key=cluster_key, true_label_key=args.celltype_key, out_key="pred_major")

    # --- metrics (overall + rare subset) ---
    y_true = adata.obs[args.celltype_key].astype(str).to_numpy()
    y_pred = adata.obs["pred_major"].astype(str).to_numpy()
    y_cluster = adata.obs[cluster_key].astype(str).to_numpy()
    X_latent = adata.obsm["X_scVI"]

    overall = {
        "ASW_true_on_scVI": asw_on_embedding(X_latent, y_true),
        "ARI(true vs cluster)": float(adjusted_rand_score(y_true, y_cluster)),
        "AMI(true vs cluster)": float(adjusted_mutual_info_score(y_true, y_cluster)),
    }
    overall.update(classification_metrics(y_true, y_pred))

    freq = adata.obs[args.celltype_key].value_counts(normalize=True)
    rare_types = freq[freq < args.rare_threshold].index.astype(str).tolist()
    mask_rare = adata.obs[args.celltype_key].astype(str).isin(rare_types).to_numpy()

    if mask_rare.sum() > 0 and len(np.unique(y_true[mask_rare])) > 1:
        rare = {
            "ASW_true_on_scVI": asw_on_embedding(X_latent[mask_rare], y_true[mask_rare]),
            "ARI(true vs cluster)": float(adjusted_rand_score(y_true[mask_rare], y_cluster[mask_rare])),
            "AMI(true vs cluster)": float(adjusted_mutual_info_score(y_true[mask_rare], y_cluster[mask_rare])),
        }
        rare.update(classification_metrics(y_true[mask_rare], y_pred[mask_rare]))
    else:
        rare = {"note": "No rare cells or only one rare label; skipped rare metrics."}

    metrics_df = pd.DataFrame({"Overall": overall, "Rare": rare})
    print("\n[metrics]\n", metrics_df)

    metrics_df.to_csv(args.outdir / "scvi_metrics_summary.csv")
    print(f"[save] Wrote: {args.outdir / 'scvi_metrics_summary.csv'}")

    # --- sankey exports (optional) ---
    if args.export_sankey:
        if not rare_types:
            print("[sankey] No rare types found; skipping.")
        else:
            out_pairs = pd.DataFrame({"True": y_true, "PredMajor": y_pred})
            out_pairs.to_csv(args.outdir / "true_vs_pred_pairs.csv", index=False)
            print(f"[sankey] Wrote pairs: {args.outdir / 'true_vs_pred_pairs.csv'}")

            # export one Sankey per rare type (can be many; use with care)
            sankey_dir = args.outdir / "sankey"
            sankey_dir.mkdir(parents=True, exist_ok=True)
            for rt in rare_types:
                build_sankey_true_to_pred(out_pairs, highlight_label=rt, out_html=sankey_dir / f"sankey_{rt}.html")

    # --- save primary arrays for downstream benchmarking ---
    np.save(args.outdir / "X_scVI.npy", adata.obsm["X_scVI"])
    np.save(args.outdir / "X_umap_pre.npy", adata.obsm["X_umap_pre"])
    np.save(args.outdir / "X_umap_scvi.npy", adata.obsm["X_umap_scvi"])
    adata.obs[[args.batch_key, args.celltype_key, cluster_key, "pred_major"]].to_csv(args.outdir / "obs_metadata.csv")

    # Optional: save full corrected AnnData (can be large)
    # adata.write(args.outdir / "BMMC_scvi_corrected.h5ad", compression="gzip")

    print("[done] scVI pipeline complete.")


if __name__ == "__main__":
    main()
