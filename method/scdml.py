#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scDML integration + evaluation on GSE194122 (OpenProblems NeurIPS 2021 CITE-seq BMMC).

High-level intent (bioinformatics rationale):
- scDML is used as an integration method: remove batch structure while preserving
  cell-type structure in a learned embedding (adata_proc.obsm['X_emb']).
- We then:
    1) transfer the integrated embedding back to the original AnnData object
    2) compute a post-integration UMAP for visualization
    3) majority-vote map clusters -> cell-type labels (quick proxy classifier)
    4) compute standard clustering/label agreement metrics (ARI/AMI/ASW etc.)

Important practical notes:
- This script assumes scDML is already installed in your environment.
  Installation belongs in environment.yml/requirements.txt, not inside the script.
- The processed GSE194122 file is large; avoid densifying adata.X.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder

# scDML
from scDML import scDMLModel


GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/"
    "GSE194122%5Fopenproblems%5Fneurips2021%5Fcite%5FBMMC%5Fprocessed.h5ad.gz"
)


# ---------------------------
# I/O utilities
# ---------------------------
def download_if_needed(url: str, out_path: Path, overwrite: bool = False) -> None:
    """Download file using stdlib (no notebook magics)."""
    if out_path.exists() and not overwrite:
        print(f"[download] Exists, skipping: {out_path}")
        return

    print(f"[download] Fetching: {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from urllib.request import urlretrieve
    urlretrieve(url, out_path)

    print(f"[download] Saved: {out_path}")


def gunzip_if_needed(gz_path: Path, out_path: Optional[Path] = None, overwrite: bool = False) -> Path:
    """Decompress .gz to output path (stream copy)."""
    if out_path is None:
        out_path = gz_path.with_suffix("")  # drop ".gz"

    if out_path.exists() and not overwrite:
        print(f"[gunzip] Exists, skipping: {out_path}")
        return out_path

    print(f"[gunzip] Decompressing: {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_path


# ---------------------------
# Metric helpers
# ---------------------------
def evaluate_clustering(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    silhouette_embedding: Optional[np.ndarray] = None,
    silhouette_sample_size: int = 10_000,
    random_state: int = 0,
) -> Dict[str, float]:
    """
    Compute a compact panel of clustering/classification-style metrics.

    Caveat (research realism):
    - Majority-vote label assignment is not a proper supervised classifier;
      these "Accuracy/F1" numbers are best interpreted as cluster purity proxies.
    - Silhouette depends strongly on distance geometry and subsampling.
    """
    metrics: Dict[str, float] = {}

    # --- Silhouette (ASW) ---
    if silhouette_embedding is not None:
        uniq = np.unique(predicted_labels)
        if len(uniq) > 1:
            n = silhouette_embedding.shape[0]
            m = min(n, silhouette_sample_size)
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=m, replace=False)

            le = LabelEncoder().fit(predicted_labels)
            y_enc = le.transform(predicted_labels[idx])

            try:
                metrics["ASW"] = float(silhouette_score(silhouette_embedding[idx], y_enc, metric="euclidean"))
            except Exception:
                metrics["ASW"] = float("nan")
        else:
            metrics["ASW"] = float("nan")
    else:
        metrics["ASW"] = float("nan")

    # --- Unsupervised agreement ---
    metrics["ARI"] = float(adjusted_rand_score(true_labels, predicted_labels))
    metrics["AMI"] = float(adjusted_mutual_info_score(true_labels, predicted_labels))

    # --- “Classification-style” agreement (purity proxy) ---
    metrics["Accuracy"] = float(accuracy_score(true_labels, predicted_labels))
    metrics["Kappa"] = float(cohen_kappa_score(true_labels, predicted_labels))
    metrics["MCC"] = float(matthews_corrcoef(true_labels, predicted_labels))

    metrics["Precision_macro"] = float(precision_score(true_labels, predicted_labels, average="macro", zero_division=0))
    metrics["Recall_macro"] = float(recall_score(true_labels, predicted_labels, average="macro", zero_division=0))
    metrics["F1_macro"] = float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0))

    metrics["Precision_weighted"] = float(precision_score(true_labels, predicted_labels, average="weighted", zero_division=0))
    metrics["Recall_weighted"] = float(recall_score(true_labels, predicted_labels, average="weighted", zero_division=0))
    metrics["F1_weighted"] = float(f1_score(true_labels, predicted_labels, average="weighted", zero_division=0))

    return metrics


# ---------------------------
# scDML workflow
# ---------------------------
def find_cluster_col(adata_proc: sc.AnnData) -> str:
    """
    scDML preprocess/integrate tends to leave cluster assignments in obs under
    either 'reassign_cluster' or 'init_cluster'. We handle both robustly.
    """
    if "reassign_cluster" in adata_proc.obs.columns:
        return "reassign_cluster"
    if "init_cluster" in adata_proc.obs.columns:
        return "init_cluster"
    raise KeyError(
        "Could not find scDML cluster labels in adata_proc.obs. "
        "Expected 'reassign_cluster' or 'init_cluster'."
    )


def majority_vote_labels(adata_proc: sc.AnnData, cluster_col: str, true_label_col: str) -> pd.Series:
    """
    Map each cluster ID -> most frequent true label.
    This is a standard quick diagnostic: tells us whether clusters align with known biology.
    """
    major_by_cluster = (
        adata_proc.obs
        .groupby(cluster_col)[true_label_col]
        .agg(lambda x: x.value_counts().idxmax())
    )
    return adata_proc.obs[cluster_col].map(major_by_cluster.to_dict())


def run_scdml(
    h5ad_path: Path,
    out_dir: Path,
    batch_key_in: str = "batch",
    celltype_key_in: str = "cell_type",
    louvain_resolution: float = 3.0,
    n_neighbors_umap: int = 15,
) -> Tuple[sc.AnnData, sc.AnnData, str]:
    """
    Run scDML and return:
      - raw adata (with embeddings added)
      - adata_proc (scDML internal/preprocessed object)
      - cluster_col used for labeling
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[io] Reading: {h5ad_path}")
    adata = sc.read_h5ad(str(h5ad_path))
    print(adata)

    # scDML code expects these exact obs keys (as used in your notebook):
    #   BATCH + celltype
    # We rename for compatibility but we keep originals in case you need them later.
    if batch_key_in not in adata.obs.columns:
        raise KeyError(f"Missing obs['{batch_key_in}'] in input AnnData.")
    if celltype_key_in not in adata.obs.columns:
        raise KeyError(f"Missing obs['{celltype_key_in}'] in input AnnData.")

    adata = adata.copy()
    adata.obs = adata.obs.rename(columns={batch_key_in: "BATCH", celltype_key_in: "celltype"})

    ncluster = int(adata.obs["celltype"].nunique())
    print(f"[qc] cell types (ncluster) = {ncluster}")

    scdml = scDMLModel(save_dir=str(out_dir))

    # Preprocess: Louvain at high resolution.
    # In practice, resolution is dataset-specific; high values can over-fragment.
    adata_proc = scdml.preprocess(adata, cluster_method="louvain", resolution=louvain_resolution)

    # Integrate: key output is adata_proc.obsm['X_emb']
    scdml.integrate(
        adata_proc,
        batch_key="BATCH",
        ncluster_list=[ncluster],
        expect_num_cluster=ncluster,
        merge_rule="rule2",
    )

    # Plot training loss (useful sanity check for optimization stability)
    if hasattr(scdml, "loss") and scdml.loss is not None and len(scdml.loss) > 0:
        plt.figure(figsize=(4, 3))
        plt.plot(range(1, len(scdml.loss) + 1), scdml.loss, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("scDML Training Loss")
        plt.tight_layout()
        plt.savefig(out_dir / "scdml_training_loss.png", dpi=200)
        plt.close()

    # Identify the cluster labels column for downstream majority vote
    cluster_col = find_cluster_col(adata_proc)
    print(f"[cluster] Using cluster column: {cluster_col}")

    # Ensure integrated embedding exists
    if "X_emb" not in adata_proc.obsm:
        raise KeyError("Expected adata_proc.obsm['X_emb'] after scDML integrate().")

    # Transfer embedding back to the (renamed) adata object
    adata.obsm["X_scDML"] = adata_proc.obsm["X_emb"]

    # Post-integration UMAP from scDML embedding
    sc.pp.neighbors(adata, use_rep="X_scDML", n_neighbors=n_neighbors_umap)
    sc.tl.umap(adata)
    adata.obsm["X_umap_scDML"] = adata.obsm["X_umap"].copy()

    # Save embeddings
    np.save(out_dir / "X_scDML.npy", adata.obsm["X_scDML"])
    np.save(out_dir / "X_umap_scDML.npy", adata.obsm["X_umap_scDML"])

    return adata, adata_proc, cluster_col


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scDML on GSE194122 processed BMMC AnnData.")
    p.add_argument("--workdir", type=Path, default=Path("./data"), help="Download/decompress directory.")
    p.add_argument("--download", action="store_true", help="Download the h5ad.gz from GEO.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite downloaded/decompressed files.")
    p.add_argument("--h5ad", type=Path, default=None, help="Local .h5ad path (skip download if provided).")
    p.add_argument("--outdir", type=Path, default=Path("./outputs/GSE194122_scDML"), help="Output directory.")

    p.add_argument("--batch_key", type=str, default="batch", help="obs column for batch.")
    p.add_argument("--celltype_key", type=str, default="cell_type", help="obs column for ground-truth cell type.")
    p.add_argument("--louvain_resolution", type=float, default=3.0, help="Resolution for Louvain in scDML preprocess.")
    p.add_argument("--neighbors", type=int, default=15, help="Neighbors for UMAP on scDML embedding.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    # Resolve input
    if args.h5ad is not None:
        h5ad_path = args.h5ad
        if not h5ad_path.exists():
            raise FileNotFoundError(f"--h5ad not found: {h5ad_path}")
    else:
        if not args.download:
            raise ValueError("Provide --h5ad or pass --download.")
        gz_path = workdir / "GSE194122_BMMC_processed.h5ad.gz"
        h5ad_path = workdir / "GSE194122_BMMC_processed.h5ad"
        download_if_needed(GEO_URL, gz_path, overwrite=args.overwrite)
        h5ad_path = gunzip_if_needed(gz_path, out_path=h5ad_path, overwrite=args.overwrite)

    adata, adata_proc, cluster_col = run_scdml(
        h5ad_path=h5ad_path,
        out_dir=args.outdir,
        batch_key_in=args.batch_key,
        celltype_key_in=args.celltype_key,
        louvain_resolution=args.louvain_resolution,
        n_neighbors_umap=args.neighbors,
    )

    # Majority-vote predicted labels on the processed object
    adata_proc.obs["predicted"] = majority_vote_labels(
        adata_proc=adata_proc,
        cluster_col=cluster_col,
        true_label_col="celltype",
    )

    # Evaluate metrics on adata_proc (same object used for clustering)
    emb = adata_proc.obsm["X_emb"]
    metrics = evaluate_clustering(
        true_labels=adata_proc.obs["celltype"].to_numpy(),
        predicted_labels=adata_proc.obs["predicted"].to_numpy(),
        silhouette_embedding=emb,
        silhouette_sample_size=10_000,
        random_state=0,
    )

    print("\n[metrics] Multiclass clustering & purity-proxy metrics:")
    pprint(metrics)

    # Save labels table for auditing (cell_id, cluster_id, true/pred)
    df_labels = pd.DataFrame(
        {
            "cell_id": adata_proc.obs_names,
            "cluster_id": adata_proc.obs[cluster_col].astype(str).to_numpy(),
            "true_label": adata_proc.obs["celltype"].astype(str).to_numpy(),
            "predicted_label": adata_proc.obs["predicted"].astype(str).to_numpy(),
        }
    )
    df_labels.to_csv(args.outdir / "cell_labels.csv", index=False)
    print(f"\n[save] Wrote labels table: {args.outdir / 'cell_labels.csv'}")

    # Optional quick UMAP plots (saved to disk; safe for headless use)
    sc.pl.umap(adata, color=["celltype", "BATCH"], wspace=0.4, frameon=False, show=False)
    plt.savefig(args.outdir / "umap_scDML_celltype_batch.png", dpi=200)
    plt.close()

    print("[done] scDML pipeline complete.")


if __name__ == "__main__":
    main()
