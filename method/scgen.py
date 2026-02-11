#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scGen batch removal on GSE194122 (OpenProblems NeurIPS 2021 CITE-seq BMMC).

Bioinformatics intent:
- scGen is typically used to learn a latent representation that is less sensitive to
  technical batch effects while preserving biological structure (cell types).
- For integration QC, we usually visualize UMAP before/after correction and quantify
  label/cluster concordance (e.g., ARI/AMI) and batch mixing proxies.

This script:
  1) Downloads + decompresses the processed .h5ad.gz (optional).
  2) Loads AnnData and computes a "pre" UMAP (for baseline visualization).
  3) Trains scGen with batch as a nuisance covariate and cell_type as a label.
  4) Runs scGen batch removal and stores corrected latent representation.
  5) Computes UMAP on corrected latent and saves arrays to disk.

Important practical notes:
- This dataset is already “processed”; depending on upstream steps, X may be log-normalized.
  scGen / scvi-tools generally expect counts-like inputs. If you have raw counts in a layer,
  you should use it and register it accordingly.
- Avoid densifying adata.X; single-cell matrices can be huge.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scanpy as sc

import pooch
import scvi
import scgen

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder


GEO_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE194122&format=file&file="
    "GSE194122%5Fopenproblems%5Fneurips2021%5Fcite%5FBMMC%5Fprocessed%2Eh5ad%2Egz"
)


# -----------------------------
# I/O helpers
# -----------------------------
def download_with_pooch(url: str, out_dir: Path, fname: str) -> Path:
    """
    Download and cache the dataset. In an actual reproducible pipeline,
    you’d pin the expected hash (known_hash) to guarantee provenance.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = pooch.retrieve(url=url, known_hash=None, path=str(out_dir), fname=fname)
    return Path(path)


def gunzip_if_needed(gz_path: Path, out_path: Optional[Path] = None, overwrite: bool = False) -> Path:
    """Decompress .gz → output path via streaming copy."""
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
# QC / plotting helpers
# -----------------------------
def require_obs_keys(adata: sc.AnnData, keys: list[str]) -> None:
    missing = [k for k in keys if k not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"Missing required obs keys: {missing}. "
            f"Available obs columns (first 30): {list(adata.obs.columns)[:30]}"
        )


def compute_umap(adata: sc.AnnData, use_rep: Optional[str], out_key: str, n_neighbors: int = 15) -> None:
    """
    Compute UMAP either from .X (use_rep=None) or a specified embedding in adata.obsm.
    We save the result into adata.obsm[out_key] so we don't overwrite other UMAPs.
    """
    if use_rep is None:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    else:
        if use_rep not in adata.obsm:
            raise KeyError(f"Requested use_rep='{use_rep}' not found in adata.obsm.")
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

    sc.tl.umap(adata)
    adata.obsm[out_key] = adata.obsm["X_umap"].copy()


def majority_vote_cluster_labels(adata: sc.AnnData, cluster_key: str, true_label_key: str) -> np.ndarray:
    """
    Classic integration sanity check:
    - cluster the latent space
    - assign each cluster its majority true label
    - quantify agreement (ARI/AMI on labels; not a proper supervised evaluation)
    """
    # Map each cluster to the most frequent true label
    major_by_cluster = (
        adata.obs.groupby(cluster_key)[true_label_key]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    return adata.obs[cluster_key].map(major_by_cluster).to_numpy()


def quick_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict[str, float]:
    """Compact unsupervised label-agreement metrics (biologically interpretable sanity checks)."""
    return {
        "ARI": float(adjusted_rand_score(true_labels, predicted_labels)),
        "AMI": float(adjusted_mutual_info_score(true_labels, predicted_labels)),
    }


# -----------------------------
# scGen core
# -----------------------------
def run_scgen(
    adata: sc.AnnData,
    batch_key: str,
    labels_key: str,
    max_epochs: int = 150,
    batch_size: int = 32,
    early_stopping: bool = True,
    early_stopping_patience: int = 25,
    seed: int = 0,
) -> sc.AnnData:
    """
    Train scGen and run batch removal.

    Important: We do NOT monkey-patch scvi internals here.
    Monkey-patching latent extraction is brittle across versions and can return
    the wrong tensor without throwing an error.
    """
    scvi.settings.seed = seed
    sc.settings.verbosity = 2

    print(f"[versions] scanpy={sc.__version__} | scvi-tools={scvi.__version__} | scgen={scgen.__version__}")

    # Register AnnData with scGen/scvi-tools
    scgen.SCGEN.setup_anndata(adata, batch_key=batch_key, labels_key=labels_key)

    model = scgen.SCGEN(adata)

    # Train first, then save (your original saved an untrained checkpoint).
    print(f"[train] Training scGen (max_epochs={max_epochs})")
    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
    )

    # Run scGen batch removal
    print("[infer] Running scGen batch removal")
    corrected = model.batch_removal()

    # scGen conventions: corrected.obsm typically contains 'corrected_latent' and 'latent'
    if "corrected_latent" not in corrected.obsm:
        raise KeyError("Expected corrected.obsm['corrected_latent'] not found after batch_removal().")

    return corrected


# -----------------------------
# CLI / main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scGen batch removal on GSE194122 BMMC processed AnnData.")
    p.add_argument("--workdir", type=Path, default=Path("./data"), help="Directory for download/decompression.")
    p.add_argument("--download", action="store_true", help="Download dataset from GEO.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite decompressed file if exists.")
    p.add_argument("--h5ad", type=Path, default=None, help="Path to local .h5ad (skip download).")

    p.add_argument("--outdir", type=Path, default=Path("./outputs/GSE194122_scgen"), help="Output directory.")
    p.add_argument("--batch_key", type=str, default="batch")
    p.add_argument("--labels_key", type=str, default="cell_type")

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--neighbors", type=int, default=15)

    p.add_argument("--do_leiden_qc", action="store_true", help="Cluster corrected latent and print ARI/AMI.")
    p.add_argument("--leiden_resolution", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Resolve input .h5ad
    if args.h5ad is not None:
        h5ad_path = args.h5ad
        if not h5ad_path.exists():
            raise FileNotFoundError(f"--h5ad not found: {h5ad_path}")
    else:
        if not args.download:
            raise ValueError("Provide --h5ad or pass --download.")
        gz_path = download_with_pooch(
            url=GEO_URL,
            out_dir=args.workdir,
            fname="GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz",
        )
        h5ad_path = gunzip_if_needed(gz_path, overwrite=args.overwrite)

    print(f"[io] Reading: {h5ad_path}")
    adata = sc.read_h5ad(str(h5ad_path))
    require_obs_keys(adata, [args.batch_key, args.labels_key])

    # --- Pre-integration baseline UMAP ---
    # We compute from whatever representation Scanpy uses by default (.X / PCA if present).
    # This is just a qualitative baseline.
    print("[umap] Computing baseline UMAP (pre)")
    compute_umap(adata, use_rep=None, out_key="X_umap_pre", n_neighbors=args.neighbors)

    # --- scGen integration ---
    corrected = run_scgen(
        adata=adata.copy(),
        batch_key=args.batch_key,
        labels_key=args.labels_key,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping=True,
        early_stopping_patience=args.patience,
        seed=0,
    )

    # Store corrected latent on a working object for downstream use
    out_adata = corrected.copy()
    out_adata.obsm["X_scGen"] = out_adata.obsm["corrected_latent"]

    # --- Post-integration UMAP ---
    print("[umap] Computing post-integration UMAP (scGen corrected latent)")
    compute_umap(out_adata, use_rep="X_scGen", out_key="X_umap_scGen", n_neighbors=args.neighbors)

    # --- Optional clustering QC ---
    if args.do_leiden_qc:
        # In benchmarking, we often cluster the integrated latent and check agreement with labels.
        sc.tl.leiden(out_adata, resolution=args.leiden_resolution, key_added="leiden_scGen")
        pred = majority_vote_cluster_labels(out_adata, "leiden_scGen", args.labels_key)

        metrics = quick_metrics(out_adata.obs[args.labels_key].to_numpy(), pred)
        print("[qc] Majority-vote cluster→label agreement (purity proxy):", metrics)

    # --- Save outputs (portable, no Google Drive assumptions) ---
    np.save(args.outdir / "X_scGen.npy", out_adata.obsm["X_scGen"])
    np.save(args.outdir / "X_umap_scGen.npy", out_adata.obsm["X_umap_scGen"])
    np.save(args.outdir / "X_umap_pre.npy", out_adata.obsm["X_umap_pre"])

    # Save minimal metadata for downstream evaluation
    out_adata.obs[[args.batch_key, args.labels_key]].to_csv(args.outdir / "obs_metadata.csv", index=True)

    print("[save] Wrote:")
    print("  - X_scGen.npy       ", out_adata.obsm["X_scGen"].shape)
    print("  - X_umap_scGen.npy  ", out_adata.obsm["X_umap_scGen"].shape)
    print("  - X_umap_pre.npy    ", out_adata.obsm["X_umap_pre"].shape)
    print("  - obs_metadata.csv")
    print("[done] scGen pipeline complete.")


if __name__ == "__main__":
    main()
