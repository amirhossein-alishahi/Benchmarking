#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Harmony integration for GSE194122 (OpenProblems NeurIPS 2021 CITE-seq BMMC).

This script:
  1) Downloads the processed .h5ad.gz from NCBI GEO (optional).
  2) Decompresses it (if needed).
  3) Loads AnnData, runs a standard Scanpy preprocessing pipeline,
     then performs batch correction with Harmony on PCA space.
  4) Computes neighbors/UMAP using Harmony-corrected embedding.
  5) Saves the corrected embeddings as .npy files.

Notes (single-cell practicality):
- DO NOT densify adata.X unless you are 100% sure it is small.
  Most scRNA/CITE matrices are sparse and large; densifying can OOM.
- Harmony expects a dense embedding (PCA matrix), not raw expression.
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import harmonypy as hm


GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/"
    "GSE194122%5Fopenproblems%5Fneurips2021%5Fcite%5FBMMC%5Fprocessed.h5ad.gz"
)


def download_file(url: str, out_path: Path, overwrite: bool = False) -> None:
    """
    Download a file via HTTP(S) without notebook magics.

    In production bioinformatics pipelines you might prefer:
      - `aria2c` for parallel download
      - checksum verification
    but for a minimal script, urllib is sufficient.
    """
    if out_path.exists() and not overwrite:
        print(f"[download] File exists, skipping: {out_path}")
        return

    print(f"[download] Fetching: {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Standard library download to avoid external dependencies.
    import urllib.request

    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())

    print(f"[download] Saved to: {out_path}")


def gunzip_if_needed(gz_path: Path, out_path: Optional[Path] = None, overwrite: bool = False) -> Path:
    """
    Decompress .gz to a target file. Returns the decompressed path.
    """
    if out_path is None:
        # Remove one suffix: *.h5ad.gz -> *.h5ad
        out_path = gz_path.with_suffix("")  # drops ".gz"

    if out_path.exists() and not overwrite:
        print(f"[gunzip] Decompressed file exists, skipping: {out_path}")
        return out_path

    print(f"[gunzip] Decompressing: {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        # Stream copy to avoid loading file into memory
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)

    return out_path


def run_harmony_pipeline(
    h5ad_path: Path,
    batch_key: str = "batch",
    celltype_key: str = "cell_type",
    n_hvg: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    harmony_max_iter: int = 30,
) -> sc.AnnData:
    """
    Core analysis pipeline: preprocess -> PCA -> Harmony -> UMAP.

    We keep this as a function so it can be unit-tested and reused.
    """
    print(f"[io] Reading AnnData: {h5ad_path}")
    adata = sc.read_h5ad(str(h5ad_path))
    print(adata)

    # ---- Minimal metadata sanity checks (fail fast) ----
    if batch_key not in adata.obs.columns:
        raise KeyError(
            f"Expected batch column adata.obs['{batch_key}'] not found. "
            f"Available obs columns: {list(adata.obs.columns)[:25]} ..."
        )
    if celltype_key not in adata.obs.columns:
        # Not always needed for integration, but you used it downstream for label inspection.
        print(
            f"[warn] cell type column adata.obs['{celltype_key}'] not found. "
            "Continuing without label stats."
        )

    # Make IDs unique (common in merged objects; protects downstream indexing)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # ---- Quick label distribution (no densification) ----
    if celltype_key in adata.obs.columns:
        label_counts = adata.obs[celltype_key].value_counts(dropna=False)
        print(f"[qc] Number of cell types: {label_counts.shape[0]}")
        print("[qc] Top 10 cell types:")
        print(label_counts.head(10))

    # ---- Standard Scanpy preprocessing ----
    # Normalize per-cell library size to 1e4 counts and log1p-transform.
    # This is a typical RNA workflow; if this is CITE-seq multi-modal,
    # you may want modality-specific normalization (not done here).
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select HVGs and subset the matrix (reduces noise and runtime)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)

    # Scale features; cap extreme values to reduce impact of outliers
    sc.pp.scale(adata, max_value=10)

    # PCA provides a low-dimensional dense embedding suitable for Harmony
    sc.tl.pca(adata, svd_solver="arpack", n_comps=n_pcs)

    # ---- Harmony batch correction in PCA space ----
    # harmonypy expects: (cells x pcs) and obs metadata DataFrame
    print(f"[harmony] Running Harmony on adata.obsm['X_pca'] using batch_key='{batch_key}'")
    ho = hm.run_harmony(
        adata.obsm["X_pca"],
        adata.obs,
        batch_key,
        max_iter_harmony=harmony_max_iter,
    )

    # ho.Z_corr is (pcs x cells), transpose to (cells x pcs)
    X_pca_harmony = ho.Z_corr.T
    adata.obsm["X_pca_harmony"] = X_pca_harmony

    # Use a consistent rep name for downstream graph construction
    adata.obsm["X_Harmony"] = adata.obsm["X_pca_harmony"]

    # ---- Neighborhood graph + UMAP on corrected space ----
    sc.pp.neighbors(adata, use_rep="X_Harmony", n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    adata.obsm["X_umap_harmony"] = adata.obsm["X_umap"].copy()

    print("[done] Harmony integration and UMAP completed.")
    return adata


def save_embeddings(adata: sc.AnnData, out_dir: Path) -> None:
    """
    Save key embeddings as .npy. This is the minimal interchange format for downstream ML.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Naming is explicit: what embedding is being saved.
    path_harmony = out_dir / "X_Harmony.npy"          # (cells x pcs)
    path_pca_harmony = out_dir / "X_pca_harmony.npy"  # (cells x pcs), same as X_Harmony here
    path_umap_harmony = out_dir / "X_umap_harmony.npy"  # (cells x 2)

    # Defensive: ensure these keys exist before saving
    for key in ("X_Harmony", "X_pca_harmony", "X_umap_harmony"):
        if key not in adata.obsm:
            raise KeyError(f"Missing adata.obsm['{key}']; pipeline did not generate it.")

    print(f"[save] Writing: {path_harmony}")
    np.save(path_harmony, adata.obsm["X_Harmony"])

    print(f"[save] Writing: {path_pca_harmony}")
    np.save(path_pca_harmony, adata.obsm["X_pca_harmony"])

    print(f"[save] Writing: {path_umap_harmony}")
    np.save(path_umap_harmony, adata.obsm["X_umap_harmony"])

    print(f"[save] Saved embeddings to: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Harmony integration on GSE194122 BMMC processed h5ad.")
    p.add_argument("--workdir", type=Path, default=Path("./data"), help="Working directory for downloads/intermediates.")
    p.add_argument("--download", action="store_true", help="Download the .h5ad.gz from NCBI GEO into workdir.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing downloaded/decompressed files.")
    p.add_argument("--h5ad", type=Path, default=None, help="Path to local .h5ad (if already available).")
    p.add_argument("--outdir", type=Path, default=Path("./outputs/GSE194122"), help="Directory to write .npy outputs.")
    p.add_argument("--batch_key", type=str, default="batch", help="obs column for batch in Harmony.")
    p.add_argument("--celltype_key", type=str, default="cell_type", help="obs column for labels/QC (optional).")
    p.add_argument("--n_hvg", type=int, default=2000, help="Number of highly variable genes.")
    p.add_argument("--n_pcs", type=int, default=50, help="Number of principal components.")
    p.add_argument("--n_neighbors", type=int, default=15, help="Neighbors for graph construction.")
    p.add_argument("--harmony_max_iter", type=int, default=30, help="Harmony maximum iterations.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    # Determine input h5ad path.
    if args.h5ad is not None:
        h5ad_path = args.h5ad
        if not h5ad_path.exists():
            raise FileNotFoundError(f"Provided --h5ad does not exist: {h5ad_path}")
    else:
        gz_path = workdir / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"
        if args.download:
            download_file(GEO_URL, gz_path, overwrite=args.overwrite)
        if not gz_path.exists():
            raise FileNotFoundError(
                f"Missing input .h5ad.gz at {gz_path}. "
                "Either pass --download or provide --h5ad /path/to/file.h5ad"
            )
        h5ad_path = gunzip_if_needed(gz_path, overwrite=args.overwrite)

    adata = run_harmony_pipeline(
        h5ad_path=h5ad_path,
        batch_key=args.batch_key,
        celltype_key=args.celltype_key,
        n_hvg=args.n_hvg,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        harmony_max_iter=args.harmony_max_iter,
    )

    save_embeddings(adata, args.outdir)


if __name__ == "__main__":
    main()
