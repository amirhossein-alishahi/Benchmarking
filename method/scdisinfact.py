#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scDisInFact workflow on GSE194122 (OpenProblems NeurIPS 2021 CITE-seq BMMC).

Goal (bioinformatics rationale):
- scDisInFact disentangles (i) shared biological structure, (ii) condition-specific effects,
  and (iii) batch effects from single-cell count data.
- In practice we care about whether "shared-bio" embedding:
    * mixes batches (technical correction)
    * preserves cell-type topology (biological signal)

This script is a cleaned version of a Colab notebook export:
- removes notebook magics (!pip/!git/Colab Drive)
- avoids hard-coded /content paths
- validates required metadata and count matrices
- produces UMAP from shared-bio latent and saves key arrays

NOTE:
- scDisInFact expects raw-ish counts (not log-normalized) as input.
  Here we use adata.X from the processed file. If your dataset stores counts in a layer,
  you should point to that layer instead.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import torch
import pooch

# scDisInFact package imports (assumes scDisInFact is installed in your environment)
from scDisInFact import scdisinfact, create_scdisinfact_dataset


GEO_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE194122&format=file&file="
    "GSE194122%5Fopenproblems%5Fneurips2021%5Fcite%5FBMMC%5Fprocessed%2Eh5ad%2Egz"
)


# -----------------------------
# I/O helpers
# -----------------------------
def download_dataset(url: str, out_dir: Path, fname: str) -> Path:
    """
    Download with pooch to get caching + resumable-ish behavior.
    For production pipelines, add checksum pinning (known_hash) for provenance.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = pooch.retrieve(url=url, known_hash=None, path=str(out_dir), fname=fname)
    return Path(path)


def gunzip(gz_path: Path, out_path: Optional[Path] = None, overwrite: bool = False) -> Path:
    """Decompress .gz -> output path (stream copy)."""
    if out_path is None:
        out_path = gz_path.with_suffix("")  # drop .gz

    if out_path.exists() and not overwrite:
        print(f"[gunzip] Exists, skipping: {out_path}")
        return out_path

    print(f"[gunzip] Decompressing: {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_path


# -----------------------------
# Metadata / QC helpers
# -----------------------------
def require_obs_columns(adata: anndata.AnnData, cols: List[str]) -> None:
    missing = [c for c in cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"Missing required adata.obs columns: {missing}. "
            f"Available columns (first 30): {list(adata.obs.columns)[:30]}"
        )


def coerce_obs_str(adata: anndata.AnnData, cols: List[str]) -> None:
    """
    scDisInFact condition/batch covariates are safest as strings/categoricals.
    We coerce to string to avoid pandas category edge cases from h5ad reading.
    """
    for c in cols:
        adata.obs[c] = adata.obs[c].astype(str)


# -----------------------------
# Core pipeline
# -----------------------------
def run_scdisinfact(
    h5ad_path: Path,
    condition_1_col: str,
    condition_2_col: str,
    batch_col: str,
    Ks: List[int],
    nepochs: int,
    n_neighbors: int,
    out_dir: Path,
    device: str,
) -> anndata.AnnData:
    """
    Train scDisInFact and compute UMAP on shared-bio latent space.

    Ks interpretation (from scDisInFact docs/usage):
    - Ks[0]: shared biological factor dimension (z_c)
    - Ks[1:]: condition-specific factor dimensions (z_d for each condition)
    """
    print(f"[io] Reading AnnData: {h5ad_path}")
    adata = sc.read_h5ad(str(h5ad_path))
    print(adata)

    # ---- validate + select metadata ----
    # We keep only what we actually model, to reduce accidental leakage/complexity.
    keep_cols = [condition_1_col, condition_2_col, batch_col, "cell_type"] if "cell_type" in adata.obs.columns else [condition_1_col, condition_2_col, batch_col]
    require_obs_columns(adata, [condition_1_col, condition_2_col, batch_col])
    adata.obs = adata.obs[keep_cols].copy()

    # Rename to the keys used downstream (makes the dataset-creation call stable)
    adata.obs = adata.obs.rename(columns={condition_1_col: "condition_1", condition_2_col: "condition_2"})

    # Consistent dtype handling: conditions/batch as strings
    coerce_obs_str(adata, ["condition_1", "condition_2", batch_col])

    # ---- counts matrix ----
    # This script assumes counts live in adata.X.
    # If you have a raw-count layer, replace with `adata.layers["counts"]`.
    counts = adata.X

    # ---- create scDisInFact dataset dict ----
    # scDisInFact internally normalizes counts; we pass the raw matrix here.
    meta_cells = adata.obs.copy()
    # scDisInFact expects a DataFrame with matching row order to counts.
    # Since meta_cells comes from adata.obs, the index ordering is aligned.
    data_dict = create_scdisinfact_dataset(
        counts=counts,
        meta_cells=meta_cells,
        condition_key=["condition_1", "condition_2"],
        batch_key=batch_col,
    )

    # ---- train model ----
    torch_device = torch.device(device)
    print(f"[train] Using device: {torch_device}")
    print(f"[train] Ks={Ks} | epochs={nepochs}")

    model = scdisinfact(data_dict=data_dict, Ks=Ks, device=torch_device)
    _losses = model.train_model(nepochs=nepochs)
    _ = model.eval()

    # ---- extract latents (one pass over datasets) ----
    # We care primarily about shared-bio (mu_c). Condition-specific (mu_d) can also be saved.
    z_c_all = []
    z_d_cond1_all = []
    z_d_cond2_all = []

    for dataset in data_dict["datasets"]:
        with torch.no_grad():
            inf = model.inference(
                counts=dataset.counts_norm.to(model.device),
                batch_ids=dataset.batch_id[:, None].to(model.device),
                print_stat=False,
            )
            mu_c = inf["mu_c"]                 # shared-bio
            mu_d = inf["mu_d"]                 # list: one per condition type

            z_c_all.append(mu_c.cpu().numpy())
            z_d_cond1_all.append(mu_d[0].cpu().numpy())
            z_d_cond2_all.append(mu_d[1].cpu().numpy())

    z_c = np.concatenate(z_c_all, axis=0)
    z_d1 = np.concatenate(z_d_cond1_all, axis=0)
    z_d2 = np.concatenate(z_d_cond2_all, axis=0)

    # Name embeddings honestly: these are scDisInFact latents, not scVI.
    adata.obsm["X_scdisinfact_shared"] = z_c
    adata.obsm["X_scdisinfact_cond1"] = z_d1
    adata.obsm["X_scdisinfact_cond2"] = z_d2

    # ---- neighborhood graph + UMAP on shared latent ----
    # This is our main visualization to assess batch mixing / biological structure.
    sc.pp.neighbors(adata, use_rep="X_scdisinfact_shared", n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    adata.obsm["X_umap_scdisinfact_shared"] = adata.obsm["X_umap"].copy()

    # ---- save outputs ----
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_scdisinfact_shared.npy", adata.obsm["X_scdisinfact_shared"])
    np.save(out_dir / "X_scdisinfact_cond1.npy", adata.obsm["X_scdisinfact_cond1"])
    np.save(out_dir / "X_scdisinfact_cond2.npy", adata.obsm["X_scdisinfact_cond2"])
    np.save(out_dir / "X_umap_scdisinfact_shared.npy", adata.obsm["X_umap_scdisinfact_shared"])

    # Also save metadata for downstream benchmarking (cluster purity, ARI/NMI, etc.)
    adata.obs.to_csv(out_dir / "obs_metadata.csv", index=True)

    print("[save] Wrote:")
    print("  - X_scdisinfact_shared.npy", adata.obsm["X_scdisinfact_shared"].shape)
    print("  - X_scdisinfact_cond1.npy ", adata.obsm["X_scdisinfact_cond1"].shape)
    print("  - X_scdisinfact_cond2.npy ", adata.obsm["X_scdisinfact_cond2"].shape)
    print("  - X_umap_scdisinfact_shared.npy", adata.obsm["X_umap_scdisinfact_shared"].shape)
    print("  - obs_metadata.csv")

    return adata


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run scDisInFact on GSE194122 processed BMMC h5ad.")
    p.add_argument("--workdir", type=Path, default=Path("./data"), help="Where to download/decompress data.")
    p.add_argument("--download", action="store_true", help="Download the GEO .h5ad.gz first.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite decompressed file if it exists.")
    p.add_argument("--h5ad", type=Path, default=None, help="Use local .h5ad (skip download).")
    p.add_argument("--outdir", type=Path, default=Path("./outputs/GSE194122_scdisinfact"), help="Output directory.")

    # Columns in the original dataset (based on your notebook)
    p.add_argument("--condition1", type=str, default="DonorGender", help="obs column for condition 1.")
    p.add_argument("--condition2", type=str, default="DonorSmoker", help="obs column for condition 2.")
    p.add_argument("--batch", type=str, default="batch", help="obs column for batch.")

    p.add_argument("--Ks", type=int, nargs=3, default=[8, 2, 2], help="Latent dims: [shared, cond1, cond2].")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve input file
    if args.h5ad is not None:
        h5ad_path = args.h5ad
        if not h5ad_path.exists():
            raise FileNotFoundError(f"--h5ad not found: {h5ad_path}")
    else:
        if not args.download:
            raise ValueError("Provide --h5ad or pass --download to fetch the dataset.")

        gz_path = download_dataset(
            url=GEO_URL,
            out_dir=args.workdir,
            fname="GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz",
        )
        h5ad_path = gunzip(gz_path, overwrite=args.overwrite)

    _adata = run_scdisinfact(
        h5ad_path=h5ad_path,
        condition_1_col=args.condition1,
        condition_2_col=args.condition2,
        batch_col=args.batch,
        Ks=list(args.Ks),
        nepochs=args.epochs,
        n_neighbors=args.neighbors,
        out_dir=args.outdir,
        device=args.device,
    )

    # Optional quick-look plots (safe: only runs if scanpy plotting backend is available)
    # For HPC/headless: consider sc.settings.set_figure_params() + savefig.
    if "cell_type" in _adata.obs.columns:
        sc.pl.umap(_adata, color=["cell_type", args.batch], wspace=0.4, frameon=False)


if __name__ == "__main__":
    main()
