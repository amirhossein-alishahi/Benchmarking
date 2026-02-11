#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MRVI integration workflow for GSE194122 (OpenProblems NeurIPS 2021 CITE-seq BMMC).

This script is a cleaned version of a Colab notebook export.
Major goals:
  - Make it runnable as a standard Python script (no notebook magics / no Colab-only dependencies).
  - Add "research-grade" comments documenting assumptions and failure modes.
  - Fail fast when required metadata or count layers are missing.

Biological context (why we do this):
  - MRVI models sample-level (donor/patient) variation separately from technical batch effects.
  - We train on raw counts (typically in a dedicated AnnData layer) and then extract:
        u: sample-unaware latent (biological structure without sample-specific shift)
        z: sample-aware latent (captures sample variation)
  - We visualize u and z via UMAP to assess mixing by batch and preservation of cell types.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scanpy as sc
import scvi
from scvi.external import MRVI


GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/"
    "GSE194122%5Fopenproblems%5Fneurips2021%5Fcite%5FBMMC%5Fprocessed.h5ad.gz"
)


# ----------------------------
# I/O utilities (boring but essential for reproducibility)
# ----------------------------
def download_if_needed(url: str, out_path: Path, overwrite: bool = False) -> None:
    """Download a file using standard library tools (no `wget`/shell magics)."""
    if out_path.exists() and not overwrite:
        print(f"[download] Exists, skipping: {out_path}")
        return

    print(f"[download] Fetching: {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from urllib.request import urlretrieve
    urlretrieve(url, out_path)

    print(f"[download] Saved: {out_path}")


def gunzip_if_needed(gz_path: Path, out_path: Optional[Path] = None, overwrite: bool = False) -> Path:
    """Decompress .gz → decompressed file (streaming; avoids loading whole file into RAM)."""
    if out_path is None:
        out_path = gz_path.with_suffix("")  # drop only .gz

    if out_path.exists() and not overwrite:
        print(f"[gunzip] Exists, skipping: {out_path}")
        return out_path

    print(f"[gunzip] Decompressing: {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_path


# ----------------------------
# Metadata helpers
# ----------------------------
def choose_first_existing(obs_columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate key present in obs columns."""
    s = set(obs_columns)
    for k in candidates:
        if k in s:
            return k
    return None


def require_obs_key(adata: sc.AnnData, key: str) -> None:
    """Fail fast if required obs column is missing."""
    if key not in adata.obs.columns:
        raise KeyError(
            f"Required obs column '{key}' not found. "
            f"Available columns (first 30): {list(adata.obs.columns)[:30]}"
        )


def require_layer(adata: sc.AnnData, layer: str) -> None:
    """Fail fast if required layer is missing."""
    if adata.layers is None or layer not in adata.layers.keys():
        available = list(adata.layers.keys()) if adata.layers is not None else []
        raise ValueError(
            f"Counts layer '{layer}' not found. Available layers: {available}. "
            "MRVI expects raw counts (not log-normalized values). "
            "If your counts are in .X, either move them to a layer or set --counts_layer X."
        )


# ----------------------------
# Plotting helpers
# ----------------------------
def plot_embedding(
    adata: sc.AnnData,
    obsm_key: str,
    colors: Sequence[str],
    title: str,
    point_size: int = 5,
) -> None:
    """
    Scanpy expects embedding bases named like 'X_*'. To avoid polluting AnnData,
    we temporarily alias the target embedding into a safe name, plot, then delete it.
    """
    if obsm_key not in adata.obsm:
        raise KeyError(f"Embedding '{obsm_key}' not found in adata.obsm")

    tmp_key = "X_tmp_plot"
    adata.obsm[tmp_key] = adata.obsm[obsm_key]

    sc.pl.embedding(
        adata,
        basis=tmp_key,
        color=list(colors),
        s=point_size,
        frameon=False,
        ncols=2,
        title=title,
    )

    del adata.obsm[tmp_key]


def ensure_pre_umap(adata: sc.AnnData, prefer_modality: str = "GEX", n_neighbors: int = 15) -> None:
    """
    Some OpenProblems AnnData objects store modality-specific embeddings.
    This function standardizes a "pre-integration" UMAP to `adata.obsm['X_umap_pre']`.

    If none exists, we try to build one from an existing PCA embedding.
    """
    if "X_umap_pre" in adata.obsm:
        print("[umap] Using existing adata.obsm['X_umap_pre']")
        return

    # Known conventions in some multi-omic pipelines
    if prefer_modality.upper() == "GEX":
        candidates = ["GEX_X_umap", "X_umap", "ADT_X_umap"]
        pca_key = "GEX_X_pca"
    else:
        candidates = ["ADT_X_umap", "X_umap", "GEX_X_umap"]
        pca_key = "ADT_X_pca"

    for key in candidates:
        if key in adata.obsm:
            adata.obsm["X_umap_pre"] = adata.obsm[key].copy()
            print(f"[umap] Aliased '{key}' -> 'X_umap_pre'")
            return

    # Fallback: build UMAP from PCA if present
    if pca_key in adata.obsm:
        print(f"[umap] Building X_umap_pre from '{pca_key}'")
        sc.pp.neighbors(adata, use_rep=pca_key, key_added="neighbors_pre", n_neighbors=n_neighbors)
        sc.tl.umap(adata, neighbors_key="neighbors_pre")
        adata.obsm["X_umap_pre"] = adata.obsm["X_umap"].copy()
        return

    raise ValueError("[umap] No suitable pre-UMAP found and could not build from PCA.")


# ----------------------------
# MRVI pipeline
# ----------------------------
def run_mrvi(
    adata: sc.AnnData,
    counts_layer: str,
    batch_key: str,
    sample_key: str,
    n_hvg: int = 10_000,
    n_neighbors: int = 15,
    max_epochs: int = 400,
    seed: int = 0,
) -> sc.AnnData:
    """
    Train MRVI and compute UMAPs on latent representations.

    Notes:
    - HVG selection is performed prior to model training to reduce compute.
      This is common in scVI-family workflows, but it is still an analysis choice.
    - For multi-omic objects, ensure you are modelling the intended modality
      (this script assumes counts correspond to the expression matrix in the chosen layer).
    """
    scvi.settings.seed = seed
    sc.settings.verbosity = 2

    print("[versions] scvi-tools:", scvi.__version__)
    print("[input] ", adata)

    # --- validate critical inputs ---
    require_layer(adata, counts_layer)
    require_obs_key(adata, batch_key)
    require_obs_key(adata, sample_key)

    # --- gene selection ---
    # We typically pick HVGs on raw-ish data; seurat_v3 flavor is standard for scRNA-seq.
    # Subsetting genes changes the model input space; document this in your methods.
    print(f"[hvg] Selecting top {n_hvg} HVGs (subset=True)")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, inplace=True, subset=True, flavor="seurat_v3")

    # --- register AnnData with scvi-tools ---
    # MRVI needs a *sample* (donor/patient) identifier; batch_key is optional but recommended.
    # Forcing backend="torch" avoids accidental JAX backend usage in mixed environments.
    print(f"[setup] MRVI.setup_anndata(sample_key='{sample_key}', batch_key='{batch_key}', layer='{counts_layer}')")
    MRVI.setup_anndata(
        adata,
        sample_key=sample_key,
        batch_key=batch_key,
        layer=counts_layer,
        backend="torch",
    )

    # --- train model ---
    print(f"[train] Training MRVI (max_epochs={max_epochs})")
    model = MRVI(adata, backend="torch")
    model.train(max_epochs=max_epochs)

    # --- extract latents ---
    # u: sample-unaware (often used for integrated structure)
    # z: sample-aware (captures sample variation)
    print("[latent] Extracting u (sample-unaware) and z (sample-aware)")
    adata.obsm["X_mrvi_u"] = model.get_latent_representation(give_z=False)
    adata.obsm["X_mrvi_z"] = model.get_latent_representation(give_z=True)

    # --- UMAPs on latent spaces ---
    print("[umap] Computing UMAP on u")
    sc.pp.neighbors(adata, use_rep="X_mrvi_u", key_added="neighbors_mrvi_u", n_neighbors=n_neighbors)
    sc.tl.umap(adata, neighbors_key="neighbors_mrvi_u")
    adata.obsm["X_umap_mrvi_u"] = adata.obsm["X_umap"].copy()

    print("[umap] Computing UMAP on z")
    sc.pp.neighbors(adata, use_rep="X_mrvi_z", key_added="neighbors_mrvi_z", n_neighbors=n_neighbors)
    sc.tl.umap(adata, neighbors_key="neighbors_mrvi_z")
    adata.obsm["X_umap_mrvi_z"] = adata.obsm["X_umap"].copy()

    return adata


def save_outputs(adata: sc.AnnData, out_dir: Path) -> None:
    """
    Save model outputs to .npy for downstream ML / benchmarking.

    We save both latent (high-dim) and UMAP (2D) representations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    needed = ["X_mrvi_u", "X_mrvi_z", "X_umap_mrvi_u", "X_umap_mrvi_z"]
    for k in needed:
        if k not in adata.obsm:
            raise KeyError(f"Missing adata.obsm['{k}']; pipeline did not generate it.")

    np.save(out_dir / "X_mrvi_u.npy", adata.obsm["X_mrvi_u"])
    np.save(out_dir / "X_mrvi_z.npy", adata.obsm["X_mrvi_z"])
    np.save(out_dir / "X_umap_mrvi_u.npy", adata.obsm["X_umap_mrvi_u"])
    np.save(out_dir / "X_umap_mrvi_z.npy", adata.obsm["X_umap_mrvi_z"])

    print(f"[save] Wrote outputs to: {out_dir}")
    for k in needed:
        print(f"  - {k}: {adata.obsm[k].shape}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MRVI integration on GSE194122 BMMC processed AnnData.")
    p.add_argument("--workdir", type=Path, default=Path("./data"), help="Directory for downloaded/decompressed data.")
    p.add_argument("--download", action="store_true", help="Download the dataset from GEO.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite downloaded/decompressed files if present.")
    p.add_argument("--h5ad", type=Path, default=None, help="Path to local .h5ad (skip download if provided).")

    p.add_argument("--counts_layer", type=str, default="counts", help="AnnData layer containing raw counts.")
    p.add_argument("--batch_key", type=str, default="batch", help="adata.obs column for batch.")
    p.add_argument("--sample_key", type=str, default=None, help="adata.obs column for sample/donor/patient.")

    p.add_argument("--prefer_modality", type=str, default="GEX", choices=["GEX", "ADT"], help="Which modality to use for pre-UMAP discovery.")
    p.add_argument("--outdir", type=Path, default=Path("./outputs/GSE194122_mrvi"), help="Where to write .npy outputs.")

    p.add_argument("--n_hvg", type=int, default=10_000)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--plot", action="store_true", help="Generate UMAP plots (requires a display backend).")
    p.add_argument("--celltype_key", type=str, default="cell_type", help="adata.obs column used for plotting cell types.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- resolve input file ---
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    if args.h5ad is not None:
        h5ad_path = args.h5ad
        if not h5ad_path.exists():
            raise FileNotFoundError(f"--h5ad not found: {h5ad_path}")
    else:
        gz_path = workdir / "BMMC_processed.h5ad.gz"
        h5ad_path = workdir / "BMMC_processed.h5ad"
        if args.download:
            download_if_needed(GEO_URL, gz_path, overwrite=args.overwrite)
        if not gz_path.exists() and not h5ad_path.exists():
            raise FileNotFoundError(
                f"No input file found. Either pass --download or provide --h5ad.\n"
                f"Expected: {gz_path} or {h5ad_path}"
            )
        if not h5ad_path.exists():
            h5ad_path = gunzip_if_needed(gz_path, out_path=h5ad_path, overwrite=args.overwrite)

    print(f"[io] Reading: {h5ad_path}")
    adata = sc.read_h5ad(str(h5ad_path))

    # --- sample_key selection ---
    # MRVI needs a sample-level covariate. In practice this is donor/patient/subject.
    # If the user doesn't provide one, we try a reasonable set of common conventions.
    if args.sample_key is None:
        candidate_keys = [
            "DonorID", "donor", "donor_id",
            "patient", "patient_id",
            "subject", "subject_id",
            "individual", "individual_id",
            "sample", "sample_id",
            args.batch_key,  # last resort: treat batch as sample if dataset is structured that way
        ]
        chosen = choose_first_existing(adata.obs.columns, candidate_keys)
        if chosen is None:
            raise ValueError(
                "Could not auto-detect a sample_key in adata.obs. "
                f"Tried: {candidate_keys}. Provide --sample_key explicitly."
            )
        sample_key = chosen
    else:
        sample_key = args.sample_key

    print(f"[config] counts_layer={args.counts_layer} | batch_key={args.batch_key} | sample_key={sample_key}")

    # --- pre-UMAP for before/after comparison ---
    ensure_pre_umap(adata, prefer_modality=args.prefer_modality, n_neighbors=args.n_neighbors)

    # --- run MRVI ---
    adata = run_mrvi(
        adata=adata,
        counts_layer=args.counts_layer,
        batch_key=args.batch_key,
        sample_key=sample_key,
        n_hvg=args.n_hvg,
        n_neighbors=args.n_neighbors,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )

    # --- optional plotting ---
    if args.plot:
        # We keep plotting optional because headless environments (HPC) often lack display.
        # For papers, we typically export figures to files here (pdf/png).
        colors = []
        if args.celltype_key in adata.obs.columns:
            colors.append(args.celltype_key)
        if args.batch_key in adata.obs.columns:
            colors.append(args.batch_key)

        if colors:
            plot_embedding(adata, "X_umap_pre", colors, "Before integration (pre)")
            plot_embedding(adata, "X_umap_mrvi_u", colors, "After MRVI — u (sample-unaware)")
            plot_embedding(adata, "X_umap_mrvi_z", colors, "After MRVI — z (sample-aware)")
        else:
            print("[plot] No valid color keys found; skipping plots.")

    # --- save outputs ---
    save_outputs(adata, args.outdir)


if __name__ == "__main__":
    main()
