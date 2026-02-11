# Single-Cell Integration Benchmark Suite

This repository provides a unified, reproducible framework for running and comparing multiple **single-cell integration and batch-correction methods** on .h5ad datasets

The project separates **method execution** and **benchmarking logic** into dedicated directories, ensuring modularity and consistent evaluation across tools.

---

## Repository Structure

```
project_root/
│
├── method/
│   ├── harmony.py
│   ├── mrvi.py
│   ├── scdisinfact.py
│   ├── scdml.py
│   ├── scgen.py
│   ├── scvi.py
│   └── seurat.ipynb
│
└── benchmark/
    ├── analysis_tools.py          # Unified evaluation utilities
    ├── benchmarking_allmethods.ipynb
    └── requirements_analysis_tools.txt
```

---

## Purpose

This framework enables:

### 1. Consistent execution of diverse integration methods

Each script in `method/` performs:

* loading of an `.h5ad` dataset
* integration / batch correction
* generation of a latent embedding
* creation of a UMAP representation
* export of standardized metadata

### 2. Centralized benchmarking

`benchmark/analysis_tools.py` provides a unified API for:

* clustering with Leiden
* majority-vote label transfer
* UMAP comparison utilities
* ARI, AMI, ASW, and purity-related metrics
* rare cell–aware evaluations
* optional Sankey visualizations

### 3. Notebook-driven comparison

`benchmark/benchmarking_allmethods.ipynb` performs:

* method-by-method embedding loading
* standardized evaluation
* metric comparisons
* UMAP visualization
* optional Sankey analysis

---

## Installation

### 1. Create an environment

```bash
conda create -n scbench python=3.10
conda activate scbench
```

### 2. Install method dependencies

Install integration tools required by the scripts you want to run:

```bash
pip install scanpy scvi-tools scgen harmonypy
pip install scdml scdisinfact
```

Some frameworks (e.g., scVI/scGen) may require appropriate PyTorch installations depending on GPU/CPU.

### 3. Install benchmarking dependencies

```
pip install -r benchmark/requirements_analysis_tools.txt
```

This includes:

* `igraph`, `leidenalg`
* `plotly`
* `umap-learn`, `scikit-learn`
* `statsmodels`

Everything required for `analysis_tools.py`.

---

## Usage Overview

### 1. Run integration methods

Each script follows a standard CLI pattern:

```bash
python method/scvi.py --h5ad path/to/data.h5ad --outdir path/to/save
```

Other examples:

```bash
python method/harmony.py --h5ad path/to/data.h5ad --outdir ...
python method/scgen.py   --h5ad path/to/data.h5ad --outdir ...
python method/mrvi.py    --h5ad path/to/data.h5ad --outdir ...
python method/scdml.py   --h5ad path/to/data.h5ad --outdir ...
python method/scdisinfact.py --h5ad path/to/data.h5ad --outdir ...
```


### 2. Benchmark all methods

Open:

```
benchmark/benchmarking_allmethods.ipynb
```

The notebook:

* loads all embeddings
* runs identical clustering and evaluation via `analysis_tools.py`
* computes all metrics
* generates visual comparisons

---


## Acknowledgements

This framework standardizes evaluation across widely used single-cell integration tools, including:

* Harmony
* scVI
* scGen
* MRVI
* scDisInFact
* scDML
* Seurat

It is designed around common single-cell multi-omic benchmarks such as
**OpenProblems NeurIPS 2021 CITE-seq BMMC (GSE194122)**.
