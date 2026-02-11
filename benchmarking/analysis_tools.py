# analysis_tools.py
# Stable utilities for scRNA-seq benchmarking notebooks.
#
# Compatibility with your notebook:
# - cluster_adata: default names drop leading "X_" from use_rep for generated keys
# - assign_majority_vote_labels: supports new_label_key= (your notebook)
# - calculate_prediction_accuracy: accepts cluster_key= (your notebook)
# - calculate_validation_metrics: returns metric as index (your notebook uses val_df.index)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import importlib.util
import numpy as np
import pandas as pd

# ---- Required deps ----
try:
    import scanpy as sc
except Exception as e:
    raise ImportError("analysis_tools requires `scanpy`. Install: pip install scanpy") from e

try:
    from anndata import AnnData
except Exception as e:
    raise ImportError("analysis_tools requires `anndata`. Install: pip install anndata") from e

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise ImportError("analysis_tools requires `matplotlib`. Install: pip install matplotlib") from e

# ---- Optional deps (checked where needed) ----
try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
        confusion_matrix,
    )
except Exception:
    accuracy_score = None
    f1_score = None
    adjusted_rand_score = None
    normalized_mutual_info_score = None
    confusion_matrix = None

ClusterMethod = Literal["leiden", "louvain"]


# =========================
# Helpers
# =========================

def _has_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_sklearn(func_name: str) -> None:
    if accuracy_score is None:
        raise ImportError(f"{func_name} requires scikit-learn. Install: pip install scikit-learn")


def _ensure_obs_key(adata: AnnData, key: str) -> None:
    if key not in adata.obs.columns:
        raise KeyError(f"obs key '{key}' not found. Available obs keys: {list(adata.obs.columns)}")


def _ensure_obsm_key(adata: AnnData, key: str) -> None:
    if key not in adata.obsm.keys():
        raise KeyError(f"obsm key '{key}' not found. Available obsm keys: {list(adata.obsm.keys())}")


def _sanitize_key(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(s))


def _safe_cat_str(series: pd.Series) -> pd.Series:
    # stable string categorical conversion
    return series.astype("category").astype(str)


def _strip_x_prefix(rep: str) -> str:
    rep = str(rep)
    return rep[2:] if rep.startswith("X_") else rep


def _resolve_cluster_key(adata: AnnData, cluster_key: str) -> str:
    """
    Accepts either 'leiden_harmony' or 'leiden_X_harmony' and resolves to whichever exists in adata.obs.
    """
    if cluster_key in adata.obs.columns:
        return cluster_key

    # toggle X_ after leiden_/louvain_
    for prefix in ("leiden_", "louvain_"):
        if cluster_key.startswith(prefix):
            tail = cluster_key[len(prefix):]
            if tail.startswith("X_"):
                alt = prefix + tail[2:]
            else:
                alt = prefix + "X_" + tail
            if alt in adata.obs.columns:
                return alt

    # generic toggles
    if "X_" not in cluster_key:
        alt = cluster_key.replace("leiden_", "leiden_X_").replace("louvain_", "louvain_X_")
        if alt in adata.obs.columns:
            return alt
    else:
        alt = cluster_key.replace("leiden_X_", "leiden_").replace("louvain_X_", "louvain_")
        if alt in adata.obs.columns:
            return alt

    _ensure_obs_key(adata, cluster_key)  # raise helpful error
    return cluster_key  # unreachable


@dataclass
class _TempObsmSwap:
    """Temporarily swap adata.obsm[target_key] with adata.obsm[source_key]."""
    adata: AnnData
    target_key: str
    source_key: str
    _had_target: bool = False
    _backup: Optional[np.ndarray] = None

    def __enter__(self):
        _ensure_obsm_key(self.adata, self.source_key)
        self._had_target = self.target_key in self.adata.obsm
        if self._had_target:
            self._backup = np.asarray(self.adata.obsm[self.target_key]).copy()
        self.adata.obsm[self.target_key] = np.asarray(self.adata.obsm[self.source_key])
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._had_target:
            self.adata.obsm[self.target_key] = self._backup
        else:
            if self.target_key in self.adata.obsm:
                del self.adata.obsm[self.target_key]
        return False


def _rare_types_from_threshold(y_true: Sequence[str], rare_threshold: float) -> List[str]:
    s = pd.Series(list(y_true), dtype="object").astype(str)
    freq = s.value_counts(normalize=True, dropna=False)
    rare = freq[freq < float(rare_threshold)].index.tolist()
    return [str(x) for x in rare]


def _ensure_plotly(func_name: str) -> None:
    try:
        import plotly.graph_objects as go  # noqa: F401
    except Exception as e:
        raise ImportError(f"{func_name} requires plotly. Install: pip install plotly") from e


# =========================
# 1) Clustering
# =========================

def cluster_adata(
    adata: AnnData,
    *,
    use_rep: str,
    method: ClusterMethod = "leiden",
    n_neighbors: int = 15,
    resolution: float = 1.0,
    cluster_key: Optional[str] = None,
    neighbors_key: Optional[str] = None,
    compute_umap: bool = False,
    umap_key: Optional[str] = None,
    random_state: int = 0,
    copy: bool = False,
    neighbors_kwargs: Optional[Dict[str, Any]] = None,
    cluster_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[AnnData, str, str, Optional[str]]:
    """
    Build neighbors graph on adata.obsm[use_rep] and run clustering.

    Default naming convention:
      use_rep="X_harmony" -> cluster_key="leiden_harmony"
                             neighbors_key="neighbors_harmony"
                             umap_key="X_umap_harmony"
    """
    if copy:
        adata = adata.copy()

    _ensure_obsm_key(adata, use_rep)

    rep_for_names = _strip_x_prefix(use_rep)
    rep_tag = _sanitize_key(rep_for_names)

    if cluster_key is None:
        cluster_key = f"{method}_{rep_tag}"
    if neighbors_key is None:
        neighbors_key = f"neighbors_{rep_tag}"

    neighbors_kwargs = neighbors_kwargs or {}
    cluster_kwargs = cluster_kwargs or {}

    sc.pp.neighbors(
        adata,
        n_neighbors=int(n_neighbors),
        use_rep=use_rep,  # actual embedding key
        key_added=neighbors_key,
        **neighbors_kwargs,
    )

    if method == "leiden":
        if not _has_pkg("igraph"):
            raise ImportError(
                "Leiden clustering requires `igraph`.\n"
                "In Colab:\n"
                "  !pip -q install igraph\n"
            )
        sc.tl.leiden(
            adata,
            resolution=float(resolution),
            neighbors_key=neighbors_key,
            key_added=cluster_key,
            random_state=int(random_state),
            flavor="igraph",
            directed=False,
            n_iterations=2,
            **cluster_kwargs,
        )
    elif method == "louvain":
        sc.tl.louvain(
            adata,
            resolution=float(resolution),
            neighbors_key=neighbors_key,
            key_added=cluster_key,
            random_state=int(random_state),
            **cluster_kwargs,
        )
    else:
        raise ValueError(f"Unsupported clustering method: {method}. Use 'leiden' or 'louvain'.")

    used_umap_key: Optional[str] = None
    if compute_umap:
        if umap_key is None:
            umap_key = f"X_umap_{rep_tag}"
        used_umap_key = umap_key

        had_default = "X_umap" in adata.obsm
        default_backup = np.asarray(adata.obsm["X_umap"]).copy() if had_default else None

        sc.tl.umap(adata, neighbors_key=neighbors_key, random_state=int(random_state))
        adata.obsm[umap_key] = np.asarray(adata.obsm["X_umap"])

        if had_default:
            adata.obsm["X_umap"] = default_backup
        else:
            del adata.obsm["X_umap"]

    return adata, cluster_key, neighbors_key, used_umap_key


# =========================
# 2) Majority-vote: cluster -> label
# =========================

def assign_majority_vote_labels(
    adata: AnnData,
    *,
    cluster_key: str,
    true_label_key: str,
    new_label_key: Optional[str] = None,          # notebook uses this
    predicted_label_key: Optional[str] = None,    # alias
    unknown_label: str = "Unknown",
    copy: bool = False,
) -> Tuple[AnnData, str, pd.DataFrame]:
    """
    Majority vote mapping cluster -> label.

    Writes:
      adata.obs[target_key]
    """
    if copy:
        adata = adata.copy()

    resolved_cluster_key = _resolve_cluster_key(adata, cluster_key)

    target_key = new_label_key if new_label_key is not None else predicted_label_key
    if target_key is None:
        target_key = f"{cluster_key}_majority_label"

    _ensure_obs_key(adata, resolved_cluster_key)
    _ensure_obs_key(adata, true_label_key)

    clusters = _safe_cat_str(adata.obs[resolved_cluster_key])
    labels = _safe_cat_str(adata.obs[true_label_key])

    tab = pd.crosstab(clusters, labels)

    maj_label = tab.idxmax(axis=1).astype(str)
    size = tab.sum(axis=1).astype(int)
    maj_count = tab.max(axis=1).astype(int)
    maj_frac = (maj_count / size).astype(float)

    mapping_df = pd.DataFrame(
        {
            "cluster_id": tab.index.astype(str),
            "majority_label": maj_label.values,
            "majority_fraction": maj_frac.values,
            "cluster_size": size.values,
        }
    )

    mapping = dict(zip(mapping_df["cluster_id"], mapping_df["majority_label"]))
    adata.obs[target_key] = clusters.map(mapping).fillna(unknown_label).astype("category")

    return adata, target_key, mapping_df


# =========================
# 3) Per-cell-type accuracy
# =========================

def calculate_prediction_accuracy(
    adata: AnnData,
    *,
    true_label_key: str,
    predicted_label_key: str,
    cluster_key: Optional[str] = None,  # <-- COMPAT: your notebook passes this
    normalize_confusion: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by cell_type to match your notebook usage.

    Index: cell_type
    Columns:
      - n_true, n_correct, accuracy, top_pred, top_pred_frac
      - (optional) cluster_key_used  (only if cluster_key is provided)
    """
    _ensure_obs_key(adata, true_label_key)
    _ensure_obs_key(adata, predicted_label_key)

    # cluster_key is optional; if provided we resolve it but do not require it
    cluster_key_used: Optional[str] = None
    if cluster_key is not None:
        # If user passes a key that doesn't exist, try to resolve X_ variants; else raise.
        cluster_key_used = _resolve_cluster_key(adata, cluster_key)

    y_true = _safe_cat_str(adata.obs[true_label_key])
    y_pred = _safe_cat_str(adata.obs[predicted_label_key])

    df = pd.DataFrame({"true": y_true, "pred": y_pred})

    n_true = df.groupby("true", observed=True).size().rename("n_true")
    n_correct = df[df["true"] == df["pred"]].groupby("true", observed=True).size().rename("n_correct")

    out = pd.concat([n_true, n_correct], axis=1).fillna(0)
    out["n_true"] = out["n_true"].astype(int)
    out["n_correct"] = out["n_correct"].astype(int)
    out["accuracy"] = (out["n_correct"] / out["n_true"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    top_pred = (
        df.groupby(["true", "pred"], observed=True).size()
        .reset_index(name="count")
        .sort_values(["true", "count"], ascending=[True, False])
        .groupby("true", observed=True)
        .first()
    )
    out["top_pred"] = top_pred["pred"].astype(str)
    out["top_pred_frac"] = (top_pred["count"] / out["n_true"]).astype(float)

    if cluster_key_used is not None:
        out["cluster_key_used"] = cluster_key_used

    if normalize_confusion:
        _require_sklearn("calculate_prediction_accuracy")
        labels_sorted = sorted(set(y_true.unique()).union(set(y_pred.unique())))
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted).astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm / row_sums
        out.attrs["confusion_matrix_normalized"] = {"labels": labels_sorted, "matrix": cm_norm.tolist()}

    out.index.name = "cell_type"
    # sort for convenience
    out = out.sort_values("accuracy", ascending=True)
    return out


# =========================
# 4) Validation metrics (overall + rare)
# =========================

def calculate_validation_metrics(
    adata: AnnData,
    *,
    true_label_key: str,
    cluster_key: str,
    predicted_label_key: str,
    latent_rep_key: Optional[str] = None,
    rare_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by metric (so your notebook can use val_df.index).
    """
    _require_sklearn("calculate_validation_metrics")

    _ensure_obs_key(adata, true_label_key)
    _ensure_obs_key(adata, predicted_label_key)

    resolved_cluster_key = _resolve_cluster_key(adata, cluster_key)
    _ensure_obs_key(adata, resolved_cluster_key)

    y_true = _safe_cat_str(adata.obs[true_label_key])
    y_pred = _safe_cat_str(adata.obs[predicted_label_key])
    y_cluster = _safe_cat_str(adata.obs[resolved_cluster_key])

    metrics: List[Tuple[str, float]] = []

    metrics.append(("overall_accuracy", float(accuracy_score(y_true, y_pred))))
    metrics.append(("overall_macro_f1", float(f1_score(y_true, y_pred, average="macro", zero_division=0))))
    metrics.append(("overall_weighted_f1", float(f1_score(y_true, y_pred, average="weighted", zero_division=0))))

    metrics.append(("clustering_ARI", float(adjusted_rand_score(y_true, y_cluster))))
    metrics.append(("clustering_NMI", float(normalized_mutual_info_score(y_true, y_cluster))))

    rare_types = set(_rare_types_from_threshold(y_true, rare_threshold=rare_threshold))
    is_rare = y_true.isin(list(rare_types)).values

    n_total = int(len(y_true))
    n_rare = int(np.sum(is_rare))
    n_types_total = int(y_true.nunique())
    n_types_rare = int(len(rare_types))

    metrics += [
        ("n_cells_total", float(n_total)),
        ("n_cells_rare", float(n_rare)),
        ("n_types_total", float(n_types_total)),
        ("n_types_rare", float(n_types_rare)),
    ]

    if n_rare > 0 and n_types_rare > 0:
        y_true_r = y_true[is_rare]
        y_pred_r = y_pred[is_rare]
        metrics.append(("rare_accuracy", float(accuracy_score(y_true_r, y_pred_r))))
        metrics.append(("rare_macro_f1", float(f1_score(y_true_r, y_pred_r, average="macro", zero_division=0))))
        metrics.append(("rare_weighted_f1", float(f1_score(y_true_r, y_pred_r, average="weighted", zero_division=0))))
    else:
        metrics.append(("rare_accuracy", 0.0))
        metrics.append(("rare_macro_f1", 0.0))
        metrics.append(("rare_weighted_f1", 0.0))

    if latent_rep_key is not None:
        missing = 0.0
        if latent_rep_key != "X" and (latent_rep_key not in adata.obsm):
            missing = 1.0
        metrics.append(("latent_rep_missing", float(missing)))

    df = pd.DataFrame(metrics, columns=["metric", "value"]).set_index("metric")
    return df


# =========================
# 5) Side-by-side UMAP
# =========================

def plot_side_by_side_umaps(
    adata: AnnData,
    *,
    umap_key1: str,
    umap_key2: str,
    color_key: str,
    title1: str = "Before",
    title2: str = "After",
    highlight_cell_type: Optional[str] = None,
    true_label_key: Optional[str] = None,
    point_size: float = 8.0,
    alpha: float = 0.8,
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 150,
) -> None:
    _ensure_obsm_key(adata, umap_key1)
    _ensure_obsm_key(adata, umap_key2)
    _ensure_obs_key(adata, color_key)

    if highlight_cell_type is not None:
        if true_label_key is None:
            raise ValueError("If highlight_cell_type is set, you must provide true_label_key.")
        _ensure_obs_key(adata, true_label_key)
        mask = (_safe_cat_str(adata.obs[true_label_key]) == str(highlight_cell_type)).values
    else:
        mask = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    def _plot(ax, key, title):
        if mask is None:
            with _TempObsmSwap(adata, "X_umap", key):
                sc.pl.umap(
                    adata,
                    color=color_key,
                    ax=ax,
                    show=False,
                    size=point_size,
                    alpha=alpha,
                    title=title,
                )
        else:
            bg = adata.copy()
            bg.obs["_bg"] = np.where(mask, "highlight", "background")
            with _TempObsmSwap(bg, "X_umap", key):
                sc.pl.umap(
                    bg,
                    color="_bg",
                    ax=ax,
                    show=False,
                    size=point_size,
                    alpha=0.15,
                    title=title,
                )

            hi = adata[mask].copy()
            with _TempObsmSwap(hi, "X_umap", key):
                sc.pl.umap(
                    hi,
                    color=color_key,
                    ax=ax,
                    show=False,
                    size=max(point_size, 10.0),
                    alpha=0.95,
                    title=title,
                )

    _plot(axes[0], umap_key1, title1)
    _plot(axes[1], umap_key2, title2)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# =========================
# 6) Sankey diagram
# =========================

def plot_sankey_diagram(
    adata: AnnData,
    *,
    true_label_key: str,
    predicted_label_key: str,
    title: str = "Sankey: True → Predicted",
    highlight_label: Optional[str] = None,
    min_flow: int = 1,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    _ensure_plotly("plot_sankey_diagram")
    import plotly.graph_objects as go

    _ensure_obs_key(adata, true_label_key)
    _ensure_obs_key(adata, predicted_label_key)

    y_true = _safe_cat_str(adata.obs[true_label_key])
    y_pred = _safe_cat_str(adata.obs[predicted_label_key])

    df = pd.DataFrame({"true": y_true, "pred": y_pred})

    if highlight_label is not None:
        hl = str(highlight_label)
        df = df[(df["true"] == hl) | (df["pred"] == hl)]

    flows = df.groupby(["true", "pred"], observed=True).size().reset_index(name="value")
    flows = flows[flows["value"] >= int(min_flow)]

    fig = go.Figure()
    if flows.empty:
        fig.update_layout(title=f"{title} (no flows after filtering)")
    else:
        true_nodes = sorted(flows["true"].unique().tolist())
        pred_nodes = sorted(flows["pred"].unique().tolist())

        nodes = [f"T:{t}" for t in true_nodes] + [f"P:{p}" for p in pred_nodes]
        node_index = {name: i for i, name in enumerate(nodes)}

        sources = [node_index[f"T:{t}"] for t in flows["true"]]
        targets = [node_index[f"P:{p}"] for p in flows["pred"]]
        values = flows["value"].astype(int).tolist()

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(pad=12, thickness=14, line=dict(width=0.5), label=nodes),
                    link=dict(source=sources, target=targets, value=values),
                )
            ]
        )
        fig.update_layout(title=title, font=dict(size=11))

    if save_path is not None:
        out_path = save_path
        if not out_path.lower().endswith((".html", ".htm")):
            out_path = out_path + ".html"
        fig.write_html(out_path)

    if show:
        fig.show()
