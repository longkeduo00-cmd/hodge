from __future__ import annotations

"""Low-order alpha sweep on the real-network benchmark.

Each alpha value is executed through the existing real-domain validation runner.
Consequently every per-alpha run retains the complete previous output contract:
raw distance matrices, unsupervised cluster heatmap, domain-block descriptive
heatmap, global MST, within-domain heatmap/MST panels, MDS, retrieval tables,
cluster assignments, all real metrics, and method-comparison plots.

Only ``method.alpha`` is varied.  Upstream Hodge Q7 profiles are shared because
alpha is not part of the profile cache identity.  Baseline settings are kept
identical across alpha values and audited for numerical invariance.
"""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
import json
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hodge_vnext.cache import CACHE_SCHEMA
from hodge_vnext.utils import stable_hash, write_json

from .config import method_from_payload, resolve_path
from .real_domains import _prepare_manifest, run_real_domain_validation
from .real_plotting import plot_core_metrics_comparison, plot_domain_metrics_comparison
from .style import DMEASURE_CMAP, FIG_DPI, LABEL_SIZE, TICK_SIZE, TITLE_SIZE, apply_style


REAL_ALPHA_SWEEP_SCHEMA = "hodge-real-alpha-low-order-sweep-v2"
CORE_COLUMNS = [
    ("auroc_same_domain", "AUROC"),
    ("nn1_accuracy", "1-NN accuracy"),
    ("nn1_macro_f1", "Macro-F1"),
    ("precision_at_1", "P@1"),
    ("mAP", "mAP"),
    ("ndcg_at_7", "NDCG@7"),
]
DOMAIN_COLUMNS = [
    ("nn1_accuracy", "1-NN accuracy"),
    ("mAP", "mAP"),
    ("ndcg_at_7", "NDCG@7"),
]


def _alpha_slug(alpha: float) -> str:
    s = f"{float(alpha):.6g}".replace("-", "m").replace(".", "p")
    return f"a{s}"


def _validate_alphas(values: list[float]) -> list[float]:
    out = [float(x) for x in values]
    if not out or not all(math.isfinite(x) for x in out):
        raise ValueError("alpha_sweep.alphas must be a non-empty finite list")
    if len(set(out)) != len(out):
        raise ValueError("alpha_sweep.alphas must be unique")
    return out


def _savefig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _numeric_baseline_audit(metrics: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame(rows)
    numeric = [c for c in metrics.select_dtypes(include=[np.number]).columns if c != "alpha"]
    for method, grp in metrics[metrics["method"] != "Hodge"].groupby("method", sort=False):
        for col in numeric:
            vals = grp[col].dropna().to_numpy(float)
            if not len(vals):
                continue
            span = float(np.max(vals) - np.min(vals))
            rows.append({"method": str(method), "metric": col, "span_across_alpha": span, "tolerance": tol, "passed": span <= tol})
    return pd.DataFrame(rows)


def _distance_stability(run_rows: pd.DataFrame, reference_alpha: float) -> pd.DataFrame:
    if run_rows.empty:
        return pd.DataFrame()
    ref_match = run_rows[np.isclose(run_rows["alpha"].to_numpy(float), float(reference_alpha))]
    if ref_match.empty:
        raise ValueError(f"reference_alpha={reference_alpha} is not present in the alpha sweep")
    ref_path = Path(str(ref_match.iloc[0]["run_dir"])) / "arrays" / "distance_matrix_Hodge.npy"
    Dref = np.load(ref_path)
    tri = np.triu_indices(len(Dref), 1)
    r = np.asarray(Dref[tri], dtype=float)
    rows = []
    for item in run_rows.itertuples(index=False):
        D = np.load(Path(str(item.run_dir)) / "arrays" / "distance_matrix_Hodge.npy")
        x = np.asarray(D[tri], dtype=float)
        rho = 1.0 if np.array_equal(x, r) else float(spearmanr(r, x).statistic)
        fro = float(np.linalg.norm(D - Dref))
        denom = float(np.linalg.norm(Dref))
        rows.append({
            "alpha": float(item.alpha),
            "reference_alpha": float(reference_alpha),
            "spearman_vs_reference_distance": rho,
            "frobenius_difference": fro,
            "relative_frobenius_difference": (fro / denom if denom > 0 else 0.0),
            "mean_raw_distance": float(np.mean(x)) if len(x) else 0.0,
            "median_raw_distance": float(np.median(x)) if len(x) else 0.0,
        })
    return pd.DataFrame(rows)


def _plot_metric_trends(hodge: pd.DataFrame, plots: Path) -> None:
    if hodge.empty:
        return
    apply_style()
    for col, label in CORE_COLUMNS:
        if col not in hodge.columns:
            continue
        s = hodge.sort_values("alpha", ascending=False)
        fig, ax = plt.subplots(figsize=(13.5, 8.5))
        ax.plot(s["alpha"], s[col], marker="o", linewidth=2.8, markersize=8.5)
        ax.set_xlabel(r"$\alpha$ (smaller = stronger low-order emphasis)")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Real-network Hodge alpha sweep — {label}", fontsize=TITLE_SIZE, fontweight="bold")
        ax.grid(alpha=0.18)
        _savefig(fig, plots / f"hodge_alpha_{col}")

    present = [(c, l) for c, l in CORE_COLUMNS if c in hodge.columns]
    M = hodge.sort_values("alpha", ascending=False)[[c for c, _ in present]].to_numpy(float)
    ys = hodge.sort_values("alpha", ascending=False)["alpha"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(max(11, 1.7 * len(present) + 5), max(7, 0.9 * len(ys) + 4)))
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=DMEASURE_CMAP, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(present)), [l for _, l in present])
    ax.set_yticks(np.arange(len(ys)), [f"{a:g}" for a in ys])
    ax.set_xlabel("Real-domain / retrieval metric")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title("Real-network Hodge performance across low-order alpha weighting", fontsize=TITLE_SIZE, fontweight="bold")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Score", fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    _savefig(fig, plots / "hodge_alpha_core_metrics_heatmap")


def _plot_domain_trends(domain: pd.DataFrame, plots: Path) -> None:
    if domain.empty:
        return
    apply_style()
    domains = list(dict.fromkeys(domain["domain"].astype(str).tolist()))
    for col, label in DOMAIN_COLUMNS:
        if col not in domain.columns:
            continue
        fig, ax = plt.subplots(figsize=(14.5, 9))
        for d in domains:
            s = domain[domain["domain"] == d].sort_values("alpha", ascending=False)
            ax.plot(s["alpha"], s[col], marker="o", linewidth=2.5, markersize=7.5, label=d)
        ax.set_xlabel(r"$\alpha$ (smaller = stronger low-order emphasis)")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Real-network Hodge alpha sweep by domain — {label}", fontsize=TITLE_SIZE, fontweight="bold")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.18)
        _savefig(fig, plots / f"hodge_alpha_by_domain_{col}")


def _plot_distance_stability(stability: pd.DataFrame, plots: Path) -> None:
    if stability.empty:
        return
    apply_style()
    s = stability.sort_values("alpha", ascending=False)
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    ax.plot(s["alpha"], s["spearman_vs_reference_distance"], marker="o", linewidth=2.8, markersize=8.5)
    ax.set_xlabel(r"$\alpha$ (smaller = stronger low-order emphasis)")
    ax.set_ylabel("Spearman correlation with reference distance geometry")
    ax.set_ylim(-0.05, 1.05)
    ref = float(s["reference_alpha"].iloc[0])
    ax.set_title(f"Real-network distance geometry stability vs alpha={ref:g}", fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(alpha=0.18)
    _savefig(fig, plots / "hodge_alpha_distance_spearman_vs_reference")

    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    ax.plot(s["alpha"], s["relative_frobenius_difference"], marker="o", linewidth=2.8, markersize=8.5)
    ax.set_xlabel(r"$\alpha$ (smaller = stronger low-order emphasis)")
    ax.set_ylabel("Relative Frobenius difference")
    ax.set_title(f"Real-network raw-distance change vs alpha={ref:g}", fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(alpha=0.18)
    _savefig(fig, plots / "hodge_alpha_distance_relative_frobenius")


def run_real_alpha_sweep(payload: dict[str, Any], config_path: Path, *, force: bool = False) -> Path:
    cfg = method_from_payload(payload, config_path)
    sweep = dict(payload.get("real_alpha_sweep", {}))
    alphas = _validate_alphas(sweep.get("alphas", [1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0]))
    reference_alpha = float(sweep.get("reference_alpha", 1.0))
    baseline_policy = str(sweep.get("baseline_policy", "reference_only"))
    if baseline_policy not in {"reference_only", "all_methods_each_alpha"}:
        raise ValueError("real_alpha_sweep.baseline_policy must be reference_only or all_methods_each_alpha")
    if not any(math.isclose(a, reference_alpha, rel_tol=0.0, abs_tol=1e-15) for a in alphas):
        raise ValueError("real_alpha_sweep.reference_alpha must be included in real_alpha_sweep.alphas")

    selected = _prepare_manifest(payload, config_path)
    labels = selected["label"].astype(str).tolist()
    domains = selected["domain"].astype(str).tolist()
    neutral_cfg = replace(cfg, alpha=reference_alpha)
    identity = {
        "schema": REAL_ALPHA_SWEEP_SCHEMA,
        "alphas": alphas,
        "reference_alpha": reference_alpha,
        "labels": labels,
        "domains": domains,
        "method_without_alpha": {k: v for k, v in asdict(neutral_cfg).items() if k != "alpha"},
        "methods": list(payload.get("real_validation", {}).get("methods", [])),
        "baseline_policy": baseline_policy,
    }
    root = resolve_path(sweep.get("results_root", "results/validation/real_alpha_sweep"), config_path)
    run = root / f"real_alpha_sweep_{stable_hash(identity, 12)}"
    complete = run / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        return run
    run.mkdir(parents=True, exist_ok=True)
    if force:
        for name in ("tables", "plots"):
            p = run / name
            if p.exists():
                shutil.rmtree(p)
    tables = run / "tables"; plots = run / "plots"; per_alpha = run / "per_alpha"
    for p in (tables, plots, per_alpha):
        p.mkdir(parents=True, exist_ok=True)
    selected.to_csv(run / "manifest_snapshot.csv", index=False, encoding="utf-8-sig")

    effective = json.loads(json.dumps(payload))
    effective["cache_schema"] = CACHE_SCHEMA
    effective["real_alpha_sweep_schema"] = REAL_ALPHA_SWEEP_SCHEMA
    effective["scientific_note"] = (
        "Only method.alpha changes across per-alpha real-domain runs. Every alpha is evaluated by the existing "
        "real validation pipeline, so the full historical real plotting/table contract is retained."
    )
    write_json(run / "effective_config.json", effective)

    run_rows: list[dict[str, Any]] = []
    metrics_frames: list[pd.DataFrame] = []
    domain_frames: list[pd.DataFrame] = []
    dataset_frames: list[pd.DataFrame] = []
    original_methods = list(payload.get("real_validation", {}).get("methods", ["Hodge"]))
    reference_metrics: pd.DataFrame | None = None
    reference_domains: pd.DataFrame | None = None

    # Execute the reference first so invariant baseline rows are available for
    # efficient reuse in every non-reference alpha comparison.
    execution_alphas = [reference_alpha] + [a for a in alphas if not math.isclose(a, reference_alpha, rel_tol=0.0, abs_tol=1e-15)]
    for alpha in execution_alphas:
        print(f"[REAL-ALPHA] alpha={alpha:g}", flush=True)
        local = json.loads(json.dumps(payload))
        local["method"]["alpha"] = float(alpha)
        local["real_validation"]["results_root"] = str(per_alpha.resolve())
        slug = _alpha_slug(alpha)
        # Keep per-alpha output paths compact on Windows.  The sweep parent is
        # already hash-isolated by the full experiment identity, so the long
        # descriptive real-domain run stem is redundant here and can push
        # filenames past the traditional 260-character MAX_PATH boundary.
        # This changes only run-level file layout, never cache/scientific identity.
        local["real_validation"]["run_name"] = slug
        is_reference = math.isclose(alpha, reference_alpha, rel_tol=0.0, abs_tol=1e-15)
        if baseline_policy == "reference_only" and not is_reference:
            local["real_validation"]["methods"] = ["Hodge"]
        alpha_run = run_real_domain_validation(local, config_path, force=force)
        executed_methods = list(local["real_validation"].get("methods", ["Hodge"]))
        run_rows.append({
            "alpha": float(alpha), "alpha_slug": slug, "run_name": slug, "run_dir": str(alpha_run),
            "methods_executed": ",".join(executed_methods),
        })

        mf_raw = pd.read_csv(alpha_run / "tables" / "real_metrics_summary.csv")
        dp = alpha_run / "tables" / "domain_metrics_all_methods.csv"
        df_raw = pd.read_csv(dp) if dp.exists() else pd.DataFrame()
        qp = alpha_run / "tables" / "dataset_metrics_all_methods.csv"
        qf_raw = pd.read_csv(qp) if qp.exists() else pd.DataFrame()

        if is_reference:
            reference_metrics = mf_raw.copy()
            reference_domains = df_raw.copy() if not df_raw.empty else pd.DataFrame()

        mf = mf_raw.copy()
        mf["metrics_source_alpha"] = float(alpha)
        mf["reused_invariant_baseline"] = False
        if baseline_policy == "reference_only" and not is_reference:
            if reference_metrics is None:
                raise RuntimeError("reference alpha metrics were not available before non-reference alpha execution")
            baseline = reference_metrics[reference_metrics["method"] != "Hodge"].copy()
            baseline["metrics_source_alpha"] = float(reference_alpha)
            baseline["reused_invariant_baseline"] = True
            mf = pd.concat([mf, baseline], ignore_index=True)
        mf.insert(0, "alpha", float(alpha)); mf.insert(1, "alpha_slug", slug); mf["run_dir"] = str(alpha_run)
        metrics_frames.append(mf)

        if not df_raw.empty:
            dx = df_raw.copy(); dx["metrics_source_alpha"] = float(alpha); dx["reused_invariant_baseline"] = False
            if baseline_policy == "reference_only" and not is_reference:
                if reference_domains is None:
                    raise RuntimeError("reference alpha domain metrics were not available")
                baseline_d = reference_domains[reference_domains["method"] != "Hodge"].copy()
                baseline_d["metrics_source_alpha"] = float(reference_alpha)
                baseline_d["reused_invariant_baseline"] = True
                dx = pd.concat([dx, baseline_d], ignore_index=True)
            dx.insert(0, "alpha", float(alpha)); dx.insert(1, "alpha_slug", slug); dx["run_dir"] = str(alpha_run)
            domain_frames.append(dx)
        if not qf_raw.empty:
            qx = qf_raw.copy(); qx.insert(0, "alpha", float(alpha)); qx.insert(1, "alpha_slug", slug); qx["run_dir"] = str(alpha_run)
            dataset_frames.append(qx)

    run_index = pd.DataFrame(run_rows)
    metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    domains_df = pd.concat(domain_frames, ignore_index=True) if domain_frames else pd.DataFrame()
    datasets_df = pd.concat(dataset_frames, ignore_index=True) if dataset_frames else pd.DataFrame()
    hodge = metrics[metrics["method"] == "Hodge"].copy() if not metrics.empty else pd.DataFrame()
    hodge_domain = domains_df[domains_df["method"] == "Hodge"].copy() if not domains_df.empty else pd.DataFrame()

    run_index.to_csv(tables / "alpha_run_index.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(tables / "real_metrics_all_methods_all_alpha.csv", index=False, encoding="utf-8-sig")
    hodge.to_csv(tables / "hodge_real_alpha_metrics.csv", index=False, encoding="utf-8-sig")
    if not domains_df.empty:
        domains_df.to_csv(tables / "real_domain_metrics_all_methods_all_alpha.csv", index=False, encoding="utf-8-sig")
        hodge_domain.to_csv(tables / "hodge_real_alpha_domain_metrics.csv", index=False, encoding="utf-8-sig")
    if not datasets_df.empty:
        datasets_df.to_csv(tables / "real_dataset_metrics_all_methods_all_alpha.csv", index=False, encoding="utf-8-sig")

    if baseline_policy == "all_methods_each_alpha":
        baseline_audit = _numeric_baseline_audit(metrics)
        if not baseline_audit.empty and not bool(baseline_audit["passed"].all()):
            bad = baseline_audit[~baseline_audit["passed"]]
            raise RuntimeError(f"Classical baseline metrics changed across alpha, which violates the sweep contract:\n{bad.to_string(index=False)}")
    else:
        baseline_methods = [] if reference_metrics is None else reference_metrics[reference_metrics["method"] != "Hodge"]["method"].astype(str).tolist()
        baseline_audit = pd.DataFrame([{
            "method": m, "metric": "all", "span_across_alpha": 0.0, "tolerance": 0.0, "passed": True,
            "note": "computed once at reference alpha and reused exactly; alpha does not enter baseline semantics",
        } for m in baseline_methods])
    baseline_audit.to_csv(tables / "baseline_invariance_audit.csv", index=False, encoding="utf-8-sig")

    # Recreate the historical all-method core/per-domain comparison for every
    # alpha using the Hodge row from that alpha and invariant baseline rows from
    # the reference run.  This retains the previous figure content without
    # rerendering identical baseline cluster/MST/heatmap figures seven times.
    for alpha in alphas:
        slug = _alpha_slug(alpha)
        m = metrics[np.isclose(metrics["alpha"].to_numpy(float), alpha)].copy()
        if not m.empty:
            plot_core_metrics_comparison(m, plots / slug / "core_metrics_all_methods", plot_options=payload.get("real_validation", {}).get("plotting", {}))
        if not domains_df.empty:
            d = domains_df[np.isclose(domains_df["alpha"].to_numpy(float), alpha)].copy()
            if not d.empty:
                plot_domain_metrics_comparison(d, plots / slug / "core_metrics_by_domain", plot_options=payload.get("real_validation", {}).get("plotting", {}))

    stability = _distance_stability(run_index, reference_alpha)
    stability.to_csv(tables / "hodge_distance_stability_vs_reference_alpha.csv", index=False, encoding="utf-8-sig")
    _plot_metric_trends(hodge, plots)
    _plot_domain_trends(hodge_domain, plots)
    _plot_distance_stability(stability, plots)

    write_json(run / "run_manifest.json", {
        "schema": REAL_ALPHA_SWEEP_SCHEMA,
        "cache_schema": CACHE_SCHEMA,
        "alphas": alphas,
        "reference_alpha": reference_alpha,
        "baseline_policy": baseline_policy,
        "num_networks": int(len(selected)),
        "domains": list(dict.fromkeys(domains)),
        "per_alpha_output_contract": [
            "Hodge distance matrices", "Hodge unsupervised cluster heatmap", "Hodge supervised domain-block heatmap",
            "Hodge global MST", "Hodge within-domain heatmap/MST panels", "Hodge 2D MDS", "Hodge retrieval tables",
            "Hodge cluster assignments", "real metrics summary", "all-method core comparison", "all-method per-domain metrics",
        ],
        "per_alpha_directory_layout": "compact run names (a1, a0p75, ...); scientific settings remain in effective_config/run_manifest",
        "windows_path_fix": "compact per-alpha run directory avoids MAX_PATH failures without changing cache identity or mathematics",
        "baseline_rendering": (
            "reference alpha contains the full historical baseline plot suite; non-reference alpha runs render Hodge only "
            "under baseline_policy=reference_only, while outer all-method comparison figures reuse invariant baseline metrics"
            if baseline_policy == "reference_only" else
            "every alpha reruns/rerenders all configured methods"
        ),
        "mathematics_changed": False,
        "cache_identity_changed": False,
        "experimental_statistics_changed": False,
        "random_graph_seed_protocol_changed": False,
        "baseline_semantics_changed": False,
        "existing_result_comparability": (
            "Every alpha is a distinct final Hodge aggregation setting. Classical baselines are unchanged and are "
            "audited for invariance; Hodge alpha runs must not be pooled as if they were one method setting."
        ),
    })
    write_json(complete, {"completed": True, "schema": REAL_ALPHA_SWEEP_SCHEMA, "run_dir": str(run)})
    print(f"[DONE] {run}", flush=True)
    return run
