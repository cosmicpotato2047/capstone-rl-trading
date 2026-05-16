"""
scripts/analyze_exp034.py

exp034 (CPCV 6-fold) 분석.

분석 메뉴:
    1) Per-variant Sharpe distribution (15 paths): mean ± std, IQM, 5% CVaR
    2) DSR (Deflated Sharpe Ratio) approx + Bonferroni-corrected t-test
    3) Path-by-path heatmap (4 variants × 15 paths Sharpe)
    4) Boxplot per variant
    5) Cluster preservation: exp032b 의 cluster 가 다중 path 에서도 유지되는지

산출:
    reports/exp034_figures/*.png
    reports/exp034_analysis.md
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VARIANTS = ["sym", "asym", "dsr", "pt"]
CLUSTERS = {"sym": "aggressive", "dsr": "aggressive",
            "asym": "conservative", "pt": "conservative"}
COLOR = {"sym": "#1f77b4", "dsr": "#ff7f0e", "asym": "#2ca02c", "pt": "#d62728"}
ATR_SHARPE = 1.505  # Env-v4 ATR baseline (Val 2021-2023, no slippage)


# ---------- statistics ----------

def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 4:
        return float(np.mean(x))
    q1, q3 = n // 4, n - n // 4
    return float(np.mean(x[q1:q3]))


def cvar_5(x):
    """5% Conditional Value-at-Risk (lower tail)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    k = max(1, int(np.ceil(n * 0.05)))
    return float(np.mean(x[:k]))


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va = np.var(a, ddof=1) if len(a) > 1 else 0
    vb = np.var(b, ddof=1) if len(b) > 1 else 0
    pooled = np.sqrt((va + vb) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def deflated_sharpe(sharpes: np.ndarray, n_trials: int = 4) -> dict:
    """Lopez de Prado (2014) 의 단순화 버전.

    H0: 실제 SR = 0. one-sided test on observed mean(SR).
    DSR = Phi((SR_mean - SR*) / std_err)  where SR* = E[max SR_i | true SR_i = 0]
    SR* approx = sqrt(2 ln(N)) * (std_dev / sqrt(N))  for null hypothesis correction.

    Returns: {sr_mean, sr_std, sr_skew, sr_kurt, t_stat, dsr_z, p_value}
    """
    s = np.asarray(sharpes, dtype=float)
    n = len(s)
    sr_mean = float(np.mean(s))
    sr_std  = float(np.std(s, ddof=1)) if n > 1 else 0.0
    sr_skew = float(stats.skew(s)) if n > 2 else 0.0
    sr_kurt = float(stats.kurtosis(s, fisher=True)) if n > 3 else 0.0  # excess kurt

    # Standard t-statistic
    t_stat = sr_mean / (sr_std / np.sqrt(n)) if sr_std > 0 else float("inf")
    p_t = 1.0 - stats.t.cdf(t_stat, df=n - 1)  # one-sided

    # Lopez de Prado DSR correction: account for multiple testing across n_trials variants
    # SR* (expected max under null) ≈ sqrt(2 ln(n_trials)) * (sr_std / sqrt(n))
    sr_star = np.sqrt(2 * np.log(max(n_trials, 2))) * (sr_std / np.sqrt(n))
    # adjusted z accounting for skew/kurt
    adj_denom = np.sqrt(max(1e-9, 1.0 - sr_skew * sr_mean + (sr_kurt) / 4.0 * sr_mean ** 2))
    dsr_z = (sr_mean - sr_star) * np.sqrt(n - 1) / (sr_std * adj_denom) if sr_std > 0 else float("inf")
    dsr_p = 1.0 - stats.norm.cdf(dsr_z)

    return {
        "n": n, "sr_mean": sr_mean, "sr_std": sr_std,
        "sr_skew": sr_skew, "sr_kurt": sr_kurt,
        "t_stat": float(t_stat), "p_t": float(p_t),
        "sr_star": float(sr_star),
        "dsr_z": float(dsr_z), "dsr_p": float(dsr_p),
    }


# ---------- analysis ----------

def menu1_per_variant(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        s = df[df.variant == v]["best_sharpe"].dropna().values
        if len(s) == 0:
            continue
        d = deflated_sharpe(s, n_trials=len(VARIANTS))
        rows.append({
            "variant": v,
            "n_paths": len(s),
            "sr_mean": d["sr_mean"], "sr_std": d["sr_std"],
            "iqm": iqm(s), "cvar_5": cvar_5(s),
            "sr_min": float(np.min(s)), "sr_max": float(np.max(s)),
            "t_stat": d["t_stat"], "p_t": d["p_t"],
            "sr_star": d["sr_star"], "dsr_z": d["dsr_z"], "dsr_p": d["dsr_p"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "menu1_per_variant.csv", index=False)
    return table


def menu2_path_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot(index="variant", columns="path_id", values="best_sharpe").reindex(VARIANTS)
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=3, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"p{i:02d}" for i in pivot.columns])
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels(VARIANTS)
    for i, v in enumerate(VARIANTS):
        for j, pid in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                color = "white" if abs(val) > 1.5 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title("exp034 Menu 2 - CPCV Sharpe per (variant, path)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Sharpe")
    fig.tight_layout()
    fig.savefig(out_dir / "menu2_heatmap.png", dpi=130)
    plt.close(fig)


def menu3_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    data = [df[df.variant == v]["best_sharpe"].values for v in VARIANTS]
    bp = ax.boxplot(data, tick_labels=VARIANTS, showmeans=True, patch_artist=True)
    for patch, v in zip(bp["boxes"], VARIANTS):
        patch.set_facecolor(COLOR[v])
        patch.set_alpha(0.6)
    ax.axhline(ATR_SHARPE, color="black", linestyle="--", alpha=0.5,
               label=f"ATR baseline ({ATR_SHARPE})")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3, label="zero")
    ax.set_ylabel("CPCV test Sharpe (15 paths)")
    ax.set_title("exp034 Menu 3 - Sharpe distribution per variant (15 CPCV paths)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "menu3_boxplot.png", dpi=130)
    plt.close(fig)


def menu4_dsr_table(table: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    rows = []
    for _, r in table.iterrows():
        rows.append([
            r["variant"],
            f"{r['n_paths']}",
            f"{r['sr_mean']:+.3f}",
            f"{r['sr_std']:.3f}",
            f"{r['iqm']:+.3f}",
            f"{r['cvar_5']:+.3f}",
            f"{r['t_stat']:+.2f}",
            f"{r['p_t']:.4f}",
            f"{r['sr_star']:+.3f}",
            f"{r['dsr_z']:+.2f}",
            f"{r['dsr_p']:.4f}",
        ])
    cols = ["variant", "n", "SR_mean", "SR_std", "IQM", "5% CVaR",
            "t-stat", "p (t)", "SR*", "DSR z", "DSR p"]
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    ax.set_title("exp034 Menu 4 - DSR (Deflated Sharpe Ratio) per variant",
                 pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "menu4_dsr_table.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def menu5_cluster_preservation(df: pd.DataFrame, out_dir: Path) -> dict:
    by_v = {v: df[df.variant == v]["best_sharpe"].dropna().values for v in VARIANTS}
    pairs_within = [("sym", "dsr"), ("asym", "pt")]
    pairs_across = [(a, b) for a in ["sym", "dsr"] for b in ["asym", "pt"]]
    within_d = [abs(cohens_d(by_v[a], by_v[b])) for a, b in pairs_within]
    across_d = [abs(cohens_d(by_v[a], by_v[b])) for a, b in pairs_across]
    return {
        "within_cluster_mean_d": float(np.mean(within_d)),
        "across_cluster_mean_d": float(np.mean(across_d)),
        "ratio": float(np.mean(across_d) / np.mean(within_d)) if np.mean(within_d) > 0 else float("inf"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="experiments/exp034_summary.csv")
    p.add_argument("--out", default="reports/exp034_figures")
    p.add_argument("--md", default="reports/exp034_analysis.md")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(df)} runs from {args.csv}\n")

    print("=== Menu 1: Per-variant DSR table ===")
    table = menu1_per_variant(df, out_dir)
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== Menu 2: Path heatmap ===")
    menu2_path_heatmap(df, out_dir)
    print(f"  saved menu2_heatmap.png")

    print("\n=== Menu 3: Boxplot ===")
    menu3_boxplot(df, out_dir)
    print(f"  saved menu3_boxplot.png")

    print("\n=== Menu 4: DSR table figure ===")
    menu4_dsr_table(table, out_dir)
    print(f"  saved menu4_dsr_table.png")

    print("\n=== Menu 5: Cluster preservation ===")
    cl = menu5_cluster_preservation(df, out_dir)
    print(f"  within-cluster |d|: {cl['within_cluster_mean_d']:.3f}")
    print(f"  across-cluster |d|: {cl['across_cluster_mean_d']:.3f}")
    print(f"  ratio: {cl['ratio']:.2f}x  (exp032b 2.22x, exp033 2.19x reference)")

    # markdown
    md = [
        f"# exp034 CPCV Analysis\n",
        f"Source: `{args.csv}` ({len(df)} runs)\n",
        f"\n## Per-variant DSR\n",
        table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
        f"\n\n## Cluster preservation\n",
        f"- within-cluster |d| mean: {cl['within_cluster_mean_d']:.3f}",
        f"- across-cluster |d| mean: {cl['across_cluster_mean_d']:.3f}",
        f"- ratio: {cl['ratio']:.2f}x  (reference: exp032b 2.22x, exp033 2.19x)",
        f"\n## Figures\n",
        f"- ![heatmap](exp034_figures/menu2_heatmap.png)",
        f"- ![boxplot](exp034_figures/menu3_boxplot.png)",
        f"- ![DSR table](exp034_figures/menu4_dsr_table.png)",
    ]
    Path(args.md).write_text("\n".join(md), encoding="utf-8")
    print(f"\nMarkdown -> {args.md}")


if __name__ == "__main__":
    main()
