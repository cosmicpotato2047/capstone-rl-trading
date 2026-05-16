"""
scripts/analyze_exp033.py

exp033 (Slippage 0.02%) 결과 분석. exp032b (no slippage) 와 비교하여
cluster 구분이 robust 하게 유지되는지 검증.

분석 메뉴:
    1) Per-variant table: exp033 metrics (mean +/- std)
    2) Side-by-side comparison: exp032b vs exp033 mean Sharpe / MDD / Calmar
    3) Pareto scatter (40 dots) - exp033 only
    4) Policy distance (within / across cluster) - exp033 vs exp032b 비교
    5) Slippage 감쇠율 per variant (얼마나 Sharpe 가 떨어졌나)

산출:
    reports/exp033_figures/*.png
    reports/exp033_analysis.md
    experiments/exp033_comparison.csv
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VARIANTS = ["sym", "asym", "dsr", "pt"]
CLUSTERS = {"sym": "aggressive", "dsr": "aggressive",
            "asym": "conservative", "pt": "conservative"}
COLOR = {"sym": "#1f77b4", "dsr": "#ff7f0e", "asym": "#2ca02c", "pt": "#d62728"}
ATR_SHARPE = 1.505
ATR_MDD = 9.83


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va = np.var(a, ddof=1) if len(a) > 1 else 0
    vb = np.var(b, ddof=1) if len(b) > 1 else 0
    pooled = np.sqrt((va + vb) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def per_variant(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        s = df[df.variant == v][metric].dropna()
        rows.append({
            "variant": v, "n": len(s),
            "mean": float(s.mean()), "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()), "max": float(s.max()),
        })
    return pd.DataFrame(rows)


def menu1_side_by_side(b: pd.DataFrame, c: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """exp032b vs exp033 side-by-side. Cohen's d between conditions per variant."""
    rows = []
    for v in VARIANTS:
        sb = b[b.variant == v]["best_val_sharpe"].values
        sc = c[c.variant == v]["best_val_sharpe"].values
        mb = b[b.variant == v]["best_mdd_pct"].values
        mc = c[c.variant == v]["best_mdd_pct"].values
        rows.append({
            "variant": v,
            "sharpe_exp032b": sb.mean(), "sharpe_exp033": sc.mean(),
            "delta_sharpe": sc.mean() - sb.mean(),
            "cohens_d_sharpe": cohens_d(sc, sb),
            "mdd_exp032b": mb.mean(), "mdd_exp033": mc.mean(),
            "delta_mdd": mc.mean() - mb.mean(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "menu1_side_by_side.csv", index=False)

    # Figure: side-by-side bar
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(VARIANTS))
    width = 0.35
    # left: Sharpe
    ax = axes[0]
    ax.bar(x - width/2, [df[df.variant == v]["sharpe_exp032b"].values[0] for v in VARIANTS],
           width, label="exp032b (no slippage)", color=[COLOR[v] for v in VARIANTS], alpha=0.5)
    ax.bar(x + width/2, [df[df.variant == v]["sharpe_exp033"].values[0] for v in VARIANTS],
           width, label="exp033 (slippage 0.02%)", color=[COLOR[v] for v in VARIANTS], alpha=1.0,
           edgecolor="black")
    ax.axhline(ATR_SHARPE, color="black", linestyle="--", alpha=0.5, label=f"ATR baseline ({ATR_SHARPE})")
    ax.set_xticks(x); ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("Mean best Val Sharpe (10 seeds)")
    ax.set_title("Sharpe: exp032b vs exp033 (slippage 0.02%)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    # right: MDD
    ax = axes[1]
    ax.bar(x - width/2, [df[df.variant == v]["mdd_exp032b"].values[0] for v in VARIANTS],
           width, label="exp032b", color=[COLOR[v] for v in VARIANTS], alpha=0.5)
    ax.bar(x + width/2, [df[df.variant == v]["mdd_exp033"].values[0] for v in VARIANTS],
           width, label="exp033", color=[COLOR[v] for v in VARIANTS], alpha=1.0,
           edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("Mean best MDD (%)")
    ax.set_title("MDD: exp032b vs exp033")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("exp033 Menu 1 - exp032b vs exp033 side-by-side")
    fig.tight_layout()
    fig.savefig(out_dir / "menu1_side_by_side.png", dpi=130)
    plt.close(fig)
    return df


def menu2_pareto_scatter(c: pd.DataFrame, out_dir: Path) -> None:
    """exp033 의 40 dots Pareto scatter. exp032b 와 동일 format."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for v in VARIANTS:
        sub = c[c.variant == v]
        ax.scatter(sub["best_mdd_pct"], sub["best_val_sharpe"],
                   s=80, c=COLOR[v], label=f"{v} ({CLUSTERS[v]})",
                   alpha=0.7, edgecolors="black", linewidths=0.5)
        ax.plot(sub["best_mdd_pct"].mean(), sub["best_val_sharpe"].mean(),
                marker="x", c=COLOR[v], markersize=15, mew=3)

    atr_mdd_shown = min(ATR_MDD, 8.0)
    ax.scatter([atr_mdd_shown], [ATR_SHARPE], s=200, c="black",
               marker="*", label=f"ATR baseline (MDD {ATR_MDD:.1f})", zorder=10)

    ax.set_xlabel("Max Drawdown (%)  [lower is better]")
    ax.set_ylabel("Val Sharpe Ratio  [higher is better]")
    ax.set_title("exp033 Menu 2 - Pareto scatter under slippage 0.02%\n"
                 "Does cluster structure survive?")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "menu2_pareto_scatter.png", dpi=130)
    plt.close(fig)


def menu3_slippage_resilience(b: pd.DataFrame, c: pd.DataFrame, out_dir: Path) -> dict:
    """Slippage 감쇠율 per variant: (exp033 Sharpe / exp032b Sharpe)."""
    res = {}
    for v in VARIANTS:
        sb = b[b.variant == v]["best_val_sharpe"].mean()
        sc = c[c.variant == v]["best_val_sharpe"].mean()
        res[v] = float(sc / sb) if sb > 0 else 0.0
    return res


def menu4_pairwise_cluster(c: pd.DataFrame, out_dir: Path) -> dict:
    """exp033 의 pairwise Cohen's d -> within-cluster 평균 vs across-cluster 평균."""
    by_v = {v: c[c.variant == v]["best_val_sharpe"].dropna().values for v in VARIANTS}
    pairs_within = [("sym", "dsr"), ("asym", "pt")]
    pairs_across = [(a, b) for a in ["sym", "dsr"] for b in ["asym", "pt"]]
    within_d = [abs(cohens_d(by_v[a], by_v[b])) for a, b in pairs_within]
    across_d = [abs(cohens_d(by_v[a], by_v[b])) for a, b in pairs_across]
    return {
        "within_cluster_mean_d": float(np.mean(within_d)),
        "across_cluster_mean_d": float(np.mean(across_d)),
        "ratio": float(np.mean(across_d) / np.mean(within_d)) if np.mean(within_d) > 0 else float("inf"),
        "pairs_within": dict(zip([f"{a}_vs_{b}" for a, b in pairs_within], within_d)),
        "pairs_across": dict(zip([f"{a}_vs_{b}" for a, b in pairs_across], across_d)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp033", default="experiments/exp033_summary.csv")
    p.add_argument("--exp032b", default="experiments/exp032b_summary.csv")
    p.add_argument("--out", default="reports/exp033_figures")
    p.add_argument("--md", default="reports/exp033_analysis.md")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    b = pd.read_csv(args.exp032b)
    c = pd.read_csv(args.exp033)
    print(f"exp032b: {len(b)} runs | exp033: {len(c)} runs\n")

    pv33 = per_variant(c, "best_val_sharpe")
    print("=== exp033 per variant (best Sharpe) ===")
    print(pv33.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== Menu 1: side-by-side ===")
    cmp = menu1_side_by_side(b, c, out_dir)
    print(cmp.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n=== Menu 2: Pareto scatter ===")
    menu2_pareto_scatter(c, out_dir)

    print("\n=== Menu 3: Slippage resilience ===")
    res = menu3_slippage_resilience(b, c, out_dir)
    for v, r in res.items():
        print(f"  {v}: Sharpe retention {r*100:.1f}% (1.0 = no degradation)")

    print("\n=== Menu 4: Cluster preservation ===")
    cl = menu4_pairwise_cluster(c, out_dir)
    print(f"  within-cluster mean |d|: {cl['within_cluster_mean_d']:.3f}")
    print(f"  across-cluster mean |d|: {cl['across_cluster_mean_d']:.3f}")
    print(f"  ratio: {cl['ratio']:.2f}x  (exp032b reference: 2.22x at policy level)")

    md = [
        f"# exp033 Slippage Robustness Analysis\n",
        f"## Per-variant exp033 (best Sharpe, 10 seeds)\n",
        pv33.to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
        f"\n\n## exp032b vs exp033 side-by-side\n",
        cmp.to_string(index=False, float_format=lambda x: f"{x:+.3f}"),
        f"\n\n## Slippage resilience (Sharpe retention)\n",
    ]
    for v, r in res.items():
        md.append(f"- {v}: {r*100:.1f}%")
    md.append(f"\n## Cluster preservation\n")
    md.append(f"- within-cluster |d| mean: {cl['within_cluster_mean_d']:.3f}")
    md.append(f"- across-cluster |d| mean: {cl['across_cluster_mean_d']:.3f}")
    md.append(f"- ratio: {cl['ratio']:.2f}x")
    md.append(f"\n## Figures\n")
    md.append(f"- ![Menu 1](exp033_figures/menu1_side_by_side.png)")
    md.append(f"- ![Menu 2](exp033_figures/menu2_pareto_scatter.png)")
    Path(args.md).write_text("\n".join(md), encoding="utf-8")
    print(f"\nMarkdown saved to {args.md}")


if __name__ == "__main__":
    main()
