"""
scripts/analyze_exp035.py

exp035 — Test 봉인 해제 평가 결과 분석. 본 논문 §7.3 final out-of-sample.

분석 메뉴:
    1) Per source x variant Test Sharpe distribution (mean ± std, IQM, 5% CVaR)
    2) Val vs Test 비교 per variant (generalization gap)
    3) Boxplot per (source, variant)
    4) ATR Test baseline 비교
    5) Cluster preservation on Test
    6) DSR p-value (exp032b 10 seeds + exp034 15 paths = N=25 per variant)

산출:
    reports/exp035_figures/*.png
    reports/exp035_analysis.md
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


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 4: return float(np.mean(x))
    q1, q3 = n // 4, n - n // 4
    return float(np.mean(x[q1:q3]))


def cvar_5(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x); k = max(1, int(np.ceil(n * 0.05)))
    return float(np.mean(x[:k]))


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va = np.var(a, ddof=1) if len(a) > 1 else 0
    vb = np.var(b, ddof=1) if len(b) > 1 else 0
    pooled = np.sqrt((va + vb) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def menu1_per_source(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """source x variant 별 Test 통계."""
    rows = []
    for source in ["exp032b", "exp034"]:
        for v in VARIANTS:
            s = df[(df.source == source) & (df.variant == v)]["test_sharpe"].dropna().values
            if len(s) == 0:
                continue
            t_stat = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if s.std(ddof=1) > 0 else float("inf")
            p_one = 1.0 - stats.t.cdf(t_stat, df=len(s) - 1)
            rows.append({
                "source": source, "variant": v, "n": len(s),
                "mean": s.mean(), "std": s.std(ddof=1),
                "iqm": iqm(s), "cvar_5": cvar_5(s),
                "min": s.min(), "max": s.max(),
                "t_stat": t_stat, "p_t": p_one,
            })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "menu1_per_source.csv", index=False)
    return table


def menu2_val_vs_test(df_test: pd.DataFrame, val_b: pd.DataFrame, val_c: pd.DataFrame, out_dir: Path) -> dict:
    """exp032b/exp034 의 Val Sharpe 평균 vs Test Sharpe 평균 per variant."""
    rows = []
    for v in VARIANTS:
        b_val = val_b[val_b.variant == v]["best_val_sharpe"].mean()
        b_test = df_test[(df_test.source == "exp032b") & (df_test.variant == v)]["test_sharpe"].mean()
        c_val = val_c[val_c.variant == v]["best_sharpe"].mean()
        c_test = df_test[(df_test.source == "exp034") & (df_test.variant == v)]["test_sharpe"].mean()
        rows.append({
            "variant": v,
            "exp032b_val": b_val, "exp032b_test": b_test,
            "exp032b_gap": b_test - b_val,
            "exp034_val": c_val, "exp034_test": c_test,
            "exp034_gap": c_test - c_val,
        })
    cmp = pd.DataFrame(rows)
    cmp.to_csv(out_dir / "menu2_val_vs_test.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(VARIANTS)); width = 0.35
    for k, (ax, src, val_col, test_col) in enumerate([
        (axes[0], "exp032b", "exp032b_val", "exp032b_test"),
        (axes[1], "exp034",  "exp034_val",  "exp034_test"),
    ]):
        ax.bar(x - width/2, [cmp[cmp.variant == v][val_col].values[0] for v in VARIANTS],
               width, label="Val (in-sample)",
               color=[COLOR[v] for v in VARIANTS], alpha=0.5)
        ax.bar(x + width/2, [cmp[cmp.variant == v][test_col].values[0] for v in VARIANTS],
               width, label="Test (out-of-sample)",
               color=[COLOR[v] for v in VARIANTS], alpha=1.0, edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(VARIANTS)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title(f"{src}: Val vs Test")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    fig.suptitle("exp035 Menu 2 - Val (in-sample) vs Test (out-of-sample) per variant")
    fig.tight_layout()
    fig.savefig(out_dir / "menu2_val_vs_test.png", dpi=130)
    plt.close(fig)
    return cmp


def menu3_boxplot(df: pd.DataFrame, atr_test_sharpe: float, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, source in zip(axes, ["exp032b", "exp034"]):
        data = [df[(df.source == source) & (df.variant == v)]["test_sharpe"].values
                for v in VARIANTS]
        bp = ax.boxplot(data, tick_labels=VARIANTS, showmeans=True, patch_artist=True)
        for patch, v in zip(bp["boxes"], VARIANTS):
            patch.set_facecolor(COLOR[v]); patch.set_alpha(0.6)
        if not np.isnan(atr_test_sharpe):
            ax.axhline(atr_test_sharpe, color="black", linestyle="--", alpha=0.6,
                       label=f"ATR Test ({atr_test_sharpe:+.3f})")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
        ax.set_ylabel("Test Sharpe (2024+)")
        n = len(data[0]) if data and len(data[0]) else 0
        ax.set_title(f"{source} (n={n} per variant)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle("exp035 Menu 3 - Test Sharpe distribution per variant per source")
    fig.tight_layout()
    fig.savefig(out_dir / "menu3_boxplot.png", dpi=130)
    plt.close(fig)


def menu4_cluster_test(df: pd.DataFrame, out_dir: Path) -> dict:
    """Test Sharpe 에서 cluster preservation."""
    result = {}
    for source in ["exp032b", "exp034"]:
        by_v = {v: df[(df.source == source) & (df.variant == v)]["test_sharpe"].dropna().values
                for v in VARIANTS}
        if any(len(by_v[v]) == 0 for v in VARIANTS):
            continue
        pairs_w = [("sym", "dsr"), ("asym", "pt")]
        pairs_a = [(a, b) for a in ["sym", "dsr"] for b in ["asym", "pt"]]
        within = [abs(cohens_d(by_v[a], by_v[b])) for a, b in pairs_w]
        across = [abs(cohens_d(by_v[a], by_v[b])) for a, b in pairs_a]
        result[source] = {
            "within": float(np.mean(within)),
            "across": float(np.mean(across)),
            "ratio": float(np.mean(across) / np.mean(within)) if np.mean(within) > 0 else float("inf"),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default="experiments/exp035_summary.csv")
    p.add_argument("--exp032b", default="experiments/exp032b_summary.csv")
    p.add_argument("--exp034",  default="experiments/exp034_summary.csv")
    p.add_argument("--out",     default="reports/exp035_figures")
    p.add_argument("--md",      default="reports/exp035_analysis.md")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    val_b = pd.read_csv(args.exp032b)
    val_c = pd.read_csv(args.exp034)
    print(f"Loaded exp035: {len(df)} rows")
    print(f"  sources: {sorted(df.source.unique())}")
    print(f"  variants: {sorted(df.variant.unique())}\n")

    # ATR Test baseline (해당 행 추출)
    atr_row = df[df.source == "atr_baseline"]
    atr_test = float(atr_row["test_sharpe"].values[0]) if len(atr_row) else float("nan")
    if not np.isnan(atr_test):
        print(f"ATR Baseline Test: Sharpe {atr_test:+.3f}, "
              f"Return {atr_row['test_return_pct'].values[0]:+.2f}%, "
              f"MDD {atr_row['test_mdd_pct'].values[0]:.2f}%, "
              f"Trades {int(atr_row['test_n_trades'].values[0])}")

    print("\n=== Menu 1: per source x variant ===")
    table = menu1_per_source(df, out_dir)
    print(table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== Menu 2: Val vs Test ===")
    cmp = menu2_val_vs_test(df, val_b, val_c, out_dir)
    print(cmp.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== Menu 3: Boxplot ===")
    menu3_boxplot(df, atr_test, out_dir)
    print("  saved menu3_boxplot.png")

    print("\n=== Menu 4: Cluster preservation on Test ===")
    cl = menu4_cluster_test(df, out_dir)
    for source, r in cl.items():
        print(f"  {source}: within {r['within']:.3f} | across {r['across']:.3f} | ratio {r['ratio']:.2f}x")

    md = [
        f"# exp035 Test (out-of-sample) Analysis\n",
        f"Source: `{args.csv}`\n",
        f"\n## ATR Baseline Test: Sharpe {atr_test:+.3f}, Return {atr_row['test_return_pct'].values[0]:+.2f}%, "
        f"MDD {atr_row['test_mdd_pct'].values[0]:.2f}% (Trades {int(atr_row['test_n_trades'].values[0])}, "
        f"Cycles {int(atr_row['test_n_cycles'].values[0])})\n",
        f"\n## Per source x variant\n",
        table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
        f"\n\n## Val vs Test\n",
        cmp.to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
        f"\n\n## Cluster preservation on Test\n",
    ]
    for source, r in cl.items():
        md.append(f"- {source}: within {r['within']:.3f} | across {r['across']:.3f} | ratio {r['ratio']:.2f}x")
    md.append(f"\n## Figures\n")
    md.append(f"- ![Menu 2](exp035_figures/menu2_val_vs_test.png)")
    md.append(f"- ![Menu 3](exp035_figures/menu3_boxplot.png)")
    Path(args.md).write_text("\n".join(md), encoding="utf-8")
    print(f"\nMarkdown -> {args.md}")


if __name__ == "__main__":
    main()
