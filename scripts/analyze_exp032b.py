"""
scripts/analyze_exp032b.py

exp032b summary CSV -> 통계 분석 (variant 간 비교).

입력:
    experiments/exp032b_summary.csv  (run_exp032b.py 가 생성)

산출:
    1) per-variant summary table (mean +/- std, IQM, n)
    2) pairwise table (Cohen's d, P(A>B) via bootstrap)
    3) figures/exp032b_*.png
       - boxplot_sharpe.png
       - bar_mean_sharpe.png
       - heatmap_cohens_d.png
       - heatmap_prob_a_gt_b.png
    4) markdown 표 (RESEARCH_LOG 에 그대로 붙일 수 있는 형식)

사용법:
    python scripts/analyze_exp032b.py
    python scripts/analyze_exp032b.py --metric best_val_sharpe  # default
    python scripts/analyze_exp032b.py --metric final_val_sharpe
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VARIANTS = ["sym", "asym", "dsr", "pt"]
N_BOOT_DEFAULT = 10000
RNG_SEED = 42


# ---------- statistics ----------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (pooled std). > 0 means a > b."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    var_a = np.var(a, ddof=1) if len(a) > 1 else 0.0
    var_b = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt((var_a + var_b) / 2)
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def prob_a_gt_b(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT_DEFAULT) -> float:
    """Bootstrap P(mean(A) > mean(B)). Resamples with replacement."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    a_b = rng.choice(a, size=(n_boot, len(a)), replace=True)
    b_b = rng.choice(b, size=(n_boot, len(b)), replace=True)
    return float(np.mean(a_b.mean(axis=1) > b_b.mean(axis=1)))


def iqm(x: np.ndarray) -> float:
    """Interquartile mean (drops top 25% and bottom 25%, averages middle)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 4:
        return float(np.mean(x))
    q1 = n // 4
    q3 = n - n // 4
    return float(np.mean(x[q1:q3]))


# ---------- analysis ----------

def per_variant_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        sub = df[df["variant"] == v][metric].dropna().values
        if len(sub) == 0:
            rows.append({"variant": v, "n": 0, "mean": float("nan"),
                         "std": float("nan"), "iqm": float("nan"),
                         "min": float("nan"), "max": float("nan")})
            continue
        rows.append({
            "variant": v,
            "n": len(sub),
            "mean": float(np.mean(sub)),
            "std": float(np.std(sub, ddof=1)) if len(sub) > 1 else 0.0,
            "iqm": iqm(sub),
            "min": float(np.min(sub)),
            "max": float(np.max(sub)),
        })
    return pd.DataFrame(rows)


def pairwise_table(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """returns (cohens_d_matrix, prob_a_gt_b_matrix). Row=A, Col=B."""
    by_v = {v: df[df["variant"] == v][metric].dropna().values for v in VARIANTS}
    d_mat = pd.DataFrame(index=VARIANTS, columns=VARIANTS, dtype=float)
    p_mat = pd.DataFrame(index=VARIANTS, columns=VARIANTS, dtype=float)
    for a in VARIANTS:
        for b in VARIANTS:
            if a == b or len(by_v[a]) == 0 or len(by_v[b]) == 0:
                d_mat.loc[a, b] = float("nan")
                p_mat.loc[a, b] = float("nan")
            else:
                d_mat.loc[a, b] = cohens_d(by_v[a], by_v[b])
                p_mat.loc[a, b] = prob_a_gt_b(by_v[a], by_v[b])
    return d_mat, p_mat


# ---------- figures ----------

def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("WARN: matplotlib 미설치. figures 건너뜀.")
        return None


def make_figures(df: pd.DataFrame, metric: str, out_dir: Path,
                 d_mat: pd.DataFrame, p_mat: pd.DataFrame) -> None:
    plt = _ensure_matplotlib()
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    by_v = [df[df["variant"] == v][metric].dropna().values for v in VARIANTS]

    # 1) boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(by_v, tick_labels=VARIANTS, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_title(f"exp032b: {metric} per variant (n={len(df)//len(VARIANTS) if len(df) else '?'} seeds)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_sharpe.png", dpi=120)
    plt.close(fig)

    # 2) bar mean +/- std
    means = [np.mean(x) if len(x) else 0 for x in by_v]
    stds  = [np.std(x, ddof=1) if len(x) > 1 else 0 for x in by_v]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(VARIANTS, means, yerr=stds, capsize=8, alpha=0.7)
    ax.set_ylabel(f"mean {metric}")
    ax.set_title(f"exp032b: mean {metric} +/- std")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_mean_sharpe.png", dpi=120)
    plt.close(fig)

    # 3) heatmap Cohen's d
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(d_mat.values.astype(float), cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels(VARIANTS)
    ax.set_yticks(range(len(VARIANTS))); ax.set_yticklabels(VARIANTS)
    for i, a in enumerate(VARIANTS):
        for j, b in enumerate(VARIANTS):
            v = d_mat.loc[a, b]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 1.0 else "black", fontsize=10)
    ax.set_title("Cohen's d (row vs col)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_cohens_d.png", dpi=120)
    plt.close(fig)

    # 4) heatmap P(A>B)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(p_mat.values.astype(float), cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels(VARIANTS)
    ax.set_yticks(range(len(VARIANTS))); ax.set_yticklabels(VARIANTS)
    for i, a in enumerate(VARIANTS):
        for j, b in enumerate(VARIANTS):
            v = p_mat.loc[a, b]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if (v < 0.2 or v > 0.8) else "black", fontsize=10)
    ax.set_title("P(row > col) via bootstrap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_prob_a_gt_b.png", dpi=120)
    plt.close(fig)

    print(f"  figures saved to {out_dir}")


# ---------- markdown printers ----------

def md_per_variant(table: pd.DataFrame, metric: str) -> str:
    lines = [f"### Per-variant ({metric})\n",
             "| Variant | n | mean | std | IQM | min | max |",
             "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        if r["n"] == 0:
            lines.append(f"| {r['variant']} | 0 | - | - | - | - | - |")
        else:
            lines.append(
                f"| {r['variant']} | {int(r['n'])} | {r['mean']:+.3f} | "
                f"{r['std']:.3f} | {r['iqm']:+.3f} | "
                f"{r['min']:+.3f} | {r['max']:+.3f} |"
            )
    return "\n".join(lines)


def md_pairwise(d_mat: pd.DataFrame, p_mat: pd.DataFrame) -> str:
    lines = ["### Pairwise (row = A, col = B; A vs B)\n",
             "**Cohen's d** (effect size, > 0 means A > B):\n",
             "| | " + " | ".join(VARIANTS) + " |",
             "|---|" + "---|" * len(VARIANTS)]
    for a in VARIANTS:
        row = [a]
        for b in VARIANTS:
            v = d_mat.loc[a, b]
            row.append("-" if pd.isna(v) else f"{v:+.2f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["\n**P(A > B)** via bootstrap:\n",
              "| | " + " | ".join(VARIANTS) + " |",
              "|---|" + "---|" * len(VARIANTS)]
    for a in VARIANTS:
        row = [a]
        for b in VARIANTS:
            v = p_mat.loc[a, b]
            row.append("-" if pd.isna(v) else f"{v:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    default="experiments/exp032b_summary.csv")
    p.add_argument("--metric", default="best_val_sharpe",
                   choices=["best_val_sharpe", "final_val_sharpe",
                            "best_return_pct", "final_return_pct"])
    p.add_argument("--figdir", default="reports/exp032b_figures")
    p.add_argument("--out-md", default="reports/exp032b_analysis.md")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. run_exp032b.py 를 먼저 실행하세요.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} runs from {csv_path}")
    print(f"Variants present: {sorted(df['variant'].unique())}")
    print(f"Metric: {args.metric}\n")

    pv = per_variant_table(df, args.metric)
    print(pv.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print()
    d_mat, p_mat = pairwise_table(df, args.metric)
    print("Cohen's d matrix:"); print(d_mat.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\nP(A>B) matrix:");   print(p_mat.to_string(float_format=lambda x: f"{x:.3f}"))

    figdir = Path(args.figdir)
    make_figures(df, args.metric, figdir, d_mat, p_mat)

    md_text = (f"# exp032b Analysis ({args.metric})\n\n"
               f"Source: `{csv_path}`  (n_runs={len(df)})\n\n"
               + md_per_variant(pv, args.metric) + "\n\n"
               + md_pairwise(d_mat, p_mat) + "\n\n"
               + f"### Figures\n\n"
                 f"- ![boxplot]({figdir.name}/boxplot_sharpe.png)\n"
                 f"- ![bar mean]({figdir.name}/bar_mean_sharpe.png)\n"
                 f"- ![Cohen d]({figdir.name}/heatmap_cohens_d.png)\n"
                 f"- ![P(A>B)]({figdir.name}/heatmap_prob_a_gt_b.png)\n")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_text, encoding="utf-8")
    print(f"\nMarkdown summary saved to {out_md}")


if __name__ == "__main__":
    main()
