"""
scripts/analyze_phase15.py

Phase 15 - 본 논문 final 통계 분석 + 종합 figures.

세 가지 분석:
    A) Bootstrap P(RL_distribution > ATR_scalar) per variant per environment
    B) Val vs Test distribution shift quantification
       (atr_ratio, returns, trend 의 KS test + Wasserstein distance)
    C) Three-environment 종합 figures (publication quality)

입력:
    experiments/exp032b_summary.csv  (Val 10 seeds)
    experiments/exp033_summary.csv   (Slippage 10 seeds)
    experiments/exp034_summary.csv   (CPCV 15 paths)
    experiments/exp035_summary.csv   (Test 100 RL + ATR)
    data/processed/btc_val.parquet, btc_test.parquet  (분포 shift)

산출:
    reports/phase15_figures/*.png
    reports/phase15_analysis.md
    experiments/phase15_significance.csv (raw)
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

# ATR baseline reference values
ATR_VAL_SHARPE = 1.378    # Env-v4 ATR Bayesian Trial #34, Val 2021-2023 (re-measured post-Phase-16a)
ATR_TEST_SHARPE = -0.055  # exp035 measured (Test 2024+)
# exp033/exp034 의 ATR baseline 은 미측정 — Val 1.378 reference 사용 (caveat)


# ---------- A: Bootstrap significance ----------

def bootstrap_p_greater(rl_samples: np.ndarray, atr_scalar: float,
                        n_boot: int = 10000, seed: int = 42) -> dict:
    """P(mean(bootstrap_resample) > atr_scalar)."""
    rng = np.random.default_rng(seed)
    n = len(rl_samples)
    boot_means = rng.choice(rl_samples, size=(n_boot, n), replace=True).mean(axis=1)
    p_gt = float(np.mean(boot_means > atr_scalar))
    ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    return {
        "n": n, "mean": float(rl_samples.mean()),
        "atr": atr_scalar,
        "p_gt_atr": p_gt,
        "ci95_low": ci_low, "ci95_high": ci_high,
    }


def menu_a(out_dir: Path) -> pd.DataFrame:
    """Bootstrap significance: RL distribution vs ATR scalar per (variant, env)."""
    val_b = pd.read_csv("experiments/exp032b_summary.csv")
    slp   = pd.read_csv("experiments/exp033_summary.csv")
    cpcv  = pd.read_csv("experiments/exp034_summary.csv")
    test  = pd.read_csv("experiments/exp035_summary.csv")

    rows = []
    for v in VARIANTS:
        # Val (exp032b)
        s = val_b[val_b.variant == v]["best_val_sharpe"].values
        r = bootstrap_p_greater(s, ATR_VAL_SHARPE)
        r.update({"variant": v, "env": "Val (single-split)", "atr_ref": "Val 1.378"})
        rows.append(r)
        # Slippage (exp033)
        s = slp[slp.variant == v]["best_val_sharpe"].values
        r = bootstrap_p_greater(s, ATR_VAL_SHARPE)
        r.update({"variant": v, "env": "Slippage 0.02%",
                  "atr_ref": "Val 1.505 (caveat: ATR not retrained with slippage)"})
        rows.append(r)
        # CPCV (exp034) - test partition sharpe
        s = cpcv[cpcv.variant == v]["best_sharpe"].values
        r = bootstrap_p_greater(s, ATR_VAL_SHARPE)
        r.update({"variant": v, "env": "CPCV (15 paths test)",
                  "atr_ref": "Val 1.505 (caveat: ATR not per-path)"})
        rows.append(r)
        # Test (exp035 exp032b source)
        s = test[(test.source == "exp032b") & (test.variant == v)]["test_sharpe"].values
        r = bootstrap_p_greater(s, ATR_TEST_SHARPE)
        r.update({"variant": v, "env": "Test (OOS, exp032b)", "atr_ref": "Test -0.055"})
        rows.append(r)
        # Test (exp035 exp034 source)
        s = test[(test.source == "exp034") & (test.variant == v)]["test_sharpe"].values
        r = bootstrap_p_greater(s, ATR_TEST_SHARPE)
        r.update({"variant": v, "env": "Test (OOS, exp034)", "atr_ref": "Test -0.055"})
        rows.append(r)

    df = pd.DataFrame(rows)[[
        "variant", "env", "n", "mean", "atr", "p_gt_atr", "ci95_low", "ci95_high", "atr_ref"
    ]]
    df.to_csv(out_dir / "menu_a_significance.csv", index=False)
    print("=== Menu A: P(RL > ATR) per (variant, environment) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    return df


# ---------- B: Distribution shift ----------

def menu_b(out_dir: Path) -> dict:
    """Val 2021-2023 vs Test 2024+ 분포 shift 정량."""
    val  = pd.read_parquet("data/processed/btc_val.parquet")
    test = pd.read_parquet("data/processed/btc_test.parquet")

    # 비교 metric: atr_ratio (volatility), log return, trend_short
    val["log_return"]  = np.log(val["close"]).diff().fillna(0)
    test["log_return"] = np.log(test["close"]).diff().fillna(0)

    cols = []
    if "volatility_raw" in val.columns and "volatility_raw" in test.columns:
        cols.append(("volatility_raw", "ATR ratio"))
    cols.append(("log_return", "hourly log return"))

    rows = []
    for col, label in cols:
        v = val[col].dropna().values
        t = test[col].dropna().values
        ks_stat, ks_p = stats.ks_2samp(v, t)
        try:
            wd = stats.wasserstein_distance(v, t)
        except Exception:
            wd = float("nan")
        rows.append({
            "metric": label, "col": col,
            "val_mean": float(v.mean()), "test_mean": float(t.mean()),
            "val_std": float(v.std()),   "test_std": float(t.std()),
            "ks_stat": float(ks_stat),   "ks_p": float(ks_p),
            "wasserstein": float(wd),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "menu_b_distribution_shift.csv", index=False)

    print("\n=== Menu B: Val vs Test distribution shift ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.6f}"))

    # Figure: overlay histograms
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4.5))
    if len(cols) == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, cols):
        v = val[col].dropna().values
        t = test[col].dropna().values
        lo, hi = np.percentile(np.concatenate([v, t]), [1, 99])
        bins = np.linspace(lo, hi, 60)
        ax.hist(v, bins=bins, alpha=0.5, label="Val 2021-2023", color="steelblue", density=True)
        ax.hist(t, bins=bins, alpha=0.5, label="Test 2024+",     color="crimson",   density=True)
        ax.axvline(v.mean(), color="steelblue", linestyle="--", alpha=0.7)
        ax.axvline(t.mean(), color="crimson",   linestyle="--", alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_title(f"{label}\n(KS p={dict(zip([c[0] for c in cols], rows))[col]['ks_p']:.2e})")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle("Phase 15 Menu B - Val vs Test distribution shift")
    fig.tight_layout()
    fig.savefig(out_dir / "menu_b_distribution_shift.png", dpi=130)
    plt.close(fig)
    return {"rows": rows}


# ---------- C: Three-environment summary figures ----------

def menu_c(out_dir: Path) -> None:
    """Three-environment 종합: Val / CPCV / Test 박스플롯 1개 figure."""
    val_b = pd.read_csv("experiments/exp032b_summary.csv")
    cpcv  = pd.read_csv("experiments/exp034_summary.csv")
    test  = pd.read_csv("experiments/exp035_summary.csv")

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5), sharey=True)
    envs = [
        ("Val 2021-2023\n(exp032b, n=10)", val_b, "best_val_sharpe", "variant", None),
        ("CPCV 15 paths\n(exp034)",          cpcv,  "best_sharpe",     "variant", None),
        ("Test 2024+ \n(exp035 exp032b, n=10)",
         test[test.source == "exp032b"], "test_sharpe", "variant", ATR_TEST_SHARPE),
        ("Test 2024+ \n(exp035 exp034, n=15)",
         test[test.source == "exp034"], "test_sharpe", "variant", ATR_TEST_SHARPE),
    ]
    for ax, (title, df, col, var_col, atr) in zip(axes, envs):
        data = [df[df[var_col] == v][col].values for v in VARIANTS]
        bp = ax.boxplot(data, tick_labels=VARIANTS, showmeans=True, patch_artist=True)
        for patch, v in zip(bp["boxes"], VARIANTS):
            patch.set_facecolor(COLOR[v]); patch.set_alpha(0.6)
        # ATR baseline line
        if atr is None:
            atr_line = ATR_VAL_SHARPE
            atr_label = f"ATR Val ({atr_line:+.3f})"
        else:
            atr_line = atr
            atr_label = f"ATR Test ({atr_line:+.3f})"
        ax.axhline(atr_line, color="black", linestyle="--", alpha=0.6, label=atr_label)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Sharpe Ratio")
    fig.suptitle("Phase 15 Menu C - Three-environment Sharpe distribution per variant\n"
                 "(Winner reversal: Val=sym → CPCV=dsr → Test=pt)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "menu_c_three_env.png", dpi=130)
    plt.close(fig)
    print("\n=== Menu C: Three-environment figure saved ===")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/phase15_figures")
    p.add_argument("--md",  default="reports/phase15_analysis.md")
    args = p.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    sig = menu_a(out_dir)
    dist = menu_b(out_dir)
    menu_c(out_dir)

    md = [f"# Phase 15 - Significance + Distribution Shift + Three-env figures\n",
          f"## A) Bootstrap P(RL > ATR) per (variant, environment)\n",
          sig.to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
          f"\n\n## B) Val vs Test distribution shift\n",
          pd.DataFrame(dist["rows"]).to_string(index=False, float_format=lambda x: f"{x:+.6f}"),
          f"\n\n## Figures\n",
          f"- ![menu_a](phase15_figures/menu_a_significance.csv)",
          f"- ![menu_b](phase15_figures/menu_b_distribution_shift.png)",
          f"- ![menu_c](phase15_figures/menu_c_three_env.png)"]
    Path(args.md).write_text("\n".join(md), encoding="utf-8")
    print(f"\nMarkdown -> {args.md}")


if __name__ == "__main__":
    main()
