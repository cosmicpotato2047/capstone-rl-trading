"""
scripts/make_paper_figures.py
Regenerate all figures used in reports/paper/main.tex with English labels.
Saves to reports/paper/figures/
"""

import sys, re, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from pathlib import Path
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
FIGDIR  = ROOT / "reports" / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

EVAL_PATTERN = re.compile(
    r'\[eval\]\s+step=\s*([\d,]+)\s*\|\s*Sharpe=([+-]?\d+\.\d+)'
    r'\s*\|\s*Return=([+-]?\d+\.\d+)%\s*\|\s*MDD=(\d+\.\d+)%'
)

def parse_eval_log(path):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return pd.DataFrame()
    rows = [(int(s.replace(",","")), float(sh), float(r), float(m))
            for s, sh, r, m in EVAL_PATTERN.findall(text)]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["step","sharpe","return_pct","mdd_pct"])


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Optuna 50-trial Sharpe distribution
# ══════════════════════════════════════════════════════════════════════════════
def make_fig01_distribution():
    df = pd.read_csv(ROOT / "experiments/optuna_coef_v1/all_trials.csv")
    valid = df[df["value"] > 0].copy()
    BASELINE = 17.579
    BEST_VAL  = 42.997

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Bayesian Coefficient Search: 50-Trial Results", fontweight="bold")

    # Histogram
    ax = axes[0]
    ax.hist(valid["value"], bins=15, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(BASELINE, color="orange", lw=2, ls="--", label=f"Baseline  {BASELINE:.3f}")
    ax.axvline(BEST_VAL,  color="crimson",  lw=2, ls="-",  label=f"Best Trial {BEST_VAL:.3f}")
    ax.set_xlabel("Validation Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Sharpe Distribution across 50 Trials")
    ax.legend()

    # Progression
    ax2 = axes[1]
    ax2.scatter(df["number"], df["value"], color="steelblue", s=35, alpha=0.7, zorder=3)
    best_so_far = df["value"].cummax()
    ax2.plot(df["number"], best_so_far, color="crimson", lw=2, label="Cumulative Best")
    ax2.axhline(BASELINE, color="orange", lw=1.5, ls="--", label=f"Baseline {BASELINE:.3f}")
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("Validation Sharpe Ratio")
    ax2.set_title("Sharpe Improvement over Trials")
    ax2.legend()

    plt.tight_layout()
    out = FIGDIR / "fig_01_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Coefficient importance (Pearson correlation)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig03_importance():
    df = pd.read_csv(ROOT / "experiments/optuna_coef_v1/all_trials.csv")
    valid = df[df["value"] > 0].copy()
    coef_cols   = [c for c in df.columns if c.startswith("params_")]
    coef_labels = [c.replace("params_","") for c in coef_cols]

    # Pearson r
    corrs = {}
    for col, label in zip(coef_cols, coef_labels):
        r, p = stats.pearsonr(valid[col], valid["value"])
        corrs[label] = {"r": r, "p": p}
    corr_df = pd.DataFrame(corrs).T.sort_values("r")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["crimson" if r > 0 else "steelblue" for r in corr_df["r"]]
    bars = ax.barh(corr_df.index, corr_df["r"], color=colors, alpha=0.8, edgecolor="white")
    for bar, (_, row) in zip(bars, corr_df.iterrows()):
        x = bar.get_width()
        sig = "*" if row["p"] < 0.05 else ""
        ax.text(x + (0.01 if x >= 0 else -0.01),
                bar.get_y() + bar.get_height()/2,
                f"{x:+.2f}{sig}", va="center",
                ha="left" if x >= 0 else "right", fontsize=10)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r  (with Validation Sharpe)   * p < 0.05")
    ax.set_title("Coefficient Importance — Pearson Correlation with Val Sharpe")
    plt.tight_layout()
    out = FIGDIR / "fig_03_importance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Gap range comparison (default vs optimized)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig06_gap_comparison():
    with open(ROOT / "experiments/optuna_coef_v1/best_coefs.yaml") as f:
        bc = yaml.safe_load(f)
    best = bc["formula_coefs"]
    default = {"A_b":0.5,"B_b":1.5,"C_b":2.5,"D_b":7.5,
               "A_s":0.5,"B_s":1.5,"C_s":2.5,"D_s":7.5}

    agg = np.linspace(0, 1, 200)
    specs = [
        ("buy_hi_gap",      "A_b","B_b","Buy-Hi Gap"),
        ("buy_lo_gap",      "C_b","D_b","Buy-Lo Gap"),
        ("sell_market_gap", "A_s","B_s","Sell-Market Gap"),
        ("sell_cost_gap",   "C_s","D_s","Sell-Cost Gap"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("ATR-Scaled Gap Ranges: Default vs Optimized Coefficients", fontweight="bold")

    for ax, (key, Ak, Bk, title) in zip(axes, specs):
        d_gaps = default[Ak] + default[Bk] * agg
        b_gaps = best[Ak]    + best[Bk]    * agg
        ax.fill_between(agg, 0, d_gaps, alpha=0.35, color="steelblue", label="Default")
        ax.fill_between(agg, 0, b_gaps, alpha=0.35, color="crimson",   label="Optimized")
        ax.plot(agg, d_gaps, color="steelblue", lw=2)
        ax.plot(agg, b_gaps, color="crimson",   lw=2)
        d_lo, d_hi = d_gaps.min(), d_gaps.max()
        b_lo, b_hi = b_gaps.min(), b_gaps.max()
        ax.set_xlabel("Action value  [0, 1]")
        ax.set_ylabel("Gap (x ATR)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.text(0.5, 0.97,
                f"Default [{d_lo:.2f}, {d_hi:.2f}] x ATR\nOptimized [{b_lo:.2f}, {b_hi:.2f}] x ATR",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, bbox=dict(fc="white", alpha=0.7, ec="none"))

    plt.tight_layout()
    out = FIGDIR / "fig_06_gap_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure — Baseline equity curves on Validation set (02_equity_curves)
# ══════════════════════════════════════════════════════════════════════════════
def make_02_equity_curves():
    from src.utils.config import load_config
    from src.agents.baselines import run_all_baselines

    cfg = load_config()
    df_val = pd.read_parquet(ROOT / "data/processed/btc_val.parquet")
    results = run_all_baselines(df_val, cfg)

    style_map = {
        "buy_and_hold":    ("#2196F3", 2.5, "-",  "Buy-and-Hold"),
        "fixed_grid_1pct": ("#4CAF50", 1.5, "-",  "Fixed Grid 1%"),
        "fixed_grid_2pct": ("#81C784", 1.5, "--", "Fixed Grid 2%"),
        "fixed_grid_5pct": ("#C8E6C9", 1.5, ":",  "Fixed Grid 5%"),
        "atr_grid_k0.5":   ("#FF9800", 1.5, "-",  "ATR Grid k=0.5"),
        "atr_grid_k1.0":   ("#F44336", 1.5, "--", "ATR Grid k=1.0"),
        "atr_grid_k2.0":   ("#B71C1C", 1.5, ":",  "ATR Grid k=2.0"),
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    initial_cash = cfg["environment"]["start_capital"]

    for key, (color, lw, ls, label) in style_map.items():
        if key not in results:
            continue
        eq = results[key].get("equity_curve", [])
        if eq:
            ax.plot(eq, color=color, lw=lw, ls=ls, label=label, alpha=0.85)

    ax.axhline(initial_cash, color="gray", ls="--", lw=0.8, alpha=0.5, label="Initial capital")
    ax.set_xlabel("Time Step (hourly candles, Validation set 2023)")
    ax.set_ylabel("Portfolio Value (USDT)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("Baseline Strategy Equity Curves — Validation Set (2023)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGDIR / "02_equity_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure — PPO learning curve (03_learning_curve)
# ══════════════════════════════════════════════════════════════════════════════
def make_03_learning_curve():
    # Two-panel: (a) early experiments before bug fix; (b) post-fix through final
    exp_pre = [
        ("experiments/exp001_baseline/train_log.txt",
         "exp001  (baseline, lr=3e-4)", "#1565C0", "-"),
        ("experiments/exp002_tuned/train_log.txt",
         "exp002  (lr=1e-4, n_steps=4096)", "#E53935", "--"),
    ]
    exp_post = [
        ("experiments/exp008_ent_coef_05/train_log.txt",
         "exp008  (ent_coef=0.05, best=12.4)",  "#1565C0", "-"),
        ("experiments/exp013b_threshold_avgprice/train_log.txt",
         "exp013b (n_splits=2, best=17.6)",      "#FF9800", "-"),
        ("experiments/exp016_final/train_log.txt",
         "exp016  (Bayesian coefs, 3M, best=35.4)", "#2E7D32", "-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.suptitle("PPO Validation Sharpe vs. Training Steps\n"
                 "(★ = best checkpoint per run)", fontsize=13, fontweight="bold")

    # Panel (a) — pre-fix experiments
    ax = axes[0]
    ax.set_title("(a)  Pre-Fix Experiments (exp001–002)\n"
                 "Fee double-counting bug not yet resolved")
    for path, label, color, ls in exp_pre:
        df = parse_eval_log(ROOT / path)
        if df.empty:
            continue
        ax.plot(df["step"] / 1e6, df["sharpe"],
                ls=ls, lw=2, color=color, label=label, alpha=0.9)
        best_idx = df["sharpe"].idxmax()
        ax.scatter([df.loc[best_idx, "step"] / 1e6],
                   [df.loc[best_idx, "sharpe"]],
                   color=color, s=180, zorder=5, marker="*")
    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Training Steps (millions)")
    ax.set_ylabel("Validation Sharpe Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (b) — post-fix experiments
    ax = axes[1]
    ax.set_title("(b)  Post-Fix Experiments (exp008 → exp016)\n"
                 "After fee & eval-pipeline bug fixes")
    for path, label, color, ls in exp_post:
        df = parse_eval_log(ROOT / path)
        if df.empty:
            print(f"    [skip] {path}")
            continue
        ax.plot(df["step"] / 1e6, df["sharpe"],
                ls=ls, lw=2, color=color, label=label, alpha=0.9)
        best_idx = df["sharpe"].idxmax()
        ax.scatter([df.loc[best_idx, "step"] / 1e6],
                   [df.loc[best_idx, "sharpe"]],
                   color=color, s=180, zorder=5, marker="*")
    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Training Steps (millions)")
    ax.set_ylabel("Validation Sharpe Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGDIR / "03_learning_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure — Val Sharpe progression (06_sharpe_progression)
# ══════════════════════════════════════════════════════════════════════════════
def make_06_sharpe_progression():
    experiments = [
        ("exp007\n(no VecNorm)",   14.39,  "VecNormalize removed"),
        ("exp008\n(ent=0.05)",     16.38,  "Entropy tuning"),
        ("exp013b\n(n_splits=2)",  17.579, "Order-sizing optimized"),
        ("Optuna\nTrial #42",      42.997, "Bayesian search (1M steps)"),
        ("exp016\n(full 3M)",      35.424, "Full training, Val Sharpe"),
    ]
    labels  = [e[0] for e in experiments]
    sharpes = [e[1] for e in experiments]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Val Sharpe Progression across Experiment Phases", fontweight="bold")

    # Bar chart
    colors_bar = ["#90CAF9"] * (len(experiments) - 1) + ["#1565C0"]
    bars = axes[0].bar(labels, sharpes, color=colors_bar, edgecolor="white", width=0.6)
    for bar, val in zip(bars, sharpes):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].axhline(17.579, color="orange", lw=1.5, ls="--", label="Baseline (n_splits=2)")
    axes[0].set_ylabel("Validation Sharpe Ratio")
    axes[0].set_title("Val Sharpe by Experiment")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, max(sharpes) * 1.15)

    # Improvement % vs baseline
    baseline = 17.579
    improvements = [(s / baseline - 1) * 100 for s in sharpes]
    colors_imp = ["#EF9A9A" if i < 0 else "#A5D6A7" for i in improvements]
    bars2 = axes[1].bar(labels, improvements, color=colors_imp, edgecolor="white", width=0.6)
    for bar, val in zip(bars2, improvements):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (1 if val >= 0 else -3),
                     f"{val:+.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Improvement vs. Baseline (%)")
    axes[1].set_title("Relative Improvement vs. n_splits Baseline")

    plt.tight_layout()
    out = FIGDIR / "06_sharpe_progression.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Figures 06_test_equity_curve, 06_behavior_analysis, 06_cycle_analysis
#   — require running the model on the test set
# ══════════════════════════════════════════════════════════════════════════════
def run_test_eval_and_make_figures():
    """Run exp016 model on full test set and generate three figures."""
    from copy import deepcopy
    from stable_baselines3 import PPO
    from src.utils.config import load_config
    from src.env.trading_env import BTCGridTradingEnv
    from src.agents.baselines import run_all_baselines

    cfg = load_config(str(ROOT / "config/exp016_final_config.yaml"))
    df_test = pd.read_parquet(ROOT / "data/processed/btc_test.parquet")

    # Run env deterministically
    eval_cfg = deepcopy(cfg)
    eval_cfg["environment"]["random_start"] = False

    model = PPO.load(str(ROOT / "experiments/exp016_final/best_model"), device="cpu")
    env   = BTCGridTradingEnv(df_test, eval_cfg)

    obs, _ = env.reset()
    done   = False
    initial_cash = cfg["environment"]["initial_cash"]

    behavior_records = []
    step = 0

    while not done:
        action, _ = model.predict(obs[None, :], deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action[0])
        done = terminated or truncated

        price     = float(df_test.iloc[env.current_step - 1]["close"])
        # volatility_raw = ATR / price (pre-computed in preprocessing)
        atr_ratio = float(df_test.iloc[env.current_step - 1]["volatility_raw"])
        equity    = env.cash + env.holdings * price

        vol_cat = ("Low"  if atr_ratio < 0.003
                   else "High" if atr_ratio > 0.008
                   else "Mid")
        behavior_records.append({
            "step":           step,
            "aggressiveness": float(action[0][0]),
            "profit_target":  float(action[0][1]),
            "atr_ratio":      atr_ratio,
            "regime":         vol_cat,
            "equity":         equity,
        })
        step += 1

    behavior_df      = pd.DataFrame(behavior_records)
    completed_cycles = list(env.completed_cycles)

    # ── Equity curve figure ─────────────────────────────────────────────────
    bl_results = run_all_baselines(df_test, cfg)
    initial_cash = cfg["environment"]["initial_cash"]
    bl_colors  = {
        "buy_and_hold":    ("#9E9E9E", "Buy & Hold"),
        "fixed_grid_1pct": ("#4CAF50", "Fixed Grid 1%"),
        "fixed_grid_2pct": ("#8BC34A", "Fixed Grid 2%"),
        "fixed_grid_5pct": ("#CDDC39", "Fixed Grid 5%"),
        "atr_grid_k0.5":   ("#FF9800", "ATR Grid k=0.5"),
        "atr_grid_k1.0":   ("#FF5722", "ATR Grid k=1.0"),
        "atr_grid_k2.0":   ("#F44336", "ATR Grid k=2.0"),
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.semilogy(behavior_df["step"].values, behavior_df["equity"].values,
                color="#2196F3", lw=1.5, label="PPO exp016 (best_model)")
    ax.axhline(initial_cash, color="gray", ls="--", alpha=0.4, lw=0.8)
    for key, (color, label) in bl_colors.items():
        if key in bl_results:
            eq = bl_results[key].get("equity_curve", [])
            if len(eq) > 0:
                ax.plot(eq, color=color, lw=1.2, ls="--", alpha=0.7, label=label)
    ax.set_xlabel("Time Step (hourly candles,  Test set 2024-01 to 2026-04)")
    ax.set_ylabel("Portfolio Value (USDT,  log scale)")
    ax.set_title("Test Set Equity Curve — PPO exp016 vs Baselines")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGDIR / "06_test_equity_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved 06_test_equity_curve.png")

    # ── Behavior analysis figure ─────────────────────────────────────────────
    regime_palette = {"Low": "#4CAF50", "Mid": "#FF9800", "High": "#F44336"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Test Set — Agent Behavior Analysis", fontsize=14, fontweight="bold")

    # Aggressiveness by regime
    ax = axes[0, 0]
    for regime, grp in behavior_df.groupby("regime"):
        ax.hist(grp["aggressiveness"], bins=40, alpha=0.65,
                color=regime_palette.get(regime, "gray"),
                label=f"{regime} volatility  (n={len(grp):,})")
    ax.set_xlabel("Aggressiveness  (action[0])")
    ax.set_ylabel("Count")
    ax.set_title("Aggressiveness Distribution by Volatility Regime")
    ax.legend()

    # Profit target
    ax = axes[0, 1]
    ax.hist(behavior_df["profit_target"], bins=40, color="#9C27B0", alpha=0.7)
    mean_pt = behavior_df["profit_target"].mean()
    ax.axvline(mean_pt, color="red", ls="--", lw=1.5, label=f"Mean = {mean_pt:.3f}")
    ax.set_xlabel("Profit Target  (action[1])")
    ax.set_ylabel("Count")
    ax.set_title("Profit Target Distribution")
    ax.legend()

    # Volatility vs aggressiveness scatter
    ax = axes[1, 0]
    sample = behavior_df.sample(min(3000, len(behavior_df)), random_state=42)
    for regime, grp in sample.groupby("regime"):
        ax.scatter(grp["atr_ratio"] * 100, grp["aggressiveness"],
                   s=6, alpha=0.35, color=regime_palette.get(regime, "gray"),
                   label=regime)
    ax.set_xlabel("ATR / Price  (%)")
    ax.set_ylabel("Aggressiveness")
    ax.set_title("Aggressiveness vs. Market Volatility")
    ax.legend(title="Volatility regime")

    # Volatility regime pie
    ax = axes[1, 1]
    regime_counts = behavior_df["regime"].value_counts()
    wedge_colors  = [regime_palette.get(r, "gray") for r in regime_counts.index]
    wedges, _, autotexts = ax.pie(
        regime_counts.values, labels=None,
        colors=wedge_colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.75)
    for t in autotexts:
        t.set_fontsize(10)
    ax.legend(wedges, [f"{r} volatility  ({c:,})"
                       for r, c in regime_counts.items()],
              loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Volatility Regime Distribution")

    plt.tight_layout()
    fig.savefig(FIGDIR / "06_behavior_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved 06_behavior_analysis.png")

    # ── Cycle analysis figure ────────────────────────────────────────────────
    if completed_cycles:
        cycle_df  = pd.DataFrame(completed_cycles)
        win_rate  = (cycle_df["pnl_pct"] > 0).mean() * 100
        avg_pnl   = cycle_df["pnl_pct"].mean() * 100
        avg_hours = cycle_df["cycle_hours"].mean()
        pnl_pct   = cycle_df["pnl_pct"] * 100

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"Test Set — Completed Cycles Analysis  (n = {len(cycle_df):,})",
            fontsize=13, fontweight="bold")

        # PnL distribution
        ax = axes[0]
        ax.hist(pnl_pct, bins=40, color="#2196F3", alpha=0.75, edgecolor="white")
        ax.axvline(0,       color="red",   ls="--", lw=1.5, label="Break-even")
        ax.axvline(avg_pnl, color="green", ls="-",  lw=1.5, label=f"Mean {avg_pnl:.3f}%")
        ax.set_xlabel("Cycle PnL (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"Cycle PnL Distribution  (Win rate {win_rate:.1f}%)")
        ax.legend()

        # Cycle duration
        ax = axes[1]
        ax.hist(cycle_df["cycle_hours"], bins=40, color="#FF9800", alpha=0.75, edgecolor="white")
        ax.axvline(avg_hours, color="red", ls="--", lw=1.5, label=f"Mean {avg_hours:.1f}h")
        ax.set_xlabel("Cycle Duration (hours)")
        ax.set_ylabel("Count")
        ax.set_title("Cycle Duration Distribution")
        ax.legend()

        # Cumulative cycles over time
        ax = axes[2]
        if "end_step" in cycle_df.columns:
            cycle_df_sorted = cycle_df.sort_values("end_step")
            ax.plot(cycle_df_sorted["end_step"].values,
                    range(1, len(cycle_df_sorted) + 1),
                    color="#9C27B0", lw=2)
            ax.set_xlabel("Time Step")
        else:
            ax.plot(range(len(cycle_df)), np.cumsum(cycle_df["pnl_pct"].values) * 100,
                    color="#9C27B0", lw=2)
            ax.set_xlabel("Cycle Number")
        ax.set_ylabel("Cumulative Cycles / Cumulative PnL")
        ax.set_title("Cycle Accumulation over Test Period")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(FIGDIR / "06_cycle_analysis.png", bbox_inches="tight")
        plt.close(fig)
        print("  Saved 06_cycle_analysis.png")
    else:
        print("  [skip] 06_cycle_analysis.png — no completed cycles")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate paper figures in English")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip test-set evaluation (06_* figures that need the model)")
    args = parser.parse_args()

    print("=== Generating paper figures ===")

    print("[1/6] Optuna trial distribution...")
    make_fig01_distribution()

    print("[2/6] Coefficient importance...")
    make_fig03_importance()

    print("[3/6] Gap range comparison...")
    make_fig06_gap_comparison()

    print("[4/6] Val Sharpe progression...")
    make_06_sharpe_progression()

    print("[5/6] PPO learning curve...")
    make_03_learning_curve()

    if not args.no_eval:
        print("[6/6] Test set evaluation + 3 figures (this takes ~2 min)...")
        run_test_eval_and_make_figures()
    else:
        print("[6/6] Skipped (--no-eval)")

    print(f"\nAll figures saved to: {FIGDIR}")
    print("Re-compile the paper:  pdflatex main.tex  (in reports/paper/)")
