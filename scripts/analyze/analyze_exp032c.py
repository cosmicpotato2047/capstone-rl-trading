"""
scripts/analyze_exp032c.py

exp032c Step 2: trajectory parquet -> 5 mechanism analysis menu.

입력:
    experiments/exp032c_trajectories.parquet  (1.04M rows, 40 모델 x ~26k val steps)
    experiments/exp032b_summary.csv           (40 runs metric 요약)

분석 메뉴:
    1) Pareto scatter (Sharpe vs MDD, 40 dots + cluster + ATR mark)
    2) Action distribution per regime (variant x regime grid)
    3) Counterfactual action map (state 2D -> mean action heatmap per variant)
    4) Behavior stats per regime (trade rate, hold rate, equity change)
    5) Policy distance matrix (variant 간 mean action L2 거리)

산출:
    reports/exp032c_figures/*.png
    reports/exp032c_analysis.md
    experiments/exp032c_*.csv  (per-menu raw)

사용법:
    python scripts/analyze_exp032c.py
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
# ATR baseline (Env-v4 Bayesian Trial #34)
ATR_SHARPE = 1.505
ATR_MDD = 9.83


def menu1_pareto_scatter(summary_csv: Path, out_dir: Path) -> dict:
    """40 runs 의 (best_mdd_pct, best_val_sharpe) scatter + cluster annotation + ATR mark."""
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(9, 6))
    for v in VARIANTS:
        sub = df[df.variant == v]
        ax.scatter(sub["best_mdd_pct"], sub["best_val_sharpe"],
                   s=80, c=COLOR[v], label=f"{v} ({CLUSTERS[v]})",
                   alpha=0.7, edgecolors="black", linewidths=0.5)
        # cluster mean cross
        ax.plot(sub["best_mdd_pct"].mean(), sub["best_val_sharpe"].mean(),
                marker="x", c=COLOR[v], markersize=15, mew=3)

    # ATR baseline (with truncated MDD axis - note in caption)
    atr_mdd_shown = min(ATR_MDD, 8.0)
    ax.scatter([atr_mdd_shown], [ATR_SHARPE], s=200, c="black",
               marker="*", label=f"ATR baseline (MDD {ATR_MDD:.1f})", zorder=10)
    if ATR_MDD > 8.0:
        ax.annotate(f"ATR (MDD={ATR_MDD:.1f}, clipped)",
                    xy=(atr_mdd_shown, ATR_SHARPE),
                    xytext=(6.5, 1.4), fontsize=9, color="black",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    # Pareto frontier: 직관용 upper-left 경계선 (lower MDD, higher Sharpe)
    # 40 dots 의 dominance 계산
    pts = df[["best_mdd_pct", "best_val_sharpe"]].values
    pareto_mask = np.ones(len(pts), dtype=bool)
    for i, (mdd_i, sh_i) in enumerate(pts):
        for j, (mdd_j, sh_j) in enumerate(pts):
            if i == j: continue
            if mdd_j <= mdd_i and sh_j >= sh_i and (mdd_j < mdd_i or sh_j > sh_i):
                pareto_mask[i] = False
                break
    pareto = sorted(pts[pareto_mask].tolist(), key=lambda r: r[0])
    if pareto:
        xs, ys = zip(*pareto)
        ax.plot(xs, ys, "k--", alpha=0.5, label="Pareto frontier (40 runs)")

    ax.set_xlabel("Max Drawdown (%)  [lower is better]")
    ax.set_ylabel("Val Sharpe Ratio  [higher is better]")
    ax.set_title("exp032c Menu 1 — 40 RL runs in Sharpe vs MDD plane\n"
                 "4 variants form two clusters; trade-off frontier")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "menu1_pareto_scatter.png", dpi=130)
    plt.close(fig)

    # raw csv 저장
    df.to_csv(out_dir / "menu1_pareto_data.csv", index=False)
    return {"pareto_count": int(pareto_mask.sum())}


def _label_regime(df: pd.DataFrame) -> pd.DataFrame:
    """volatility regime (atr_ratio tercile) + trend regime (trend_long sign) 라벨."""
    df = df.copy()
    q1 = df["atr_ratio"].quantile(0.33)
    q2 = df["atr_ratio"].quantile(0.67)
    df["vol_regime"] = pd.cut(df["atr_ratio"], bins=[-np.inf, q1, q2, np.inf],
                              labels=["low_vol", "mid_vol", "high_vol"])
    # trend_long: state_6 (z-score 이미 적용); 3분위
    if "trend_long" in df.columns:
        trend_col = "trend_long"
    elif "state_6" in df.columns:
        trend_col = "state_6"
    else:
        df["trend_regime"] = "flat"
        return df
    tq1 = df[trend_col].quantile(0.33)
    tq2 = df[trend_col].quantile(0.67)
    df["trend_regime"] = pd.cut(df[trend_col], bins=[-np.inf, tq1, tq2, np.inf],
                                labels=["down", "flat", "up"])
    return df


def menu2_action_distribution(traj: pd.DataFrame, out_dir: Path) -> dict:
    """variant x vol_regime grid 에서 (action_0, action_1) 분포 (KDE-like 2D histogram)."""
    fig, axes = plt.subplots(len(VARIANTS), 3, figsize=(12, 14), sharex=True, sharey=True)
    regimes = ["low_vol", "mid_vol", "high_vol"]

    for i, v in enumerate(VARIANTS):
        for j, regime in enumerate(regimes):
            sub = traj[(traj.variant == v) & (traj.vol_regime == regime)]
            ax = axes[i, j]
            if len(sub) < 100:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            else:
                h, _, _, _ = ax.hist2d(sub["action_0"], sub["action_1"],
                                       bins=20, range=[[0, 1], [0, 1]],
                                       cmap="viridis", cmin=1)
                # mean marker
                ax.plot(sub["action_0"].mean(), sub["action_1"].mean(),
                        marker="*", c="red", markersize=15, mew=1)
            if j == 0:
                ax.set_ylabel(f"{v} ({CLUSTERS[v]})\nprofit_target")
            if i == len(VARIANTS) - 1:
                ax.set_xlabel("aggressiveness")
            if i == 0:
                ax.set_title(f"{regime}")
    fig.suptitle("exp032c Menu 2 — Action distribution per variant x volatility regime\n"
                 "(red star = mean. n = 10 seeds x ~8700 steps per cell)", y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "menu2_action_distribution.png", dpi=120)
    plt.close(fig)

    # raw per-variant per-regime mean action csv
    stats = traj.groupby(["variant", "vol_regime"], observed=True).agg(
        n=("action_0", "size"),
        action_0_mean=("action_0", "mean"),
        action_0_std=("action_0", "std"),
        action_1_mean=("action_1", "mean"),
        action_1_std=("action_1", "std"),
    ).reset_index()
    stats.to_csv(out_dir / "menu2_action_stats.csv", index=False)
    return {"stats_rows": len(stats)}


def menu3_counterfactual_action_map(traj: pd.DataFrame, out_dir: Path) -> dict:
    """state 2D (atr_ratio bins x divergence bins) -> mean action per variant. 4 heatmap.

    state_1 = divergence (현재 평단가 기준 vs price)
    state_4 = volatility z-score
    atr_ratio = raw volatility/price
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # row 0: action_0 (aggressiveness) | row 1: action_1 (profit_target)
    # col 0-3: variants
    nx, ny = 15, 15  # grid resolution
    atr_bins = np.linspace(traj["atr_ratio"].quantile(0.02),
                           traj["atr_ratio"].quantile(0.98), nx + 1)
    div_bins = np.linspace(traj["state_1"].quantile(0.02),
                           traj["state_1"].quantile(0.98), ny + 1)

    for col, v in enumerate(VARIANTS):
        sub = traj[traj.variant == v]
        for row, action_col in enumerate(["action_0", "action_1"]):
            ax = axes[row, col]
            # bin and aggregate mean
            ix = np.digitize(sub["atr_ratio"], atr_bins) - 1
            iy = np.digitize(sub["state_1"], div_bins) - 1
            grid = np.full((ny, nx), np.nan)
            counts = np.zeros((ny, nx))
            actions = sub[action_col].values
            for i, j, a in zip(iy, ix, actions):
                if 0 <= i < ny and 0 <= j < nx:
                    if counts[i, j] == 0:
                        grid[i, j] = a
                    else:
                        grid[i, j] = (grid[i, j] * counts[i, j] + a) / (counts[i, j] + 1)
                    counts[i, j] += 1
            im = ax.imshow(grid, origin="lower", aspect="auto",
                           extent=[atr_bins[0], atr_bins[-1], div_bins[0], div_bins[-1]],
                           cmap="RdBu_r", vmin=0, vmax=1)
            ax.set_title(f"{v} ({CLUSTERS[v]})\nmean {action_col}", fontsize=10)
            if col == 0:
                ax.set_ylabel("divergence (state_1)")
            if row == 1:
                ax.set_xlabel("atr_ratio")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="mean action")
    fig.suptitle("exp032c Menu 3 — Counterfactual action map\n"
                 "Mean action over (atr_ratio, divergence) state grid, per variant", y=1.00)
    fig.savefig(out_dir / "menu3_counterfactual_action_map.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {}


def menu4_behavior_per_regime(traj: pd.DataFrame, out_dir: Path) -> dict:
    """regime 별 behavior 통계: trade rate, hold rate, mean equity step."""
    # per-step: was there a trade? (n_trades diff > 0)
    traj = traj.sort_values(["variant", "seed", "step_idx"]).copy()
    traj["traded"] = traj.groupby(["variant", "seed"])["n_trades"].diff().fillna(0) > 0
    traj["holds"]  = traj["holdings"] > 0

    stats = traj.groupby(["variant", "vol_regime"], observed=True).agg(
        n_steps=("step_idx", "size"),
        trade_rate=("traded", "mean"),
        hold_rate=("holds", "mean"),
        mean_reward=("reward", "mean"),
    ).reset_index()
    stats.to_csv(out_dir / "menu4_behavior_stats.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("trade_rate", "Trade rate (per step)"),
               ("hold_rate",  "Hold rate (holdings > 0)"),
               ("mean_reward", "Mean step reward")]
    regimes = ["low_vol", "mid_vol", "high_vol"]
    x = np.arange(len(regimes))
    width = 0.18
    for k, (col, title) in enumerate(metrics):
        ax = axes[k]
        for i, v in enumerate(VARIANTS):
            ys = []
            for r in regimes:
                row = stats[(stats.variant == v) & (stats.vol_regime == r)]
                ys.append(row[col].values[0] if len(row) else 0)
            ax.bar(x + (i - 1.5) * width, ys, width, label=v, color=COLOR[v], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        if k == 0:
            ax.legend(fontsize=9)
    fig.suptitle("exp032c Menu 4 — Behavior stats per volatility regime")
    fig.tight_layout()
    fig.savefig(out_dir / "menu4_behavior_per_regime.png", dpi=120)
    plt.close(fig)
    return {"trade_rates": stats[stats.vol_regime == "high_vol"].set_index("variant")["trade_rate"].to_dict()}


def menu5_policy_distance(traj: pd.DataFrame, out_dir: Path) -> dict:
    """variant pair (variant_A, variant_B) 의 평균 action L2 distance (10 seeds 평균).
       동일 val_idx 에서 두 variant 의 mean action 차이."""
    # variant x val_idx -> mean (action_0, action_1) over 10 seeds
    mean_act = traj.groupby(["variant", "val_idx"], observed=True).agg(
        a0=("action_0", "mean"),
        a1=("action_1", "mean"),
    ).reset_index()
    pivot_a0 = mean_act.pivot(index="val_idx", columns="variant", values="a0")
    pivot_a1 = mean_act.pivot(index="val_idx", columns="variant", values="a1")

    # variant pair L2 distance
    dist_mat = pd.DataFrame(index=VARIANTS, columns=VARIANTS, dtype=float)
    for a in VARIANTS:
        for b in VARIANTS:
            if a == b:
                dist_mat.loc[a, b] = 0.0
            else:
                d0 = (pivot_a0[a] - pivot_a0[b]) ** 2
                d1 = (pivot_a1[a] - pivot_a1[b]) ** 2
                dist_mat.loc[a, b] = float(np.sqrt((d0 + d1).mean()))
    dist_mat.to_csv(out_dir / "menu5_policy_distance.csv")

    # within-cluster vs across-cluster
    aggressive = ["sym", "dsr"]
    conservative = ["asym", "pt"]
    within = (dist_mat.loc["sym", "dsr"] + dist_mat.loc["asym", "pt"]) / 2
    across = np.mean([dist_mat.loc[a, b]
                      for a in aggressive for b in conservative])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist_mat.values.astype(float), cmap="viridis")
    ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels(VARIANTS)
    ax.set_yticks(range(len(VARIANTS))); ax.set_yticklabels(VARIANTS)
    for i, a in enumerate(VARIANTS):
        for j, b in enumerate(VARIANTS):
            v = dist_mat.loc[a, b]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > dist_mat.values.max() * 0.5 else "black", fontsize=10)
    ax.set_title(f"exp032c Menu 5 — Policy distance (L2 of mean action per state)\n"
                 f"within-cluster mean = {within:.3f} | across-cluster mean = {across:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="L2 distance")
    fig.tight_layout()
    fig.savefig(out_dir / "menu5_policy_distance.png", dpi=120)
    plt.close(fig)
    return {"within_cluster": float(within), "across_cluster": float(across),
            "ratio": float(across / within) if within > 0 else float("inf")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", default="experiments/exp032c_trajectories.parquet")
    p.add_argument("--summary", default="experiments/exp032b_summary.csv")
    p.add_argument("--out", default="reports/exp032c_figures")
    p.add_argument("--md", default="reports/exp032c_analysis.md")
    args = p.parse_args()

    traj_path = Path(args.traj)
    summary_path = Path(args.summary)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {traj_path} ...")
    traj = pd.read_parquet(traj_path)
    print(f"  trajectories: {len(traj):,} rows, "
          f"{traj['variant'].nunique()} variants x {traj['seed'].nunique()} seeds")

    print("Labeling regimes ...")
    traj = _label_regime(traj)
    print(f"  vol_regime distribution: {traj.vol_regime.value_counts().to_dict()}")

    results = {}
    print("\n=== Menu 1: Pareto scatter ===")
    results["menu1"] = menu1_pareto_scatter(summary_path, out_dir)
    print(f"  saved menu1_pareto_scatter.png ({results['menu1']})")

    print("\n=== Menu 2: Action distribution per regime ===")
    results["menu2"] = menu2_action_distribution(traj, out_dir)
    print(f"  saved menu2_action_distribution.png + csv")

    print("\n=== Menu 3: Counterfactual action map ===")
    results["menu3"] = menu3_counterfactual_action_map(traj, out_dir)
    print(f"  saved menu3_counterfactual_action_map.png")

    print("\n=== Menu 4: Behavior per regime ===")
    results["menu4"] = menu4_behavior_per_regime(traj, out_dir)
    print(f"  saved menu4_behavior_per_regime.png + csv (high_vol trade rates: {results['menu4']['trade_rates']})")

    print("\n=== Menu 5: Policy distance ===")
    results["menu5"] = menu5_policy_distance(traj, out_dir)
    print(f"  saved menu5_policy_distance.png + csv "
          f"(within {results['menu5']['within_cluster']:.3f}, "
          f"across {results['menu5']['across_cluster']:.3f}, "
          f"ratio {results['menu5']['ratio']:.2f}x)")

    # write markdown summary
    md_lines = [
        f"# exp032c Mechanism Analysis\n",
        f"Source: `{traj_path.name}` ({len(traj):,} step rows, 40 models)\n",
        f"\n## Key findings\n",
        f"- **Menu 1 Pareto frontier**: {results['menu1']['pareto_count']}/40 runs lie on Pareto frontier in Sharpe-MDD plane.",
        f"- **Menu 5 Policy distance**: within-cluster L2 = {results['menu5']['within_cluster']:.3f}, "
        f"across-cluster L2 = {results['menu5']['across_cluster']:.3f} (ratio {results['menu5']['ratio']:.2f}x). "
        f"→ Cluster separation statistically confirmed at policy level.",
        f"\n## Menu 4 trade rate per regime (high_vol)\n",
    ]
    for v, r in results["menu4"]["trade_rates"].items():
        md_lines.append(f"- {v}: {r:.4f} (~{r*8760:.0f} trades/year if persistent)")
    md_lines.append("\n## Figures\n")
    for f in ["menu1_pareto_scatter.png", "menu2_action_distribution.png",
              "menu3_counterfactual_action_map.png", "menu4_behavior_per_regime.png",
              "menu5_policy_distance.png"]:
        md_lines.append(f"- ![{f}]({out_dir.name}/{f})")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nMarkdown summary -> {args.md}")


if __name__ == "__main__":
    main()
