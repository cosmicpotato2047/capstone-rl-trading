"""
scripts/analyze_phase16d.py

Phase 16d - pt 의 Test 우위 + DSR 의 Test 실패 메커니즘 분석.

입력:
    experiments/exp032c_trajectories.parquet      (Val, 40 models)
    experiments/exp032c_test_trajectories.parquet (Test, 40 models)

분석 메뉴:
    1) Test per-variant behavior stats (trade rate, hold rate per vol regime)
       → exp032c Menu 4 와 같은 format on Test
    2) Val vs Test behavior shift per variant
       → "같은 모델이 Val/Test 에서 어떻게 다르게 행동"
    3) Hold duration distribution per variant on Test
       → DSR 의 hold rate 가 Test 에서 왜 단점인가
    4) Cycle PnL distribution per variant per environment
    5) Action distribution per variant on Test
       → pt 의 conservative entry 시각화

산출:
    reports/phase16d_figures/*.png
    reports/phase16d_analysis.md
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


def label_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    q1 = df["atr_ratio"].quantile(0.33)
    q2 = df["atr_ratio"].quantile(0.67)
    df["vol_regime"] = pd.cut(df["atr_ratio"], bins=[-np.inf, q1, q2, np.inf],
                              labels=["low_vol", "mid_vol", "high_vol"])
    return df


def menu1_test_behavior(test_traj: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Test trajectory 위에서 variant × vol_regime 의 trade/hold rate."""
    test_traj = test_traj.sort_values(["variant", "seed", "step_idx"]).copy()
    test_traj["traded"] = test_traj.groupby(["variant", "seed"])["n_trades"].diff().fillna(0) > 0
    test_traj["holds"]  = test_traj["holdings"] > 0

    stats = test_traj.groupby(["variant", "vol_regime"], observed=True).agg(
        n=("step_idx", "size"),
        trade_rate=("traded", "mean"),
        hold_rate=("holds", "mean"),
    ).reset_index()
    stats.to_csv(out_dir / "menu1_test_behavior.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    regimes = ["low_vol", "mid_vol", "high_vol"]
    x = np.arange(len(regimes)); width = 0.18
    for k, (ax, col, title) in enumerate([
        (axes[0], "trade_rate", "Test trade rate per step"),
        (axes[1], "hold_rate",  "Test hold rate"),
    ]):
        for i, v in enumerate(VARIANTS):
            ys = [stats[(stats.variant == v) & (stats.vol_regime == r)][col].values[0]
                  if len(stats[(stats.variant == v) & (stats.vol_regime == r)]) else 0
                  for r in regimes]
            ax.bar(x + (i - 1.5) * width, ys, width, label=v, color=COLOR[v], alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(regimes)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        if k == 0:
            ax.legend(fontsize=9)
    fig.suptitle("Phase 16d Menu 1 - Test behavior per volatility regime")
    fig.tight_layout()
    fig.savefig(out_dir / "menu1_test_behavior.png", dpi=130)
    plt.close(fig)
    return stats


def menu2_val_vs_test_shift(val_traj: pd.DataFrame, test_traj: pd.DataFrame,
                             out_dir: Path) -> pd.DataFrame:
    """Val vs Test 의 variant 별 평균 behavior 변화."""
    def _stats(df: pd.DataFrame) -> dict:
        df = df.sort_values(["variant", "seed", "step_idx"]).copy()
        df["traded"] = df.groupby(["variant", "seed"])["n_trades"].diff().fillna(0) > 0
        df["holds"]  = df["holdings"] > 0
        agg = df.groupby("variant").agg(
            trade_rate=("traded", "mean"),
            hold_rate=("holds", "mean"),
            mean_action_0=("action_0", "mean"),
            mean_action_1=("action_1", "mean"),
        )
        return agg

    val_s = _stats(val_traj)
    test_s = _stats(test_traj)

    rows = []
    for v in VARIANTS:
        rows.append({
            "variant": v,
            "trade_rate_val":  val_s.loc[v, "trade_rate"],
            "trade_rate_test": test_s.loc[v, "trade_rate"],
            "trade_rate_delta": test_s.loc[v, "trade_rate"] - val_s.loc[v, "trade_rate"],
            "hold_rate_val":   val_s.loc[v, "hold_rate"],
            "hold_rate_test":  test_s.loc[v, "hold_rate"],
            "hold_rate_delta": test_s.loc[v, "hold_rate"] - val_s.loc[v, "hold_rate"],
            "action_0_val":    val_s.loc[v, "mean_action_0"],
            "action_0_test":   test_s.loc[v, "mean_action_0"],
            "action_0_delta":  test_s.loc[v, "mean_action_0"] - val_s.loc[v, "mean_action_0"],
            "action_1_val":    val_s.loc[v, "mean_action_1"],
            "action_1_test":   test_s.loc[v, "mean_action_1"],
            "action_1_delta":  test_s.loc[v, "mean_action_1"] - val_s.loc[v, "mean_action_1"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "menu2_val_test_shift.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(VARIANTS)); width = 0.35
    for ax, (col_v, col_t, title) in zip(axes, [
        ("trade_rate_val", "trade_rate_test", "Trade rate: Val vs Test"),
        ("hold_rate_val",  "hold_rate_test",  "Hold rate: Val vs Test"),
    ]):
        ax.bar(x - width/2, [df[df.variant == v][col_v].values[0] for v in VARIANTS],
               width, label="Val", color=[COLOR[v] for v in VARIANTS], alpha=0.5)
        ax.bar(x + width/2, [df[df.variant == v][col_t].values[0] for v in VARIANTS],
               width, label="Test", color=[COLOR[v] for v in VARIANTS], alpha=1.0,
               edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(VARIANTS)
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Phase 16d Menu 2 - Behavior shift (Val → Test) per variant")
    fig.tight_layout()
    fig.savefig(out_dir / "menu2_val_test_shift.png", dpi=130)
    plt.close(fig)
    return df


def menu3_hold_duration(test_traj: pd.DataFrame, out_dir: Path) -> dict:
    """Test trajectory 에서 variant 별 hold 지속 시간 분포.
       DSR 의 hold rate 가 Test 에서 왜 단점인가."""
    # 연속 holdings>0 구간 (hold session) length 추출
    test_traj = test_traj.sort_values(["variant", "seed", "step_idx"]).copy()
    test_traj["holds"] = test_traj["holdings"] > 0

    durations_by_variant = {v: [] for v in VARIANTS}
    for (v, s), g in test_traj.groupby(["variant", "seed"]):
        h = g["holds"].values
        # find consecutive runs of True
        i = 0
        while i < len(h):
            if h[i]:
                j = i
                while j < len(h) and h[j]:
                    j += 1
                durations_by_variant[v].append(j - i)
                i = j
            else:
                i += 1

    rows = []
    for v in VARIANTS:
        d = np.array(durations_by_variant[v])
        if len(d) == 0:
            continue
        rows.append({
            "variant": v, "n_sessions": len(d),
            "mean_duration_h": float(d.mean()), "median_duration_h": float(np.median(d)),
            "p95_duration_h": float(np.percentile(d, 95)),
            "max_duration_h": float(d.max()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "menu3_hold_duration.csv", index=False)

    # Histogram (log scale, hours)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(0, np.log10(max(max(d) for d in durations_by_variant.values() if len(d))), 30)
    for v in VARIANTS:
        d = durations_by_variant[v]
        if len(d):
            ax.hist(d, bins=bins, alpha=0.5, label=f"{v} (median {np.median(d):.0f}h, max {max(d):.0f}h)",
                    color=COLOR[v])
    ax.set_xscale("log")
    ax.set_xlabel("Hold session duration (hours, log scale)")
    ax.set_ylabel("count")
    ax.set_title("Phase 16d Menu 3 - Hold session duration on Test per variant\n"
                 "Longer hold = more sell-side timing risk in bull market")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "menu3_hold_duration.png", dpi=130)
    plt.close(fig)
    return {"summary": summary}


def menu4_action_test(test_traj: pd.DataFrame, out_dir: Path) -> None:
    """Test trajectory 에서 variant 별 action 분포 (aggressiveness, profit_target)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, v in enumerate(VARIANTS):
        sub = test_traj[test_traj.variant == v]
        # row 0: aggressiveness histogram
        ax = axes[0, col]
        ax.hist(sub["action_0"], bins=40, range=(0, 1), color=COLOR[v], alpha=0.7)
        ax.axvline(sub["action_0"].mean(), color="black", linestyle="--", alpha=0.6)
        ax.set_title(f"{v} ({CLUSTERS[v]})\nmean agg={sub['action_0'].mean():.2f}")
        if col == 0:
            ax.set_ylabel("aggressiveness density")
        # row 1: profit_target
        ax = axes[1, col]
        ax.hist(sub["action_1"], bins=40, range=(0, 1), color=COLOR[v], alpha=0.7)
        ax.axvline(sub["action_1"].mean(), color="black", linestyle="--", alpha=0.6)
        ax.set_title(f"mean prf_tgt={sub['action_1'].mean():.2f}")
        if col == 0:
            ax.set_ylabel("profit_target density")
        ax.set_xlabel("action value")
    fig.suptitle("Phase 16d Menu 4 - Action distribution on Test per variant")
    fig.tight_layout()
    fig.savefig(out_dir / "menu4_action_test.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-traj",  default="experiments/exp032c_trajectories.parquet")
    p.add_argument("--test-traj", default="experiments/exp032c_test_trajectories.parquet")
    p.add_argument("--out", default="reports/phase16d_figures")
    p.add_argument("--md", default="reports/phase16d_analysis.md")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trajectories...")
    val_traj  = pd.read_parquet(args.val_traj)
    test_traj = pd.read_parquet(args.test_traj)
    test_traj = label_regime(test_traj)
    print(f"  Val: {len(val_traj):,} rows | Test: {len(test_traj):,} rows\n")

    print("=== Menu 1: Test behavior per regime ===")
    s1 = menu1_test_behavior(test_traj, out_dir)
    print(s1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Menu 2: Val vs Test shift ===")
    s2 = menu2_val_vs_test_shift(val_traj, test_traj, out_dir)
    print(s2.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== Menu 3: Hold duration on Test ===")
    h = menu3_hold_duration(test_traj, out_dir)
    print(h["summary"].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Menu 4: Action distribution on Test ===")
    menu4_action_test(test_traj, out_dir)
    print("  saved menu4_action_test.png")

    md = [
        f"# Phase 16d - pt OOS + DSR OOS mechanism analysis\n",
        f"## Menu 1: Test behavior per regime\n",
        s1.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        f"\n\n## Menu 2: Val vs Test behavior shift\n",
        s2.to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
        f"\n\n## Menu 3: Hold duration on Test\n",
        h["summary"].to_string(index=False, float_format=lambda x: f"{x:.2f}"),
        f"\n\n## Figures\n",
        f"- ![Menu 1](phase16d_figures/menu1_test_behavior.png)",
        f"- ![Menu 2](phase16d_figures/menu2_val_test_shift.png)",
        f"- ![Menu 3](phase16d_figures/menu3_hold_duration.png)",
        f"- ![Menu 4](phase16d_figures/menu4_action_test.png)",
    ]
    Path(args.md).write_text("\n".join(md), encoding="utf-8")
    print(f"\nMarkdown -> {args.md}")


if __name__ == "__main__":
    main()
