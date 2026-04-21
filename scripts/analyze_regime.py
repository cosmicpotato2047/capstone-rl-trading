"""
scripts/analyze_regime.py

exp018 best model의 레짐별 행동 분석.

레짐 정의: trend_1w (168h pct_change z-score) 기준
  bull  : trend_1w > +0.5
  bear  : trend_1w < -0.5
  sideways: 그 외

분석 대상: profit_target, aggressiveness 분포를 레짐별로 비교
통계 검증: Kruskal-Wallis test (비모수, 3그룹)

사용법:
    python scripts/analyze_regime.py
    python scripts/analyze_regime.py --model-path experiments/exp018_final/best_model.zip
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.env.trading_env import BTCGridTradingEnv as TradingEnv
from stable_baselines3 import PPO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str,
                   default="experiments/exp018_final/best_model.zip")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="trend_1w z-score 레짐 구분 임계값")
    return p.parse_args()


def collect_actions(model, df, cfg, n_steps=None):
    """Val set 전체를 deterministic policy로 실행, (state, action) 수집."""
    env = TradingEnv(df, cfg)
    obs, _ = env.reset()

    records = []
    n_steps = n_steps or len(df)

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        aggressiveness = float(np.clip(action[0], 0, 1))
        profit_target  = float(np.clip(action[1], 0, 1))

        # trend_1w는 obs[6] (7D state 기준)
        trend_1w_zscore = float(obs[6])

        records.append({
            "aggressiveness": aggressiveness,
            "profit_target":  profit_target,
            "trend_1w":       trend_1w_zscore,
        })

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    return pd.DataFrame(records)


def assign_regime(df, threshold):
    conditions = [
        df["trend_1w"] >  threshold,
        df["trend_1w"] < -threshold,
    ]
    choices = ["bull", "bear"]
    df["regime"] = np.select(conditions, choices, default="sideways")
    return df


def print_regime_stats(df, col):
    print(f"\n── {col} ──────────────────────────────")
    groups = {}
    for regime in ["bull", "bear", "sideways"]:
        g = df.loc[df["regime"] == regime, col].values
        groups[regime] = g
        print(f"  {regime:>8s}  n={len(g):5d}  "
              f"mean={g.mean():.4f}  median={np.median(g):.4f}  std={g.std():.4f}")

    # Kruskal-Wallis (3그룹 비모수 검정)
    stat, p = stats.kruskal(groups["bull"], groups["bear"], groups["sideways"])
    print(f"\n  Kruskal-Wallis  H={stat:.3f}  p={p:.4f}  "
          f"{'OK (p&lt;0.05)' if p < 0.05 else 'NOT significant'}")

    # 사후 검정: Mann-Whitney U (bull vs bear)
    u, p2 = stats.mannwhitneyu(groups["bull"], groups["bear"], alternative="two-sided")
    print(f"  Mann-Whitney (bull vs bear)  U={u:.0f}  p={p2:.4f}  "
          f"{'OK' if p2 < 0.05 else 'NOT significant'}")


def main():
    args = parse_args()

    cfg      = load_config()
    df_val   = pd.read_parquet("data/processed/btc_val.parquet")
    model    = PPO.load(args.model_path)

    print(f"모델 로드: {args.model_path}")
    print(f"Val 데이터: {len(df_val):,} 스텝")
    print(f"레짐 임계값: trend_1w z-score ±{args.threshold}")

    df = collect_actions(model, df_val, cfg)
    df = assign_regime(df, args.threshold)

    print(f"\n레짐 분포:")
    for r, cnt in df["regime"].value_counts().items():
        print(f"  {r:>8s}: {cnt:5d} ({cnt/len(df)*100:.1f}%)")

    print_regime_stats(df, "profit_target")
    print_regime_stats(df, "aggressiveness")

    # 결과 저장
    out_path = Path(args.model_path).parent / "regime_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\n원본 데이터 저장: {out_path}")

    # Phase 2 완료 기준 판단
    from scipy.stats import kruskal
    pt_groups = [df.loc[df["regime"]==r, "profit_target"].values
                 for r in ["bull","bear","sideways"]]
    _, p_pt = kruskal(*pt_groups)

    bv_pt = df.groupby("regime")["profit_target"].mean()
    print(f"\n{'='*50}")
    print(f"Phase 2 완료 기준 판단")
    print(f"{'='*50}")
    print(f"  profit_target Kruskal-Wallis p = {p_pt:.4f}")
    print(f"  regime별 profit_target 평균:")
    for r in ["bull","bear","sideways"]:
        print(f"    {r:>8s}: {bv_pt.get(r, float('nan')):.4f}")

    if p_pt < 0.05:
        print(f"\n  → Regime adaptation confirmed (p&lt;0.05)")
        print(f"  → Phase 2 완료 기준 충족 여부: Val Sharpe 확인 필요")
    else:
        print(f"\n  → No significant regime difference")
        print(f"  → 고정 공식 라이브 트레이딩으로 전환 검토 필요")


if __name__ == "__main__":
    main()
