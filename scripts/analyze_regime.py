"""
scripts/analyze_regime.py

RL best model의 레짐별 행동 분석.

레짐 정의: trend_1w (168h pct_change z-score) 기준
  bull  : trend_1w > +0.5
  bear  : trend_1w < -0.5
  sideways: 그 외

exp024 (4D 절대 gap):
    분석 대상: buy_hi_coef, buy_lo_extra, sell_m_coef, sell_c_coef
exp022 이전 (2D ATR 비례):
    분석 대상: aggressiveness, profit_target

통계 검증: Kruskal-Wallis test (비모수, 3그룹)

사용법:
    python scripts/analyze_regime.py
    python scripts/analyze_regime.py --model-path experiments/exp024_rl_abs/best_model.zip
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
                   default="experiments/exp024_rl_abs/best_model.zip")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="trend_1w z-score 레짐 구분 임계값")
    return p.parse_args()


def collect_actions(model, df, cfg, n_steps=None):
    """Val set 전체를 deterministic policy로 실행, (state, action) 수집.

    exp024 (4D 절대 gap):
        action[0] = buy_hi_coef  ∈ [0,1]
        action[1] = buy_lo_extra ∈ [0,1]
        action[2] = sell_m_coef  ∈ [0,1]
        action[3] = sell_c_coef  ∈ [0,1]

    exp022 이전 (2D ATR 비례, 하위 호환):
        action[0] = aggressiveness ∈ [0,1]
        action[1] = profit_target  ∈ [0,1]
    """
    env = TradingEnv(df, cfg)
    obs, _ = env.reset()

    records = []
    n_steps = n_steps or len(df)

    is_4d = env.action_space.shape[0] == 4

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        trend_1w_zscore = float(obs[6])

        if is_4d:
            records.append({
                "buy_hi_coef":  float(np.clip(action[0], 0, 1)),
                "buy_lo_extra": float(np.clip(action[1], 0, 1)),
                "sell_m_coef":  float(np.clip(action[2], 0, 1)),
                "sell_c_coef":  float(np.clip(action[3], 0, 1)),
                "trend_1w":     trend_1w_zscore,
            })
        else:
            records.append({
                "aggressiveness": float(np.clip(action[0], 0, 1)),
                "profit_target":  float(np.clip(action[1], 0, 1)),
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

    # exp024 (4D) vs exp022 이전 (2D) 자동 분기
    if "buy_hi_coef" in df.columns:
        for col in ["buy_hi_coef", "buy_lo_extra", "sell_m_coef", "sell_c_coef"]:
            print_regime_stats(df, col)
    else:
        print_regime_stats(df, "profit_target")
        print_regime_stats(df, "aggressiveness")

    # 결과 저장
    out_path = Path(args.model_path).parent / "regime_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\n원본 데이터 저장: {out_path}")

    # Regime adaptation 판단
    from scipy.stats import kruskal

    if "buy_hi_coef" in df.columns:
        key_col = "sell_m_coef"   # 수익 결정의 핵심 차원
    else:
        key_col = "profit_target"

    key_groups = [df.loc[df["regime"]==r, key_col].values
                  for r in ["bull","bear","sideways"]]
    _, p_key = kruskal(*key_groups)

    bv_key = df.groupby("regime")[key_col].mean()
    print(f"\n{'='*50}")
    print(f"Regime adaptation 판단 ({key_col})")
    print(f"{'='*50}")
    print(f"  {key_col} Kruskal-Wallis p = {p_key:.4f}")
    print(f"  regime별 {key_col} 평균:")
    for r in ["bull","bear","sideways"]:
        print(f"    {r:>8s}: {bv_key.get(r, float('nan')):.4f}")

    if p_key < 0.05:
        print(f"\n  → Regime adaptation confirmed (p<0.05)")
    else:
        print(f"\n  → No significant regime difference — RL treats all regimes equally")


if __name__ == "__main__":
    main()
