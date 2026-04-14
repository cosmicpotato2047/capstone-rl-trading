"""
src/evaluation/behavior.py

PPO 에이전트의 행동 분석 모듈.

PPO가 "어떤 시장 레짐에서 어떤 그리드 간격을 선택하는가?"라는
부 연구 질문(Sub-RQ)에 답하기 위한 (state, action) 수집 및 분석.

주요 기능:
  1. collect()       : 에이전트 실행 중 (state, action) 전체 기록
  2. action_stats()  : aggressiveness / profit_target 분포 통계
  3. regime_analysis(): 변동성 레짐별 평균 행동 비교
  4. gap_series()    : 실제 gap 값 시계열 (buy_hi_gap, sell_lo_gap 등)
  5. plot_behavior() : 행동 시각화 (저장 경로 지정 가능)

레짐 정의 (zscore_volatility 기준):
  Low  : zscore_volatility < -0.5
  Mid  : -0.5 ≤ zscore_volatility ≤ 0.5
  High : zscore_volatility > 0.5
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from stable_baselines3 import PPO
    from src.env.trading_env import BTCGridTradingEnv


# ─────────────────────────────────────────────────────────────────────────────
# 레짐 상수
# ─────────────────────────────────────────────────────────────────────────────

REGIME_LOW_THRESHOLD  = -0.5
REGIME_HIGH_THRESHOLD =  0.5


def _assign_regime(zscore_vol: float) -> str:
    if zscore_vol < REGIME_LOW_THRESHOLD:
        return "Low"
    elif zscore_vol > REGIME_HIGH_THRESHOLD:
        return "High"
    else:
        return "Mid"


# ─────────────────────────────────────────────────────────────────────────────
# Action → Gap 변환 (CLAUDE.md 공식과 동일)
# ─────────────────────────────────────────────────────────────────────────────

def actions_to_gaps(actions: np.ndarray) -> pd.DataFrame:
    """
    action 배열 (N, 2)을 실제 gap 값으로 변환한다.

    Args:
        actions : shape (N, 2), columns = [aggressiveness, profit_target]

    Returns:
        DataFrame with columns:
            aggressiveness, profit_target,
            buy_hi_gap, buy_lo_gap, sell_lo_gap, sell_hi_gap
    """
    agg = actions[:, 0]
    pt  = actions[:, 1]

    df = pd.DataFrame({
        "aggressiveness": agg,
        "profit_target":  pt,
        "buy_hi_gap":     0.0001 + agg * 0.05,    # [0.01%,  5%]
        "buy_lo_gap":     0.001  + agg * 0.10,    # [0.10%, 10%]
        "sell_lo_gap":    0.0001 + pt  * 0.05,    # [0.01%,  5%]
        "sell_hi_gap":    0.001  + pt  * 0.15,    # [0.10%, 15%]
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. 수집
# ─────────────────────────────────────────────────────────────────────────────

def collect(
    model: "PPO",
    df: pd.DataFrame,
    config: dict,
    deterministic: bool = True,
) -> pd.DataFrame:
    """
    모델을 df로 1에피소드 실행하며 매 스텝의 (state, action, regime)을 기록한다.

    Args:
        model         : 학습된 PPO 모델
        df            : 평가 데이터 (Val 또는 Train)
        config        : experiment_config.yaml 로드 결과
        deterministic : True=결정론적 행동, False=확률적 탐색

    Returns:
        DataFrame, 1행 = 1스텝:
            step, close, zscore_volatility, regime,
            aggressiveness, profit_target,
            buy_hi_gap, buy_lo_gap, sell_lo_gap, sell_hi_gap,
            cash, holdings, equity, reward
    """
    from src.env.trading_env import BTCGridTradingEnv

    env = BTCGridTradingEnv(df, config)
    obs, _ = env.reset()

    records = []
    done = False
    initial_cash = config["environment"]["initial_cash"]

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_before = env.current_step

        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 현재 봉 정보
        idx   = step_before
        price = float(env.df.loc[idx, "close"])
        zvol  = float(env.df.loc[idx, "zscore_volatility"]) \
                if "zscore_volatility" in env.df.columns else 0.0

        equity = env.cash + env.holdings * price

        records.append({
            "step":              idx,
            "close":             price,
            "zscore_volatility": zvol,
            "regime":            _assign_regime(zvol),
            "aggressiveness":    float(action[0]),
            "profit_target":     float(action[1]),
            "cash":              env.cash,
            "holdings":          env.holdings,
            "equity":            equity,
            "reward":            float(reward),
        })

        obs = obs_next

    result_df = pd.DataFrame(records)

    # gap 컬럼 추가
    gaps = actions_to_gaps(result_df[["aggressiveness", "profit_target"]].values)
    result_df = pd.concat(
        [result_df, gaps[["buy_hi_gap", "buy_lo_gap", "sell_lo_gap", "sell_hi_gap"]]],
        axis=1,
    )

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. 행동 통계
# ─────────────────────────────────────────────────────────────────────────────

def action_stats(behavior_df: pd.DataFrame) -> pd.DataFrame:
    """
    aggressiveness / profit_target / gap 값의 기술통계.

    Returns:
        DataFrame (mean, std, min, 25%, 50%, 75%, max)
    """
    cols = ["aggressiveness", "profit_target",
            "buy_hi_gap", "buy_lo_gap", "sell_lo_gap", "sell_hi_gap"]
    return behavior_df[cols].describe().T


# ─────────────────────────────────────────────────────────────────────────────
# 3. 레짐별 행동 비교
# ─────────────────────────────────────────────────────────────────────────────

def regime_analysis(behavior_df: pd.DataFrame) -> pd.DataFrame:
    """
    변동성 레짐(Low/Mid/High)별 평균 행동값 비교.

    Sub-RQ: "어떤 시장 레짐에서 어떤 간격을 선택하는가?"

    Returns:
        DataFrame, index=regime, columns=gap 지표들
    """
    cols = ["aggressiveness", "profit_target",
            "buy_hi_gap", "buy_lo_gap", "sell_lo_gap", "sell_hi_gap"]

    result = (
        behavior_df.groupby("regime")[cols]
        .agg(["mean", "std", "count"])
    )
    # 레짐 순서 정렬
    order = ["Low", "Mid", "High"]
    result = result.reindex([r for r in order if r in result.index])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. 레짐별 간단 요약 (출력용)
# ─────────────────────────────────────────────────────────────────────────────

def print_regime_summary(behavior_df: pd.DataFrame) -> None:
    """레짐별 평균 aggressiveness / profit_target을 보기 좋게 출력한다."""
    cols = ["aggressiveness", "profit_target", "buy_hi_gap", "sell_lo_gap"]
    summary = behavior_df.groupby("regime")[cols].mean()
    order = ["Low", "Mid", "High"]
    summary = summary.reindex([r for r in order if r in summary.index])

    print("\n[ 레짐별 평균 행동 ]")
    print(f"{'레짐':<6} {'Aggressive':>11} {'ProfitTgt':>10} "
          f"{'BuyHiGap%':>10} {'SellLoGap%':>11}")
    print("-" * 52)
    for regime, row in summary.iterrows():
        print(
            f"{regime:<6} "
            f"{row['aggressiveness']:>11.4f} "
            f"{row['profit_target']:>10.4f} "
            f"{row['buy_hi_gap']*100:>10.3f} "
            f"{row['sell_lo_gap']*100:>11.3f}"
        )

    # 레짐 분포
    counts = behavior_df["regime"].value_counts()
    total  = len(behavior_df)
    print(f"\n[ 레짐 분포 ] 총 {total:,}스텝")
    for r in order:
        if r in counts:
            print(f"  {r:<5}: {counts[r]:>6,}스텝 ({counts[r]/total*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_behavior(
    behavior_df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = False,
) -> None:
    """
    행동 시각화 4종:
      (a) aggressiveness 시계열
      (b) profit_target 시계열
      (c) 레짐별 aggressiveness 박스플롯
      (d) 레짐별 profit_target 박스플롯

    Args:
        behavior_df : collect() 반환 DataFrame
        save_dir    : 저장 폴더 (None이면 저장 안 함)
        show        : True이면 plt.show() 호출
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 한글 폰트
    plt.rcParams["font.family"]       = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("PPO 행동 분석", fontsize=14, fontweight="bold")

    steps = behavior_df["step"].values
    regime_colors = {"Low": "#2196F3", "Mid": "#FF9800", "High": "#F44336"}
    colors = behavior_df["regime"].map(regime_colors).values

    # (a) aggressiveness 시계열
    ax = axes[0, 0]
    ax.scatter(steps, behavior_df["aggressiveness"], c=colors, s=1, alpha=0.4)
    ax.set_title("Aggressiveness (매수 공격성)")
    ax.set_xlabel("Step")
    ax.set_ylabel("값 [0, 1]")
    ax.set_ylim(-0.05, 1.05)

    # (b) profit_target 시계열
    ax = axes[0, 1]
    ax.scatter(steps, behavior_df["profit_target"], c=colors, s=1, alpha=0.4)
    ax.set_title("Profit Target (매도 목표)")
    ax.set_xlabel("Step")
    ax.set_ylabel("값 [0, 1]")
    ax.set_ylim(-0.05, 1.05)

    # 레전드
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markersize=8, label=r)
        for r, c in regime_colors.items()
        if r in behavior_df["regime"].values
    ]
    axes[0, 1].legend(handles=legend_elements, title="레짐", loc="upper right")

    # (c) 레짐별 aggressiveness 박스플롯
    ax = axes[1, 0]
    order = [r for r in ["Low", "Mid", "High"] if r in behavior_df["regime"].values]
    data_agg = [behavior_df[behavior_df["regime"] == r]["aggressiveness"].values
                for r in order]
    bp = ax.boxplot(data_agg, labels=order, patch_artist=True)
    for patch, r in zip(bp["boxes"], order):
        patch.set_facecolor(regime_colors[r])
        patch.set_alpha(0.7)
    ax.set_title("레짐별 Aggressiveness 분포")
    ax.set_ylabel("Aggressiveness")

    # (d) 레짐별 profit_target 박스플롯
    ax = axes[1, 1]
    data_pt = [behavior_df[behavior_df["regime"] == r]["profit_target"].values
               for r in order]
    bp = ax.boxplot(data_pt, labels=order, patch_artist=True)
    for patch, r in zip(bp["boxes"], order):
        patch.set_facecolor(regime_colors[r])
        patch.set_alpha(0.7)
    ax.set_title("레짐별 Profit Target 분포")
    ax.set_ylabel("Profit Target")

    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / "ppo_behavior_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"행동 분석 그래프 저장: {path}")

    if show:
        plt.show()

    plt.close(fig)
