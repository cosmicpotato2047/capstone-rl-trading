"""
src/evaluation/metrics.py

포트폴리오 성능 지표 계산 모듈.

모든 에이전트(PPO, Buy-and-Hold, Fixed Grid, ATR Grid)에 동일한 함수를 적용하여
공정한 비교를 보장한다.

주요 지표:
  - total_return_pct  : 누적 수익률 (%)
  - sharpe_ratio      : 연율화 Sharpe Ratio (주 평가 기준)
  - max_drawdown_pct  : 최대 낙폭 (%)
  - n_trades          : 총 체결 횟수
  - avg_cycle_pnl_pct : 사이클 평균 수익률 (%)
  - avg_cycle_hours   : 사이클 평균 소요 시간 (봉)

1시간봉 기준: 연율화 계수 = sqrt(8760)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# 1시간봉 기준 연 거래 시간
HOURS_PER_YEAR: int = 8_760


# ─────────────────────────────────────────────────────────────────────────────
# 개별 지표 함수
# ─────────────────────────────────────────────────────────────────────────────

def total_return_pct(equity_curve: pd.Series, initial_cash: float) -> float:
    """
    누적 수익률 (%).

    Args:
        equity_curve : 시간별 포트폴리오 가치 (pd.Series)
        initial_cash : 초기 자본금

    Returns:
        (최종 equity / initial_cash - 1) × 100
    """
    if len(equity_curve) == 0 or initial_cash <= 0:
        return 0.0
    return (float(equity_curve.iloc[-1]) / initial_cash - 1) * 100


def sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = HOURS_PER_YEAR,
) -> float:
    """
    연율화 Sharpe Ratio.

    Sharpe = mean(r_t - r_f) / std(r_t) × sqrt(periods_per_year)

    r_t: 각 스텝의 단순 수익률 (pct_change)
    r_f: 무위험 수익률 (기본값 0)

    Args:
        equity_curve      : 시간별 포트폴리오 가치
        risk_free_rate    : 연율화 무위험 수익률 (0.0 = 무시)
        periods_per_year  : 연율화 계수 (1h봉 = 8760)

    Returns:
        float. 수익률 분산이 0이면 0.0 반환.
    """
    if len(equity_curve) < 2:
        return 0.0

    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return 0.0

    # 무위험 수익률을 스텝 단위로 변환
    rf_per_step = risk_free_rate / periods_per_year
    excess = returns - rf_per_step

    std = float(excess.std(ddof=1))
    if std < 1e-12:
        return 0.0

    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown_pct(equity_curve: pd.Series) -> float:
    """
    최대 낙폭 (Maximum Drawdown, %).

    MDD = max((peak_t - trough_t) / peak_t) × 100

    Args:
        equity_curve : 시간별 포트폴리오 가치

    Returns:
        양수 값 (예: 낙폭 30% → 30.0). 낙폭 없으면 0.0.
    """
    if len(equity_curve) < 2:
        return 0.0

    rolling_peak = equity_curve.cummax()
    drawdown = (rolling_peak - equity_curve) / rolling_peak
    return float(drawdown.max() * 100)


def avg_cycle_pnl_pct(completed_cycles: list[dict]) -> float:
    """
    완료된 사이클의 평균 수익률 (%).

    Args:
        completed_cycles : [{"pnl_pct": float, ...}, ...]
                           pnl_pct는 소수 형태 (0.05 = 5%)

    Returns:
        평균 pnl_pct × 100. 사이클 없으면 0.0.
    """
    if not completed_cycles:
        return 0.0
    pnls = [c["pnl_pct"] for c in completed_cycles]
    return float(np.mean(pnls) * 100)


def avg_cycle_hours(completed_cycles: list[dict]) -> float:
    """
    완료된 사이클의 평균 소요 시간 (봉 단위 = 시간).

    Args:
        completed_cycles : [{"cycle_hours": int, ...}, ...]

    Returns:
        평균 cycle_hours. 사이클 없으면 0.0.
    """
    if not completed_cycles:
        return 0.0
    hours = [c["cycle_hours"] for c in completed_cycles]
    return float(np.mean(hours))


# ─────────────────────────────────────────────────────────────────────────────
# 편의 함수: 전체 지표 일괄 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(
    equity_curve: pd.Series,
    initial_cash: float,
    n_trades: int,
    completed_cycles: list[dict],
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """
    모든 평가 지표를 한 번에 계산한다.

    Args:
        equity_curve      : 시간별 포트폴리오 가치
        initial_cash      : 초기 자본금
        n_trades          : 총 체결 횟수
        completed_cycles  : 사이클 통계 리스트
        risk_free_rate    : 연율화 무위험 수익률

    Returns:
        {
          "total_return_pct"  : float,  # 누적 수익률 (%)
          "sharpe_ratio"      : float,  # 연율화 Sharpe
          "max_drawdown_pct"  : float,  # 최대 낙폭 (%)
          "n_trades"          : int,
          "n_cycles"          : int,    # 완료 사이클 수
          "avg_cycle_pnl_pct" : float,  # 사이클 평균 수익률 (%)
          "avg_cycle_hours"   : float,  # 사이클 평균 소요 봉
        }
    """
    return {
        "total_return_pct":  total_return_pct(equity_curve, initial_cash),
        "sharpe_ratio":      sharpe_ratio(equity_curve, risk_free_rate),
        "max_drawdown_pct":  max_drawdown_pct(equity_curve),
        "n_trades":          n_trades,
        "n_cycles":          len(completed_cycles),
        "avg_cycle_pnl_pct": avg_cycle_pnl_pct(completed_cycles),
        "avg_cycle_hours":   avg_cycle_hours(completed_cycles),
    }


def print_metrics(metrics: dict[str, float], label: str = "") -> None:
    """지표 dict를 보기 좋게 출력한다."""
    header = f"[ {label} ]" if label else "[ 결과 ]"
    print(f"\n{header}")
    print(f"  {'누적 수익률':<16}: {metrics['total_return_pct']:>8.2f} %")
    print(f"  {'Sharpe Ratio':<16}: {metrics['sharpe_ratio']:>8.3f}")
    print(f"  {'최대 낙폭':<16}: {metrics['max_drawdown_pct']:>8.2f} %")
    print(f"  {'거래 횟수':<16}: {metrics['n_trades']:>8}")
    print(f"  {'완료 사이클':<16}: {metrics['n_cycles']:>8}")
    print(f"  {'사이클 평균 수익':<16}: {metrics['avg_cycle_pnl_pct']:>8.3f} %")
    print(f"  {'사이클 평균 시간':<16}: {metrics['avg_cycle_hours']:>8.1f} 봉")
