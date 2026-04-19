"""
src/agents/baselines.py

비교 기준 에이전트 (Baselines).

PPO 성능 비교를 위한 규칙 기반 전략 3종:
  1. BuyAndHoldAgent     — 최초 전액 매수 후 보유
  2. FixedGridAgent      — 고정 gap% 그리드 (매수/매도 동일 간격)
  3. ATRGridAgent        — ATR 비례 gap (gap = k × ATR/price)

공통 인터페이스:
  agent.run(df, config) → dict
    {
      "equity_curve":     pd.Series,   # 시간별 포트폴리오 가치
      "total_return_pct": float,        # 누적 수익률 (%)
      "n_trades":         int,
      "completed_cycles": list,         # FixedGrid / ATRGrid만 해당
    }

설계 원칙:
- BTCGridTradingEnv와 동일한 수수료(fee_rate), 초기 자본(initial_cash) 사용
- 체결 판단: 다음 봉 high/low 기준 (환경과 동일)
- SELL 우선 원칙 적용 (환경과 동일)
- Test 파티션 열람 금지 — df는 외부에서 주입
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────────────────────
# 공통 베이스
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """모든 베이스라인 에이전트의 공통 인터페이스."""

    def __init__(self, config: dict):
        env_cfg = config["environment"]
        self.initial_cash: float = env_cfg["initial_cash"]
        self.fee_rate:     float = env_cfg["transaction_cost"]
        self.warmup:       int   = config["indicators"]["atr_period"]

    @abstractmethod
    def run(self, df: pd.DataFrame) -> dict:
        """
        df 전체를 시뮬레이션하고 결과 dict를 반환한다.

        Returns:
            equity_curve     (pd.Series) : 인덱스=timestamp, 값=포트폴리오 총가치
            total_return_pct (float)     : (최종 equity / initial_cash - 1) × 100
            n_trades         (int)       : 총 체결 횟수
            completed_cycles (list)      : 사이클 통계 (그리드 전략만)
        """

    def _equity(self, cash: float, holdings: float, price: float) -> float:
        return cash + holdings * price

    def _buy(
        self,
        cash: float,
        holdings: float,
        avg_price: float,
        fill_price: float,
        spend: float,
    ) -> tuple[float, float, float]:
        """
        지정 금액(spend)으로 매수.
        Returns: (new_cash, new_holdings, new_avg_price)
        """
        spend = min(spend, cash)
        if spend <= 0:
            return cash, holdings, avg_price
        fee       = spend * self.fee_rate
        net_spend = spend - fee
        qty       = net_spend / fill_price

        total_cost  = avg_price * holdings + fill_price * qty
        new_holdings = holdings + qty
        new_avg      = total_cost / new_holdings if new_holdings > 0 else 0.0
        return cash - spend, new_holdings, new_avg

    def _sell(
        self,
        cash: float,
        holdings: float,
        fill_price: float,
        qty: float,
    ) -> tuple[float, float]:
        """
        qty BTC를 fill_price에 매도.
        Returns: (new_cash, new_holdings)
        """
        qty = min(qty, holdings)
        if qty <= 0:
            return cash, holdings
        proceeds  = qty * fill_price
        fee       = proceeds * self.fee_rate
        return cash + proceeds - fee, holdings - qty


# ─────────────────────────────────────────────────────────────────────────────
# 1. Buy-and-Hold
# ─────────────────────────────────────────────────────────────────────────────

class BuyAndHoldAgent(BaseAgent):
    """
    warmup 이후 첫 봉에서 전액 매수, 에피소드 종료까지 보유.
    수수료 1회(매수) 부담.
    """

    def run(self, df: pd.DataFrame) -> dict:
        df = df.reset_index(drop=False)   # timestamp 컬럼으로 보존
        start = self.warmup

        cash     = self.initial_cash
        holdings = 0.0
        avg_price = 0.0
        n_trades  = 0
        equity_list = []

        for i in range(len(df)):
            price = float(df.loc[i, "close"])

            # warmup 이후 첫 봉에서 전액 매수
            if i == start and holdings == 0.0:
                cash, holdings, avg_price = self._buy(
                    cash, holdings, avg_price, price, cash
                )
                n_trades += 1

            equity_list.append(self._equity(cash, holdings, price))

        timestamps = df.iloc[:, 0] if df.columns[0] != "close" else df.index

        # DataFrame에 원래 인덱스가 있었다면 복원
        try:
            index = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns \
                    else df.iloc[:, 0]
        except Exception:
            index = range(len(equity_list))

        equity_curve = pd.Series(equity_list, index=df.index if hasattr(df, 'index') else range(len(equity_list)))

        final_equity = equity_list[-1]
        return {
            "equity_curve":     equity_curve,
            "total_return_pct": (final_equity / self.initial_cash - 1) * 100,
            "n_trades":         n_trades,
            "completed_cycles": [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fixed Grid Agent
# ─────────────────────────────────────────────────────────────────────────────

class FixedGridAgent(BaseAgent):
    """
    고정 gap%로 buy/sell 주문을 매 봉마다 갱신하는 그리드 봇.

    매수/매도 간격 동일(symmetric grid):
        buy_price  = price × (1 - gap)
        sell_price = price × (1 + gap)  — 미보유 시: avg_price × (1 + gap)

    주문 크기: BTCGridTradingEnv의 n_splits / n_buy_orders 방식과 동일.
    사이클 구조: holdings 0 → buy → holdings > 0 → 전량 매도 → holdings 0.
    """

    def __init__(self, config: dict, gap: float):
        """
        Args:
            config: experiment_config.yaml 로드 결과
            gap:    그리드 간격 비율 (예: 0.01 = 1%)
        """
        super().__init__(config)
        self.gap       = gap
        env_cfg        = config["environment"]
        self.n_splits  = env_cfg["n_splits"]
        self.n_buy     = env_cfg["n_buy_orders"]

    def run(self, df: pd.DataFrame) -> dict:
        df = df.reset_index(drop=True)

        cash      = self.initial_cash
        holdings  = 0.0
        avg_price = 0.0
        n_trades  = 0
        in_cycle  = False
        cycle_start_cash    = self.initial_cash
        cycle_start_step    = self.warmup
        cycle_slot_size     = 0.0
        per_order_size      = 0.0
        budget_remaining    = 0.0
        completed_cycles    = []
        equity_list         = []

        for i in range(len(df)):
            price = float(df.loc[i, "close"])
            equity_list.append(self._equity(cash, holdings, price))

            if i < self.warmup or i >= len(df) - 1:
                continue

            next_high  = float(df.loc[i + 1, "high"])
            next_low   = float(df.loc[i + 1, "low"])
            next_price = float(df.loc[i + 1, "close"])

            # 주문 가격 계산
            buy_price  = price * (1 - self.gap)
            ref        = avg_price if avg_price > 0 else price
            sell_price = ref * (1 + self.gap)

            # ── SELL 먼저 ──────────────────────────────────────
            if holdings > 0.0 and next_high >= sell_price:
                avg_sell = sell_price   # fixed grid: sell 1개
                threshold_btc = (cycle_slot_size / avg_sell
                                 if cycle_slot_size > 0 and avg_sell > 0 else 0.0)
                sell_qty = (holdings if holdings <= threshold_btc
                            else holdings / self.n_splits)
                sell_qty = min(sell_qty, holdings)

                cash, holdings = self._sell(cash, holdings, sell_price, sell_qty)
                n_trades += 1

                if holdings < 1e-10:
                    holdings  = 0.0
                    # 사이클 종료
                    if in_cycle:
                        cycle_pnl = (cash - cycle_start_cash) / cycle_start_cash
                        completed_cycles.append({
                            "start_step":  cycle_start_step,
                            "end_step":    i + 1,
                            "cycle_hours": max((i + 1) - cycle_start_step, 1),
                            "pnl_pct":     cycle_pnl,
                        })
                        in_cycle          = False
                        cycle_start_cash  = cash

            # ── BUY ────────────────────────────────────────────
            if next_low <= buy_price:
                # 사이클 시작
                if not in_cycle and holdings == 0.0:
                    in_cycle         = True
                    cycle_start_cash = cash
                    cycle_start_step = i
                    cycle_slot_size  = cash / self.n_splits
                    per_order_size   = cycle_slot_size / self.n_buy
                    budget_remaining = cash

                if in_cycle and budget_remaining >= per_order_size:
                    cash, holdings, avg_price = self._buy(
                        cash, holdings, avg_price, buy_price, per_order_size
                    )
                    budget_remaining -= per_order_size
                    n_trades += 1

        final_equity = self._equity(cash, holdings, float(df.loc[len(df)-1, "close"]))
        equity_curve = pd.Series(equity_list)

        return {
            "equity_curve":     equity_curve,
            "total_return_pct": (final_equity / self.initial_cash - 1) * 100,
            "n_trades":         n_trades,
            "completed_cycles": completed_cycles,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ATR-Proportional Grid Agent
# ─────────────────────────────────────────────────────────────────────────────

class ATRGridAgent(BaseAgent):
    """
    ATR 비례 gap으로 그리드 간격을 동적으로 조정하는 규칙 기반 봇.

        gap = k × (ATR(168) / price)   = k × volatility_raw

    변동성이 높을수록 그리드를 넓게, 낮을수록 좁게 설정한다.
    나머지 주문 구조(사이클, 예산 분할)는 FixedGridAgent와 동일.
    """

    def __init__(self, config: dict, k: float):
        """
        Args:
            config: experiment_config.yaml 로드 결과
            k:      ATR 배수 (예: 1.0 → gap = 1 × ATR/price)
        """
        super().__init__(config)
        self.k        = k
        env_cfg       = config["environment"]
        self.n_splits = env_cfg["n_splits"]
        self.n_buy    = env_cfg["n_buy_orders"]

    def run(self, df: pd.DataFrame) -> dict:
        df = df.reset_index(drop=True)

        if "volatility_raw" not in df.columns:
            raise ValueError("df에 'volatility_raw' 컬럼이 필요합니다. "
                             "preprocess_data.py 실행 후 사용하세요.")

        cash      = self.initial_cash
        holdings  = 0.0
        avg_price = 0.0
        n_trades  = 0
        in_cycle  = False
        cycle_start_cash = self.initial_cash
        cycle_start_step = self.warmup
        cycle_slot_size  = 0.0
        per_order_size   = 0.0
        budget_remaining = 0.0
        completed_cycles = []
        equity_list      = []

        for i in range(len(df)):
            price = float(df.loc[i, "close"])
            equity_list.append(self._equity(cash, holdings, price))

            if i < self.warmup or i >= len(df) - 1:
                continue

            next_high  = float(df.loc[i + 1, "high"])
            next_low   = float(df.loc[i + 1, "low"])
            next_price = float(df.loc[i + 1, "close"])

            # ATR 비례 gap 계산
            vol_raw   = float(df.loc[i, "volatility_raw"])
            gap       = self.k * vol_raw
            gap       = max(gap, 0.0001)   # 최소 0.01% 보장

            buy_price  = price * (1 - gap)
            ref        = avg_price if avg_price > 0 else price
            sell_price = ref * (1 + gap)

            # ── SELL 먼저 ──────────────────────────────────────
            if holdings > 0.0 and next_high >= sell_price:
                threshold_btc = (cycle_slot_size / sell_price
                                 if cycle_slot_size > 0 and sell_price > 0 else 0.0)
                sell_qty = (holdings if holdings <= threshold_btc
                            else holdings / self.n_splits)
                sell_qty = min(sell_qty, holdings)

                cash, holdings = self._sell(cash, holdings, sell_price, sell_qty)
                n_trades += 1

                if holdings < 1e-10:
                    holdings = 0.0
                    if in_cycle:
                        cycle_pnl = (cash - cycle_start_cash) / cycle_start_cash
                        completed_cycles.append({
                            "start_step":  cycle_start_step,
                            "end_step":    i + 1,
                            "cycle_hours": max((i + 1) - cycle_start_step, 1),
                            "pnl_pct":     cycle_pnl,
                        })
                        in_cycle         = False
                        cycle_start_cash = cash

            # ── BUY ────────────────────────────────────────────
            if next_low <= buy_price:
                if not in_cycle and holdings == 0.0:
                    in_cycle         = True
                    cycle_start_cash = cash
                    cycle_start_step = i
                    cycle_slot_size  = cash / self.n_splits
                    per_order_size   = cycle_slot_size / self.n_buy
                    budget_remaining = cash

                if in_cycle and budget_remaining >= per_order_size:
                    cash, holdings, avg_price = self._buy(
                        cash, holdings, avg_price, buy_price, per_order_size
                    )
                    budget_remaining -= per_order_size
                    n_trades += 1

        final_equity = self._equity(cash, holdings, float(df.loc[len(df)-1, "close"]))
        equity_curve = pd.Series(equity_list)

        return {
            "equity_curve":     equity_curve,
            "total_return_pct": (final_equity / self.initial_cash - 1) * 100,
            "n_trades":         n_trades,
            "completed_cycles": completed_cycles,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 편의 함수: 전체 베이스라인 일괄 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_all_baselines(df: pd.DataFrame, config: dict) -> dict[str, dict]:
    """
    설정 파일에 정의된 모든 베이스라인을 실행하고 결과를 반환한다.

    Returns:
        {
          "buy_and_hold":    {...},
          "fixed_grid_1pct": {...},
          "fixed_grid_2pct": {...},
          "fixed_grid_5pct": {...},
          "atr_grid_k0.5":   {...},
          "atr_grid_k1.0":   {...},
          "atr_grid_k2.0":   {...},
        }
    """
    bl_cfg = config["baselines"]
    results = {}

    # 1. Buy-and-Hold
    if bl_cfg.get("buy_and_hold", True):
        results["buy_and_hold"] = BuyAndHoldAgent(config).run(df)

    # 2. Fixed Grid
    for gap in bl_cfg["fixed_grid"]["gaps"]:
        key = f"fixed_grid_{int(gap*100)}pct"
        results[key] = FixedGridAgent(config, gap=gap).run(df)

    # 3. ATR Grid
    for k in bl_cfg["atr_grid"]["k_values"]:
        key = f"atr_grid_k{k}"
        results[key] = ATRGridAgent(config, k=k).run(df)

    return results


def print_summary(results: dict[str, dict]) -> None:
    """결과 dict를 보기 좋게 출력한다."""
    print(f"{'전략':<22} {'수익률(%)':>10} {'거래횟수':>8} {'완료사이클':>10}")
    print("-" * 54)
    for name, r in results.items():
        cycles = len(r.get("completed_cycles", []))
        print(f"{name:<22} {r['total_return_pct']:>10.2f} "
              f"{r['n_trades']:>8} {cycles:>10}")
