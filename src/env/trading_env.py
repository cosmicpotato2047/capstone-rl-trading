"""
BTC 그리드 트레이딩 Gymnasium 환경

State (5차원, rolling z-score 정규화):
    [0] log_price            = log(price / price.rolling(168).mean())
    [1] divergence           = (avg_price - price) / avg_price  (미보유 시 0)
    [2] holdings_value_ratio = (holdings × price) / start_capital  (미보유 시 0)
    [3] cash_ratio           = cash / start_capital
    [4] volatility           = ATR(168) / price

Action (2차원 연속, [0, 1]²):
    [0] aggressiveness → buy_hi_gap, buy_lo_gap 결정
    [1] profit_target  → sell_lo_gap, sell_hi_gap 결정

주문 (매 스텝 4개 지정가 갱신):
    buy_hi  = price × (1 - buy_hi_gap)   # 공격적 매수
    buy_lo  = price × (1 - buy_lo_gap)   # 보수적 매수
    sell_lo = price × (1 + sell_lo_gap)  # 빠른 익절
    sell_hi = avg_price × (1 + sell_hi_gap)  # 느린 익절 (평단 기준)

체결: 다음 봉 high/low 기준
    next_high >= sell_lo  →  sell_lo 체결
    next_low  <= buy_hi   →  buy_hi  체결

Reward:
    매 스텝: (equity_t - equity_{t-1}) / start_capital - fee × n_trades
    사이클 종료 시: cycle_pnl_pct + alpha / cycle_hours 추가
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class BTCGridTradingEnv(gym.Env):
    """BTC 동적 그리드 트레이딩 환경 (PPO 연속 행동 공간)."""

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, config: dict):
        """
        Args:
            df: 전처리 완료된 OHLCV + 지표 DataFrame.
                필수 컬럼: open, high, low, close, volume,
                           log_price, atr, zscore_log_price,
                           zscore_volatility
            config: experiment_config.yaml 로드 결과
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.cfg_env = config["environment"]
        self.cfg_ind = config["indicators"]

        # ── 환경 파라미터 ──────────────────────────────────────
        self.start_capital: float = self.cfg_env["initial_cash"]
        self.fee_rate: float = self.cfg_env["transaction_cost"]
        self.order_size_frac: float = self.cfg_env["order_size_fraction"]
        self.cycle_alpha: float = self.cfg_env["cycle_alpha"]
        self.warmup: int = self.cfg_ind["atr_period"]  # 168봉

        # ── Gymnasium 공간 정의 ────────────────────────────────
        # Action: [aggressiveness, profit_target] ∈ [0, 1]²
        self.action_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(2,),
            dtype=np.float32,
        )

        # Observation: z-score 정규화 후 대략 [-4, 4] 범위
        self.observation_space = spaces.Box(
            low=np.float32(-5.0),
            high=np.float32(5.0),
            shape=(5,),
            dtype=np.float32,
        )

        # 내부 상태 (reset에서 초기화)
        self._init_state()

    # ──────────────────────────────────────────────────────────
    # Gymnasium 인터페이스
    # ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.warmup

        # 포트폴리오
        self.cash: float = self.start_capital
        self.holdings: float = 0.0      # BTC 수량
        self.avg_price: float = 0.0     # 평단가

        # 사이클 추적
        self.in_cycle: bool = False
        self.cycle_start_cash: float = self.start_capital
        self.cycle_start_step: int = self.warmup

        # 통계 (디버깅/분석용)
        self.n_trades: int = 0
        self.completed_cycles: list = []

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action.astype(np.float32)), \
            f"Invalid action: {action}"

        row = self.df.iloc[self.current_step]
        price = float(row["close"])

        # ── 1. 액션 → 4개 지정가 계산 ────────────────────────
        aggressiveness = float(action[0])
        profit_target = float(action[1])
        buy_hi, buy_lo, sell_lo, sell_hi = self._compute_order_prices(
            aggressiveness, profit_target, price
        )

        # ── 2. 다음 봉 체결 판단 ──────────────────────────────
        next_step = self.current_step + 1
        terminated = next_step >= len(self.df) - 1

        n_trades_this_step = 0
        cycle_bonus = 0.0
        equity_before = self._equity(price)

        if not terminated:
            next_row = self.df.iloc[next_step]
            next_high = float(next_row["high"])
            next_low = float(next_row["low"])
            next_price = float(next_row["close"])

            n_trades_this_step, cycle_bonus = self._process_fills(
                next_high, next_low, next_price,
                buy_hi, buy_lo, sell_lo, sell_hi,
            )
        else:
            next_price = price

        # ── 3. Reward 계산 ────────────────────────────────────
        equity_after = self._equity(next_price)
        step_reward = (
            (equity_after - equity_before) / self.start_capital
            - self.fee_rate * n_trades_this_step
            + cycle_bonus
        )

        self.n_trades += n_trades_this_step
        self.current_step = next_step

        obs = self._get_observation()
        info = {
            "equity": equity_after,
            "cash": self.cash,
            "holdings": self.holdings,
            "n_trades": self.n_trades,
            "buy_hi": buy_hi,
            "buy_lo": buy_lo,
            "sell_lo": sell_lo,
            "sell_hi": sell_hi,
        }

        return obs, float(step_reward), terminated, False, info

    # ──────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────

    def _init_state(self):
        """내부 상태 변수 초기화 (타입 힌트용)."""
        self.current_step: int = self.warmup
        self.cash: float = self.start_capital
        self.holdings: float = 0.0
        self.avg_price: float = 0.0
        self.in_cycle: bool = False
        self.cycle_start_cash: float = self.start_capital
        self.cycle_start_step: int = self.warmup
        self.n_trades: int = 0
        self.completed_cycles: list = []

    def _equity(self, price: float) -> float:
        """현재 포트폴리오 총 가치 (현금 + 보유 BTC 평가액)."""
        return self.cash + self.holdings * price

    def _compute_order_prices(
        self,
        aggressiveness: float,
        profit_target: float,
        price: float,
    ) -> tuple[float, float, float, float]:
        """액션값 → 4개 지정가 가격 반환."""
        buy_hi_gap  = 0.0001 + aggressiveness * 0.05   # [0.01%,  5%]
        buy_lo_gap  = 0.001  + aggressiveness * 0.10   # [0.10%, 10%]
        sell_lo_gap = 0.0001 + profit_target  * 0.05   # [0.01%,  5%]
        sell_hi_gap = 0.001  + profit_target  * 0.15   # [0.10%, 15%]

        buy_hi  = price * (1.0 - buy_hi_gap)
        buy_lo  = price * (1.0 - buy_lo_gap)
        sell_lo = price * (1.0 + sell_lo_gap)

        ref = self.avg_price if self.avg_price > 0.0 else price
        sell_hi = ref * (1.0 + sell_hi_gap)

        return buy_hi, buy_lo, sell_lo, sell_hi

    def _process_fills(
        self,
        next_high: float,
        next_low: float,
        next_price: float,
        buy_hi: float,
        buy_lo: float,
        sell_lo: float,
        sell_hi: float,
    ) -> tuple[int, float]:
        """
        다음 봉 고/저가로 체결 여부 판단 후 포트폴리오 업데이트.

        원칙: sell 먼저, 그 다음 buy.
        같은 봉에서 buy + sell 동시 체결 가능.

        Returns:
            (n_trades, cycle_bonus)
        """
        n_trades = 0
        cycle_bonus = 0.0

        # ── 1. SELL 먼저 처리 (보유 포지션 우선 정리) ──────────────

        # sell_lo: 빠른 익절 (현재가 근처)
        if self.holdings > 0.0 and next_high >= sell_lo:
            sell_qty = self.holdings * self.order_size_frac
            if sell_qty > 0.0:
                bonus = self._execute_sell(sell_lo, sell_qty, next_price)
                cycle_bonus += bonus
                n_trades += 1

        # sell_hi: 느린 익절 (평단 기준, 더 높은 목표)
        if self.holdings > 0.0 and next_high >= sell_hi:
            sell_qty = self.holdings * self.order_size_frac
            if sell_qty > 0.0:
                bonus = self._execute_sell(sell_hi, sell_qty, next_price)
                cycle_bonus += bonus
                n_trades += 1

        # ── 2. BUY 처리 ─────────────────────────────────────────

        # buy_hi: 공격적 매수 (현재가에 가까운 주문)
        if next_low <= buy_hi:
            success = self._execute_buy(buy_hi)
            if success:
                n_trades += 1

        # buy_lo: 보수적 매수 (현재가에서 더 낮은 주문)
        if next_low <= buy_lo:
            success = self._execute_buy(buy_lo)
            if success:
                n_trades += 1

        return n_trades, cycle_bonus

    def _execute_buy(self, fill_price: float) -> bool:
        """
        매수 체결 처리.

        Args:
            fill_price: 체결가
        Returns:
            체결 성공 여부 (현금 부족 시 False)
        """
        spend = self.cash * self.order_size_frac
        if spend < 1.0:          # 최소 주문 금액 ($1) 미만이면 스킵
            return False

        fee = spend * self.fee_rate
        net_spend = spend - fee  # 수수료 차감 후 실제 매수에 쓰이는 금액
        buy_qty = net_spend / fill_price

        # 사이클 시작 감지: 미보유 → 첫 매수
        if self.holdings == 0.0 and not self.in_cycle:
            self.in_cycle = True
            self.cycle_start_cash = self.cash
            self.cycle_start_step = self.current_step

        # 평단가 가중평균 업데이트
        total_cost = self.avg_price * self.holdings + fill_price * buy_qty
        self.holdings += buy_qty
        self.avg_price = total_cost / self.holdings

        self.cash -= spend
        return True

    def _execute_sell(self, fill_price: float, qty: float, next_price: float) -> float:
        """
        매도 체결 처리.

        Args:
            fill_price: 체결가
            qty:        매도 수량
            next_price: 사이클 종료 시 _close_cycle에 전달할 현재가
        Returns:
            cycle_bonus (사이클 종료 시 양수, 아니면 0.0)
        """
        qty = min(qty, self.holdings)   # 보유량 초과 방지
        if qty <= 0.0:
            return 0.0

        proceeds = qty * fill_price
        fee = proceeds * self.fee_rate
        self.cash += proceeds - fee
        self.holdings -= qty

        # 수량이 매우 작으면 0으로 강제 (부동소수점 오차 방지)
        if self.holdings < 1e-10:
            self.holdings = 0.0
            self.avg_price = 0.0

        # 사이클 종료 감지: 전량 청산 → holdings == 0
        if self.holdings == 0.0 and self.in_cycle:
            return self._close_cycle(next_price)

        return 0.0

    def _close_cycle(self, price: float) -> float:
        """
        사이클 종료 시 보너스 계산.

        Returns:
            cycle_bonus = cycle_pnl_pct + alpha / cycle_hours
        """
        cycle_pnl_pct = (self.cash - self.cycle_start_cash) / self.cycle_start_cash
        cycle_hours   = max(self.current_step - self.cycle_start_step, 1)
        bonus         = cycle_pnl_pct + self.cycle_alpha / cycle_hours

        self.completed_cycles.append({
            "start_step":    self.cycle_start_step,
            "end_step":      self.current_step,
            "cycle_hours":   cycle_hours,
            "pnl_pct":       cycle_pnl_pct,
            "bonus":         bonus,
        })

        self.in_cycle         = False
        self.cycle_start_cash = self.cash  # 다음 사이클 기준점 갱신
        return bonus

    def _get_observation(self) -> np.ndarray:
        """
        현재 스텝의 5차원 state 벡터 반환.

        Returns:
            np.ndarray shape (5,) float32, z-score 정규화 완료
        """
        row   = self.df.iloc[self.current_step]
        price = float(row["close"])

        # [0] 시장 상태: log(price / rolling_mean_168), rolling z-score 적용
        zscore_log_price = float(row["zscore_log_price"])

        # [1] 포지션 손익 방향: 평단가 대비 현재가 괴리율
        #     미보유 시 0 (에이전트에게 포지션 없음을 알림)
        if self.holdings > 0.0 and self.avg_price > 0.0:
            divergence = (self.avg_price - price) / self.avg_price
        else:
            divergence = 0.0

        # [2] 포지션 규모: 현재 BTC 평가액 / 초기 자본
        #     가격이 오르면 같은 수량이어도 비율 증가 → 리스크 인식
        holdings_value_ratio = (self.holdings * price) / self.start_capital

        # [3] 현금 여력: 추가 매수 가능 자본 비율
        cash_ratio = self.cash / self.start_capital

        # [4] 시장 변동성: ATR(168) / price, rolling z-score 적용
        zscore_volatility = float(row["zscore_volatility"])

        obs = np.array([
            zscore_log_price,
            divergence,
            holdings_value_ratio,
            cash_ratio,
            zscore_volatility,
        ], dtype=np.float32)

        return np.clip(obs, -5.0, 5.0)
