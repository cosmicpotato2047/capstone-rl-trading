"""
BTC 그리드 트레이딩 Gymnasium 환경

State (5차원, rolling z-score 정규화):
    [0] log_price            = log(price / price.rolling(168).mean())
    [1] divergence           = (avg_price - price) / avg_price  (미보유 시 last_avg_price 활용, 거래 이력 없으면 0)
    [2] holdings_value_ratio = (holdings × price) / start_capital  (미보유 시 0)
    [3] cash_ratio           = cash / start_capital
    [4] volatility           = ATR(168) / price

Action (2차원 연속, [0, 1]²):
    [0] aggressiveness → buy_hi_gap, buy_lo_gap 결정
    [1] profit_target  → sell_market_gap, sell_cost_gap 결정

주문 (매 스텝 4개 지정가 갱신 — ATR 비례 스케일링):
    atr_ratio   = ATR(168) / price              # 현재 변동성 수준

    buy_hi_gap      = atr_ratio × (0.1 + aggressiveness × 0.9)  # [0.1×ATR, 1.0×ATR]
    buy_lo_gap      = atr_ratio × (0.5 + aggressiveness × 4.5)  # [0.5×ATR, 5.0×ATR]
    sell_market_gap = atr_ratio × (0.1 + profit_target  × 0.9)  # [0.1×ATR, 1.0×ATR]
    sell_cost_gap   = atr_ratio × (0.5 + profit_target  × 4.5)  # [0.5×ATR, 5.0×ATR]

    buy_hi      = price     × (1 - buy_hi_gap)
    buy_lo      = price     × (1 - buy_lo_gap)
    sell_market = price     × (1 + sell_market_gap)
    sell_cost   = avg_price × (1 + sell_cost_gap)    # 원가 수익 보호 (평단가 기준)

체결: 다음 봉 high/low 기준, 조건 충족 시 그 시점 시장가(next_high / next_low)로 체결
    next_high >= sell_market  →  next_high 가격으로 체결
    next_high >= sell_cost    →  next_high 가격으로 체결
    next_low  <= buy_hi       →  next_low  가격으로 체결
    next_low  <= buy_lo       →  next_low  가격으로 체결

주문 크기:
    매수: 사이클 시작 시 현금을 n_splits 슬롯으로 분할
          per_order_size = (cycle_start_cash / n_splits) / n_buy_orders
          슬롯 소진(cycle_budget_remaining < per_order_size) 후 추가 매수 완전 차단
    매도: threshold_btc = cycle_slot_size / price  (현재가 기준 1슬롯 BTC 수량)
          holdings ≤ threshold_btc → 전량 청산 (사이클 종료 유도)
          holdings > threshold_btc → holdings / n_splits (매수와 대칭 균등 분할)

Reward:
    매 스텝: (equity_t - equity_{t-1}) / start_capital
             수수료는 _execute_buy/_execute_sell에서 cash에 반영되므로 별도 차감 없음.
    사이클 종료 시: 보너스 없음 (통계 기록만 — completed_cycles)
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
        self.start_capital: float  = self.cfg_env["initial_cash"]
        self.fee_rate: float       = self.cfg_env["transaction_cost"]
        self.n_buy_orders: int     = self.cfg_env["n_buy_orders"]
        self.n_sell_orders: int    = self.cfg_env["n_sell_orders"]
        self.n_splits: int         = self.cfg_env["n_splits"]
        self.warmup: int           = self.cfg_ind["atr_period"]  # 168봉

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

        # 사이클 예산 (사이클 시작 시 확정, 0이면 매수 차단)
        self.cycle_slot_size: float        = 0.0
        self.per_order_size: float         = 0.0
        self.cycle_budget_remaining: float = 0.0

        # 직전 사이클 청산 시 평단가 (미보유 구간 divergence 계산용)
        self.last_avg_price: float = 0.0

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
        price     = float(row["close"])
        atr_ratio = float(row["volatility_raw"])   # ATR(168) / price

        # ── 1. 액션 → 4개 지정가 계산 ────────────────────────
        aggressiveness = float(action[0])
        profit_target = float(action[1])
        buy_hi, buy_lo, sell_market, sell_cost = self._compute_order_prices(
            aggressiveness, profit_target, price, atr_ratio
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
                price, next_high, next_low, next_price,
                buy_hi, buy_lo, sell_market, sell_cost,
            )
        else:
            next_price = price

        # ── 3. Reward 계산 ────────────────────────────────────
        # 수수료는 _execute_buy/_execute_sell에서 cash에 이미 반영됨:
        #   매수: cash -= spend (fee 포함),  BTC += (spend - fee) / price
        #   매도: cash += qty × price - fee
        # → equity 변화분 자체가 수수료를 포함하므로 별도 패널티 불필요.
        # (- fee_rate * n_trades 항은 단위 불일치로 수수료를 8배 중복 계산함)
        equity_after = self._equity(next_price)
        step_reward = (equity_after - equity_before) / self.start_capital

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
            "sell_market": sell_market,
            "sell_cost": sell_cost,
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
        # 사이클 예산 (사이클 시작 시 확정, 0이면 매수 차단)
        self.cycle_slot_size: float        = 0.0
        self.per_order_size: float         = 0.0
        self.cycle_budget_remaining: float = 0.0
        # 직전 사이클 청산 시 평단가 (미보유 구간 divergence 계산용)
        self.last_avg_price: float = 0.0
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
        atr_ratio: float,
    ) -> tuple[float, float, float, float]:
        """액션값 → 4개 지정가 가격 반환.

        ATR 비례 스케일링:
            간격을 현재 변동성(ATR/price)에 비례시켜 action 전 범위 [0,1]이
            실제 체결 확률 스펙트럼 [~13%, ~87%]에 고르게 대응되도록 설계.

            aggressiveness=0.0 → buy_hi_gap = 0.1 × ATR  (소극적 매수, 낮은 체결률)
            aggressiveness=1.0 → buy_hi_gap = 1.0 × ATR  (공격적 매수, 높은 체결률)

        Args:
            atr_ratio: ATR(168) / price (volatility_raw 컬럼)

        Returns:
            (buy_hi, buy_lo, sell_market, sell_cost)
            sell_market : 현재가(price) 기준 — 시장 모멘텀 활용
            sell_cost   : 평단가(avg_price) 기준 — 원가 수익 보호
        """
        # ATR 비례 간격 — 변동성이 크면 자동으로 간격이 넓어짐
        buy_hi_gap      = atr_ratio * (0.1 + aggressiveness * 0.9)  # [0.1×ATR, 1.0×ATR]
        buy_lo_gap      = atr_ratio * (0.5 + aggressiveness * 4.5)  # [0.5×ATR, 5.0×ATR]
        sell_market_gap = atr_ratio * (0.1 + profit_target  * 0.9)  # [0.1×ATR, 1.0×ATR]
        sell_cost_gap   = atr_ratio * (0.5 + profit_target  * 4.5)  # [0.5×ATR, 5.0×ATR]

        buy_hi      = price * (1.0 - buy_hi_gap)
        buy_lo      = price * (1.0 - buy_lo_gap)
        sell_market = price * (1.0 + sell_market_gap)

        ref       = self.avg_price if self.avg_price > 0.0 else price
        sell_cost = ref * (1.0 + sell_cost_gap)

        return buy_hi, buy_lo, sell_market, sell_cost

    def _process_fills(
        self,
        price: float,
        next_high: float,
        next_low: float,
        next_price: float,
        buy_hi: float,
        buy_lo: float,
        sell_market: float,
        sell_cost: float,
    ) -> tuple[int, float]:
        """
        다음 봉 고/저가로 체결 여부 판단 후 포트폴리오 업데이트.

        원칙: sell 먼저, 그 다음 buy.
        같은 봉에서 buy + sell 동시 체결 가능.

        체결가 (시장가 방식):
            매도 조건 충족 시 → next_high 가격으로 체결
            매수 조건 충족 시 → next_low  가격으로 체결

        Args:
            price:       현재 봉 close (threshold_btc 계산용)
            next_high:   다음 봉 high
            next_low:    다음 봉 low
            next_price:  다음 봉 close (사이클 종료 처리용)
            buy_hi:      공격적 매수 지정가
            buy_lo:      보수적 매수 지정가
            sell_market: 현재가 기준 매도 지정가
            sell_cost:   평단가 기준 매도 지정가

        Returns:
            (n_trades, cycle_bonus)
        """
        n_trades = 0
        cycle_bonus = 0.0

        # ── 1. SELL 먼저 처리 (보유 포지션 우선 정리) ──────────────

        # threshold_btc: 현재가 기준 1슬롯 BTC 수량
        #   holdings ≤ threshold_btc → 전량 청산 (사이클 종료 유도)
        #   holdings >  threshold_btc → holdings / n_splits (매수와 대칭 균등 분할)
        threshold_btc = (self.cycle_slot_size / price
                         if price > 0.0 else 0.0)

        # sell_market: 현재가 기준 매도 — 시장 모멘텀 활용
        if self.holdings > 0.0 and next_high >= sell_market:
            sell_qty = (self.holdings
                        if self.holdings <= threshold_btc
                        else self.holdings / self.n_splits)
            if sell_qty > 0.0:
                # 조건 충족 시 그 시점 시장가(next_high)로 체결
                bonus = self._execute_sell(next_high, sell_qty, next_price)
                cycle_bonus += bonus
                n_trades += 1

        # sell_cost: 평단가 기준 매도 — 원가 수익 보호
        if self.holdings > 0.0 and next_high >= sell_cost:
            sell_qty = (self.holdings
                        if self.holdings <= threshold_btc
                        else self.holdings / self.n_splits)
            if sell_qty > 0.0:
                # 조건 충족 시 그 시점 시장가(next_high)로 체결
                bonus = self._execute_sell(next_high, sell_qty, next_price)
                cycle_bonus += bonus
                n_trades += 1

        # ── 2. BUY 처리 ─────────────────────────────────────────

        # buy_hi: 공격적 매수 — 조건 충족 시 next_low로 체결 (지정가 이하 보장)
        if next_low <= buy_hi:
            success = self._execute_buy(next_low)
            if success:
                n_trades += 1

        # buy_lo: 보수적 매수 — 조건 충족 시 next_low로 체결
        if next_low <= buy_lo:
            success = self._execute_buy(next_low)
            if success:
                n_trades += 1

        return n_trades, cycle_bonus

    def _execute_buy(self, fill_price: float) -> bool:
        """
        매수 체결 처리.

        사이클 최초 매수 시 cycle_budget_remaining / per_order_size를 확정한 뒤
        매번 고정 금액(per_order_size)을 소비한다.
        cycle_budget_remaining < per_order_size 이면 추가 매수를 완전 차단한다.

        Args:
            fill_price: 체결가
        Returns:
            체결 성공 여부 (예산 소진 또는 현금 부족 시 False)
        """
        # 사이클 시작 감지: 미보유 → 첫 매수 시 예산 확정
        if self.holdings == 0.0 and not self.in_cycle:
            self.in_cycle = True
            self.cycle_start_cash          = self.cash
            self.cycle_start_step          = self.current_step
            self.cycle_slot_size           = self.cash / self.n_splits
            self.per_order_size            = self.cycle_slot_size / self.n_buy_orders
            self.cycle_budget_remaining    = self.cash

        # 예산 소진 체크 → 완전 차단
        if self.cycle_budget_remaining < self.per_order_size:
            return False

        spend = self.per_order_size
        if spend > self.cash:    # 부동소수점 오차 안전망
            return False

        fee       = spend * self.fee_rate
        net_spend = spend - fee  # 수수료 차감 후 실제 매수에 쓰이는 금액
        buy_qty   = net_spend / fill_price

        # 평단가 가중평균 업데이트
        total_cost     = self.avg_price * self.holdings + fill_price * buy_qty
        self.holdings  += buy_qty
        self.avg_price  = total_cost / self.holdings

        self.cash                   -= spend
        self.cycle_budget_remaining -= spend
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
            self.last_avg_price = self.avg_price   # 전량 청산 직전 평단가 보존
            self.holdings  = 0.0
            self.avg_price = 0.0

        # 사이클 종료 감지: 전량 청산 → holdings == 0
        if self.holdings == 0.0 and self.in_cycle:
            return self._close_cycle(next_price)

        return 0.0

    def _close_cycle(self, price: float) -> float:
        """
        사이클 종료 시 통계 기록.

        보너스는 step_reward의 equity 변화에 이미 반영되므로 반환하지 않는다.

        Returns:
            0.0 (보너스 없음)
        """
        cycle_pnl_pct = (self.cash - self.cycle_start_cash) / self.cycle_start_cash
        cycle_hours   = max(self.current_step - self.cycle_start_step, 1)

        self.completed_cycles.append({
            "start_step":    self.cycle_start_step,
            "end_step":      self.current_step,
            "cycle_hours":   cycle_hours,
            "pnl_pct":       cycle_pnl_pct,
        })

        self.in_cycle         = False
        self.cycle_start_cash = self.cash  # 다음 사이클 기준점 갱신
        return 0.0

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
        #     보유 중: 현재 평단가 기준
        #     미보유 + 직전 사이클 있음: last_avg_price 기준 (재진입 신호 유지)
        #     미보유 + 거래 이력 없음: 0.0 (에피소드 초반)
        if self.holdings > 0.0 and self.avg_price > 0.0:
            divergence = (self.avg_price - price) / self.avg_price
        elif self.last_avg_price > 0.0:
            # 미보유 구간: 직전 사이클 평단가 기준 괴리율 유지
            divergence = (self.last_avg_price - price) / self.last_avg_price
        else:
            divergence = 0.0  # 거래 이력 없음 (에피소드 초반)

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
