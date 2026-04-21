"""
BTC 그리드 트레이딩 Gymnasium 환경

State (7차원, rolling z-score 정규화):
    [0] log_price            = log(price / price.rolling(168).mean())
    [1] divergence           = (avg_price - price) / avg_price  (미보유 시 last_avg_price 활용, 거래 이력 없으면 0)
    [2] holdings_value_ratio = (holdings × price) / start_capital  (미보유 시 0)
    [3] cash_ratio           = cash / start_capital
    [4] volatility           = ATR(168) / price
    [5] trend_1d             = pct_change(24)  — 24h 단기 방향성
    [6] trend_1w             = pct_change(168) — 168h 주간 방향성

    trend 피처 추가 이유:
        방향성 피처 없이는 RL이 상승/하락/횡보 regime을 구분할 수 없어 [0,0] 포화 수렴.

Action (4차원 연속, [0, 1]⁴) — exp024 재설계:
    ATR을 공식에서 제거. RL이 절대 비율 gap을 직접 결정.
    state[4] volatility(ATR/price)는 유지 — RL이 필요하면 간접 활용 가능.

    [0] buy_hi_coef   → buy_hi_gap = action[0] × 0.10          [0, 10%]
    [1] buy_lo_extra  → buy_lo_gap = buy_hi_gap + action[1] × 0.20  [buy_hi_gap, buy_hi_gap+20%]
                        (항상 buy_lo < buy_hi 보장)
    [2] sell_m_coef   → sell_market_gap = action[2] × 0.10     [0, 10%]  현재가 기준
    [3] sell_c_coef   → sell_cost_gap   = action[3] × 0.20     [0, 20%]  평단가 기준 (독립)

    ATR 고정 최적값(exp023)의 대응 절대값 (BTC ATR/price ≈ 0.2%~2%):
        A_b=0.106 → buy_hi_gap ≈ 0.02%~0.21% → action[0] ≈ 0.002~0.021 (범위 내)
        C_b=13.92 → buy_lo_gap ≈ 2.8%~27.8%  → action[1] 범위 내
        A_s=0.080 → sell_m_gap ≈ 0.02%~0.16% → action[2] ≈ 0.002~0.016 (범위 내)
        C_s=4.309 → sell_c_gap ≈ 0.86%~8.6%  → action[3] ≈ 0.043~0.43 (범위 내)

주문 (매 스텝 4개 지정가 갱신 — ATR 없음, 절대 비율):
    buy_hi_gap      = action[0] × 0.10
    buy_lo_gap      = buy_hi_gap + action[1] × 0.20

    sell_market_gap = action[2] × 0.10
    sell_cost_gap   = action[3] × 0.20

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
    매수: n_buy_orders개 가격 레벨 (buy_hi ~ buy_lo 선형 보간)
          per_order_size = cycle_slot_size / n_buy_orders
          각 레벨에서 next_low ≤ level_price 조건 충족 시 체결
          슬롯 소진(cycle_budget_remaining < per_order_size) 후 추가 매수 완전 차단
    매도: 고정 2레벨 (n_sell_orders 파라미터 없음)
          sell_market: 현재가 기준 단기 반등 / 현금 확보
          sell_cost  : 평단가 기준 수익 실현
          threshold_btc = cycle_slot_size / price  (현재가 기준 1슬롯 BTC 수량)
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
        self.n_splits: int         = self.cfg_env["n_splits"]
        # threshold_basis: "price"(현재가) or "avg_price"(평단가)
        # 미지정 시 기존 동작("price") 유지 → 하위 호환
        self.threshold_basis: str  = self.cfg_env.get("threshold_basis", "price")
        self.warmup: int           = self.cfg_ind["atr_period"]  # 168봉

        # ── MDP 공식 계수 (Bayesian 튜닝 가능, 기본값 = 설계 기본값) ──
        # buy_hi_gap  = atr_ratio × (A_b + aggressiveness × B_b)
        # buy_lo_gap  = atr_ratio × (C_b + aggressiveness × D_b)
        # sell_market_gap = atr_ratio × (A_s + profit_target × B_s)
        # sell_cost_gap   = atr_ratio × (C_s + profit_target × D_s)
        # buy side: Bayesian Trial #42 참고값 (config 재정의 가능)
        # sell side: 재설계값 — A_s/B_s는 Bayesian 대상에서 제외 (RL action 충돌 방지)
        _coefs = self.cfg_env.get("formula_coefs", {})
        self.A_b: float = float(_coefs.get("A_b", 0.285))
        self.B_b: float = float(_coefs.get("B_b", 1.748))
        self.C_b: float = float(_coefs.get("C_b", 5.223))
        self.D_b: float = float(_coefs.get("D_b", 18.683))
        self.A_s: float = float(_coefs.get("A_s", 0.05))   # 재설계: 0.5 → 0.05
        self.B_s: float = float(_coefs.get("B_s", 1.95))   # 재설계: 1.5 → 1.95
        self.C_s: float = float(_coefs.get("C_s", 2.5))
        self.D_s: float = float(_coefs.get("D_s", 7.5))

        # ── 랜덤 시작점 (멀티 환경 + 짧은 에피소드용) ──────────
        # random_start=True: reset() 시 훈련 데이터 전체에서 무작위 시작
        # max_episode_steps: 시작 가능 범위를 df 끝에서 역산 (TimeLimit과 연동)
        self.random_start: bool = self.cfg_env.get("random_start", False)
        self._max_ep_steps: int | None = config.get("training", {}).get(
            "max_episode_steps", None
        )

        # ── Gymnasium 공간 정의 ────────────────────────────────
        # Action: [aggressiveness, profit_target] ∈ [0, 1]²
        # aggressiveness → buy gap 계수 범위 선택
        # profit_target  → sell gap 계수 범위 선택
        self.action_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(4,),
            dtype=np.float32,
        )

        # Observation: z-score 정규화 후 대략 [-4, 4] 범위
        # shape=(7,): log_price, divergence, holdings_value_ratio, cash_ratio,
        #             volatility, trend_1d, trend_1w
        self.observation_space = spaces.Box(
            low=np.float32(-5.0),
            high=np.float32(5.0),
            shape=(7,),
            dtype=np.float32,
        )

        # 내부 상태 초기화
        self._init_state()

    # ──────────────────────────────────────────────────────────
    # Gymnasium 인터페이스
    # ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ── 시작 스텝 결정 ──────────────────────────────────────
        if self.random_start:
            # 유효 시작 범위: [warmup, len(df) - ep_len - 2]
            # TimeLimit 래퍼가 ep_len 이후 truncate하므로 끝에서 여유를 둠
            ep_len = self._max_ep_steps or (len(self.df) - self.warmup - 1)
            max_start = len(self.df) - ep_len - 2
            if max_start > self.warmup:
                self.current_step = int(
                    self.np_random.integers(self.warmup, max_start)
                )
            else:
                self.current_step = self.warmup
        else:
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
        price = float(row["close"])

        # ── 1. 액션 → 4개 지정가 계산 ────────────────────────
        buy_hi_coef  = float(action[0])
        buy_lo_extra = float(action[1])
        sell_m_coef  = float(action[2])
        sell_c_coef  = float(action[3])
        buy_hi, buy_lo, sell_market, sell_cost = self._compute_order_prices(
            buy_hi_coef, buy_lo_extra, sell_m_coef, sell_c_coef, price
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
            "buy_hi_coef": buy_hi_coef,
            "buy_lo_extra": buy_lo_extra,
            "sell_m_coef": sell_m_coef,
            "sell_c_coef": sell_c_coef,
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
        buy_hi_coef: float,
        buy_lo_extra: float,
        sell_m_coef: float,
        sell_c_coef: float,
        price: float,
    ) -> tuple[float, float, float, float]:
        """4D action → 4개 지정가 반환 (exp024: ATR 없음, 절대 비율).

        Args:
            buy_hi_coef:  action[0] ∈ [0,1] → buy_hi_gap = coef × 0.10   [0%, 10%]
            buy_lo_extra: action[1] ∈ [0,1] → buy_lo_gap = buy_hi_gap + extra × 0.20
                          (항상 buy_lo < buy_hi 보장)
            sell_m_coef:  action[2] ∈ [0,1] → sell_market_gap = coef × 0.10  [0%, 10%]
            sell_c_coef:  action[3] ∈ [0,1] → sell_cost_gap   = coef × 0.20  [0%, 20%]
            price:        현재 봉 close

        Returns:
            (buy_hi, buy_lo, sell_market, sell_cost)
        """
        buy_hi_gap      = buy_hi_coef  * 0.10
        buy_lo_gap      = buy_hi_gap   + buy_lo_extra * 0.20
        sell_market_gap = sell_m_coef  * 0.10
        sell_cost_gap   = sell_c_coef  * 0.20

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
            buy_hi:      매수 가격 레벨 상단 (가장 공격적, 체결 확률 높음)
            buy_lo:      매수 가격 레벨 하단 (가장 보수적, 체결 확률 낮음)
            sell_market: 현재가 기준 매도 지정가
            sell_cost:   평단가 기준 매도 지정가

        Returns:
            (n_trades, cycle_bonus)
        """
        n_trades = 0
        cycle_bonus = 0.0

        # ── 1. SELL 먼저 처리 (보유 포지션 우선 정리) ──────────────

        # threshold_btc: 1슬롯에 해당하는 BTC 수량
        #   holdings ≤ threshold_btc → 전량 청산 (사이클 종료 유도)
        #   holdings >  threshold_btc → holdings / n_splits (매수와 대칭 균등 분할)
        if self.threshold_basis == "avg_price":
            ref_price = (self.avg_price if self.avg_price > 0.0 else price)
        else:
            ref_price = price
        threshold_btc = (self.cycle_slot_size / ref_price
                         if ref_price > 0.0 else 0.0)

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

        # buy_hi ~ buy_lo 사이를 n_buy_orders개 가격으로 선형 보간
        # n_buy_orders=2 → [buy_hi, buy_lo] (하위 호환)
        if self.n_buy_orders == 1:
            buy_prices = [buy_hi]
        else:
            buy_prices = [
                buy_hi + i / (self.n_buy_orders - 1) * (buy_lo - buy_hi)
                for i in range(self.n_buy_orders)
            ]

        for bp in buy_prices:
            if next_low <= bp:
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
        # entry_gate는 _process_fills에서 이미 검사 완료 → 여기서는 전액 배정
        if self.holdings == 0.0 and not self.in_cycle:
            self.in_cycle = True
            self.cycle_start_cash = self.cash
            self.cycle_start_step = self.current_step
            # 전액 투입: cash 전체를 n_splits 슬롯으로 분할
            self.cycle_slot_size        = self.cash / self.n_splits
            self.per_order_size         = self.cycle_slot_size / self.n_buy_orders
            self.cycle_budget_remaining = self.cash

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
        현재 스텝의 7차원 state 벡터 반환.

        Returns:
            np.ndarray shape (7,) float32, z-score 정규화 완료
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
            divergence = (self.last_avg_price - price) / self.last_avg_price
        else:
            divergence = 0.0

        # [2] 포지션 규모: 현재 BTC 평가액 / 초기 자본
        holdings_value_ratio = (self.holdings * price) / self.start_capital

        # [3] 현금 여력: 추가 매수 가능 자본 비율
        cash_ratio = self.cash / self.start_capital

        # [4] 시장 변동성: ATR(168) / price, rolling z-score 적용
        zscore_volatility = float(row["zscore_volatility"])

        # [5] 단기 방향성: 24h 수익률 rolling z-score
        zscore_trend_1d = float(row["zscore_trend_1d"])

        # [6] 주간 방향성: 168h 수익률 rolling z-score
        zscore_trend_1w = float(row["zscore_trend_1w"])

        obs = np.array([
            zscore_log_price,
            divergence,
            holdings_value_ratio,
            cash_ratio,
            zscore_volatility,
            zscore_trend_1d,
            zscore_trend_1w,
        ], dtype=np.float32)

        return np.clip(obs, -5.0, 5.0)
