"""
tests/test_trading_env.py

BTCGridTradingEnv 포트폴리오 수학 검증 테스트.

검증 항목:
  1.  초기 상태 (reset)
  2.  매수 수수료 및 평단가 계산
  3.  n_splits 예산 분할 — per_order_size / cycle_slot_size
  4.  예산 소진 후 매수 완전 차단
  5.  매도 수수료 및 보유량 변화
  6.  전량 청산 시 last_avg_price 보존
  7.  사이클 종료 — completed_cycles 구조 (bonus 키 없음)
  8.  threshold_btc 기준 전량 / 분할 매도
  9.  SELL 우선 원칙 (같은 봉에서 buy + sell 동시 체결)
  10. 미보유 구간 divergence — last_avg_price 활용
  11. Reward = equity_change / start_capital - fee_rate × n_trades
  12. gymnasium env_checker 통과
"""

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from src.env.trading_env import BTCGridTradingEnv


# ─────────────────────────────────────────────────────────────────────────────
# 공통 상수 & 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

WARMUP        = 5        # 테스트용 소형 warmup (atr_period=5)
INITIAL_CASH  = 10_000.0
FEE           = 0.0005   # 0.05%
N_SPLITS      = 4
N_BUY         = 2
N_SELL        = 2
PRICE         = 1_000.0  # 기본 테스트 가격


def _cfg(**env_overrides):
    """테스트용 최소 config dict."""
    cfg = {
        "environment": {
            "initial_cash":     INITIAL_CASH,
            "transaction_cost": FEE,
            "n_buy_orders":     N_BUY,
            "n_sell_orders":    N_SELL,
            "n_splits":         N_SPLITS,
        },
        "indicators": {"atr_period": WARMUP},
    }
    cfg["environment"].update(env_overrides)
    return cfg


def _df(n_rows=30, price=PRICE, high_offset=0.0, low_offset=0.0):
    """
    close = price 고정.
    high = price + high_offset, low = price - low_offset.

    기본값(offset=0): high == low == price → 어떤 체결도 발생하지 않는 안전한 봉.
      sell_lo = price × 1.0001 > high=price  → sell 안 됨
      buy_hi  = price × 0.9999 < low=price   → buy  안 됨
    """
    return pd.DataFrame({
        "open":              [price] * n_rows,
        "high":              [price + high_offset] * n_rows,
        "low":               [price - low_offset]  * n_rows,
        "close":             [price] * n_rows,
        "volume":            [100.0] * n_rows,
        "log_price":         [0.0]   * n_rows,
        "atr":               [10.0]  * n_rows,
        "zscore_log_price":  [0.0]   * n_rows,
        "zscore_volatility": [0.0]   * n_rows,
    })


def _df_rows(prices, highs, lows):
    """봉별 가격을 정밀 제어하는 DataFrame."""
    n = len(prices)
    return pd.DataFrame({
        "open":              prices,
        "high":              highs,
        "low":               lows,
        "close":             prices,
        "volume":            [100.0] * n,
        "log_price":         [0.0]   * n,
        "atr":               [10.0]  * n,
        "zscore_log_price":  [0.0]   * n,
        "zscore_volatility": [0.0]   * n,
    })


def _env(df=None, **env_overrides):
    """env 생성 + reset."""
    if df is None:
        df = _df()
    env = BTCGridTradingEnv(df, _cfg(**env_overrides))
    env.reset(seed=0)
    return env


def _act(aggressiveness=0.0, profit_target=0.0):
    return np.array([aggressiveness, profit_target], dtype=np.float32)


def _buy(env, fill_price=PRICE):
    """내부 메서드 직접 호출 (step 없이 포트폴리오 수학만 검증)."""
    env.current_step = WARMUP
    return env._execute_buy(fill_price)


def _sell(env, fill_price, qty=None):
    qty = qty if qty is not None else env.holdings
    env._execute_sell(fill_price, qty, fill_price)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 초기 상태
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialState:
    def test_cash(self):
        assert _env().cash == INITIAL_CASH

    def test_holdings_zero(self):
        assert _env().holdings == 0.0

    def test_avg_price_zero(self):
        assert _env().avg_price == 0.0

    def test_in_cycle_false(self):
        assert _env().in_cycle is False

    def test_last_avg_price_zero(self):
        assert _env().last_avg_price == 0.0

    def test_completed_cycles_empty(self):
        assert _env().completed_cycles == []

    def test_n_trades_zero(self):
        assert _env().n_trades == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. 매수 수학 — per_order_size / 평단가 / 현금
# ─────────────────────────────────────────────────────────────────────────────

class TestBuyMath:
    def test_per_order_size(self):
        """per_order_size = (initial_cash / n_splits) / n_buy_orders = 1250."""
        env = _env()
        _buy(env, 1000.0)
        assert env.per_order_size == pytest.approx(INITIAL_CASH / N_SPLITS / N_BUY)

    def test_cycle_slot_size(self):
        """cycle_slot_size = initial_cash / n_splits = 2500."""
        env = _env()
        _buy(env, 1000.0)
        assert env.cycle_slot_size == pytest.approx(INITIAL_CASH / N_SPLITS)

    def test_avg_price_equals_fill_price_first_buy(self):
        """첫 매수: 이전 포지션 없으므로 avg_price == fill_price."""
        env = _env()
        _buy(env, 1000.0)
        assert env.avg_price == pytest.approx(1000.0)

    def test_avg_price_weighted_average_two_buys(self):
        """
        두 번 매수 후 avg_price = 수량 가중 평균.
        1차: fill=1000, spend=1250, net=1249.375, qty1 = 1249.375 / 1000
        2차: fill= 900, spend=1250, net=1249.375, qty2 = 1249.375 /  900
        avg = (qty1×1000 + qty2×900) / (qty1+qty2)
        """
        env = _env()
        _buy(env, 1000.0)
        _buy(env, 900.0)

        net = (INITIAL_CASH / N_SPLITS / N_BUY) * (1 - FEE)
        qty1 = net / 1000.0
        qty2 = net / 900.0
        expected = (qty1 * 1000.0 + qty2 * 900.0) / (qty1 + qty2)
        assert env.avg_price == pytest.approx(expected, rel=1e-6)

    def test_cash_decreases_by_per_order_size(self):
        """매수 후 cash = initial_cash - per_order_size."""
        env = _env()
        _buy(env, 1000.0)
        per_order = INITIAL_CASH / N_SPLITS / N_BUY
        assert env.cash == pytest.approx(INITIAL_CASH - per_order)

    def test_holdings_net_of_fee(self):
        """매수 수량 = per_order_size × (1 - fee) / fill_price."""
        env = _env()
        _buy(env, 1000.0)
        per_order = INITIAL_CASH / N_SPLITS / N_BUY
        expected_qty = per_order * (1 - FEE) / 1000.0
        assert env.holdings == pytest.approx(expected_qty, rel=1e-6)

    def test_in_cycle_set_on_first_buy(self):
        env = _env()
        _buy(env, 1000.0)
        assert env.in_cycle is True

    def test_cycle_start_cash_set_to_cash_at_first_buy(self):
        """cycle_start_cash = 첫 매수 직전의 cash."""
        env = _env()
        cash_before = env.cash
        _buy(env, 1000.0)
        assert env.cycle_start_cash == pytest.approx(cash_before)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 예산 소진 차단
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetExhaustion:
    def test_max_buys_all_succeed(self):
        """총 n_splits × n_buy_orders = 8번 매수가 모두 성공해야 한다."""
        env = _env()
        for _ in range(N_SPLITS * N_BUY):
            assert _buy(env, 1000.0) is True

    def test_buy_after_exhaustion_returns_false(self):
        """8번 매수 후 9번째는 False 반환."""
        env = _env()
        for _ in range(N_SPLITS * N_BUY):
            _buy(env, 1000.0)
        assert _buy(env, 1000.0) is False

    def test_budget_remaining_never_negative(self):
        """cycle_budget_remaining은 절대 음수가 되면 안 된다."""
        env = _env()
        for _ in range(N_SPLITS * N_BUY + 5):
            _buy(env, 1000.0)
        assert env.cycle_budget_remaining >= 0.0

    def test_cash_never_negative(self):
        """cash는 절대 음수가 되면 안 된다."""
        env = _env()
        for _ in range(N_SPLITS * N_BUY + 5):
            _buy(env, 1000.0)
        assert env.cash >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. 매도 수학 — 수수료 / 보유량 / avg_price 불변
# ─────────────────────────────────────────────────────────────────────────────

class TestSellMath:
    def setup_method(self):
        self.env = _env()
        _buy(self.env, 1000.0)

    def test_cash_increases_net_of_fee(self):
        """cash += qty × fill_price × (1 - fee)."""
        cash_before = self.env.cash
        qty = self.env.holdings
        sell_price = 1100.0
        _sell(self.env, sell_price, qty)
        expected = cash_before + qty * sell_price * (1 - FEE)
        assert self.env.cash == pytest.approx(expected, rel=1e-6)

    def test_holdings_zero_after_full_sell(self):
        _sell(self.env, 1100.0)
        assert self.env.holdings == 0.0

    def test_avg_price_zero_after_full_sell(self):
        _sell(self.env, 1100.0)
        assert self.env.avg_price == 0.0

    def test_avg_price_unchanged_after_partial_sell(self):
        """분할 매도 시 avg_price는 변하지 않는다."""
        _buy(self.env, 900.0)   # 두 번째 매수 → avg_price 변동
        avg_before = self.env.avg_price
        _sell(self.env, 1100.0, qty=self.env.holdings / 2)
        assert self.env.avg_price == pytest.approx(avg_before, rel=1e-9)

    def test_partial_sell_reduces_holdings(self):
        holdings_before = self.env.holdings
        _sell(self.env, 1100.0, qty=holdings_before / 2)
        assert self.env.holdings == pytest.approx(holdings_before / 2, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 5. last_avg_price & 사이클 종료
# ─────────────────────────────────────────────────────────────────────────────

class TestCycleAndLastAvgPrice:
    def test_last_avg_price_stored_on_full_liquidation(self):
        """전량 청산 시 last_avg_price = 청산 직전 avg_price."""
        env = _env()
        _buy(env, 1000.0)
        avg_before = env.avg_price
        _sell(env, 1100.0)
        assert env.last_avg_price == pytest.approx(avg_before)

    def test_last_avg_price_zero_before_any_sell(self):
        env = _env()
        _buy(env, 1000.0)
        assert env.last_avg_price == 0.0

    def test_last_avg_price_unchanged_after_partial_sell(self):
        """전량이 아닌 분할 매도 시 last_avg_price는 변하지 않는다."""
        env = _env()
        _buy(env, 1000.0)
        _sell(env, 1100.0, qty=env.holdings / 2)
        assert env.last_avg_price == 0.0

    def test_in_cycle_false_after_full_sell(self):
        env = _env()
        _buy(env, 1000.0)
        _sell(env, 1100.0)
        assert env.in_cycle is False

    def test_completed_cycles_count(self):
        """매수 → 전량 매도 1회 = completed_cycles 1개."""
        env = _env()
        _buy(env, 1000.0)
        _sell(env, 1100.0)
        assert len(env.completed_cycles) == 1

    def test_completed_cycles_no_bonus_key(self):
        """사이클 보너스 제거 확인 — 'bonus' 키가 없어야 한다."""
        env = _env()
        _buy(env, 1000.0)
        _sell(env, 1100.0)
        assert "bonus" not in env.completed_cycles[0]

    def test_completed_cycles_required_keys(self):
        env = _env()
        _buy(env, 1000.0)
        _sell(env, 1100.0)
        cycle = env.completed_cycles[0]
        for key in ("start_step", "end_step", "cycle_hours", "pnl_pct"):
            assert key in cycle

    def test_cycle_pnl_pct_positive_on_profit(self):
        """익절 사이클: pnl_pct > 0."""
        env = _env()
        _buy(env, 1000.0)
        _sell(env, 1100.0)
        assert env.completed_cycles[0]["pnl_pct"] > 0.0

    def test_cycle_pnl_pct_negative_on_loss(self):
        """손절 사이클: pnl_pct < 0 (수수료 포함)."""
        env = _env()
        _buy(env, 1000.0)
        _sell(env, 500.0)   # 손절
        assert env.completed_cycles[0]["pnl_pct"] < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. threshold_btc — 전량 vs 분할 매도
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholdBtcSell:
    """
    threshold_btc = cycle_slot_size / avg(sell_lo, sell_hi)
    holdings ≤ threshold_btc → 전량 청산
    holdings >  threshold_btc → holdings / n_sell_orders (균등 분할)

    _process_fills 직접 호출로 검증.
    """

    SELL_LO = 1050.0
    SELL_HI = 1100.0
    AVG_SELL = (SELL_LO + SELL_HI) / 2   # 1075.0
    # cycle_slot_size = 10000/4 = 2500
    # threshold_btc = 2500 / 1075 ≈ 2.326

    def _env_with_n_buys(self, n_buys):
        env = _env()
        for _ in range(n_buys):
            _buy(env, PRICE)
        return env

    def test_full_sell_when_below_threshold(self):
        """
        1번 매수 → holdings ≈ 1.249 < threshold ≈ 2.326 → 전량 청산.
        buy 는 차단 (next_low > buy_hi).
        """
        env = self._env_with_n_buys(1)
        assert env.holdings < INITIAL_CASH / N_SPLITS / self.AVG_SELL

        env._process_fills(
            next_high=2000.0,   # sell_lo(1050) 체결
            next_low=999.5,     # buy_hi(≈999.9) 미체결 (999.5 <= 999.9 → 체결됨!)
            # ↑ 체결 방지: next_low > buy_hi 필요. buy_hi = 1000*(1-0.0001) = 999.9
            # 999.5 < 999.9 이므로 buy_hi도 체결됨. 그러나 sell이 먼저.
            # sell 후 holdings=0 → 사이클 종료 후 buy → 새 사이클 시작
            next_price=self.SELL_LO,
            buy_hi=999.9,
            buy_lo=899.0,
            sell_lo=self.SELL_LO,
            sell_hi=self.SELL_HI,
        )
        # 전량 청산 후 사이클 종료 확인
        assert len(env.completed_cycles) == 1

    def test_partial_sell_when_above_threshold(self):
        """
        8번 매수 → holdings ≈ 9.995 > threshold ≈ 2.326 → 분할 매도.
        sell_lo만 체결(next_high=1060: sell_lo=1050 체결, sell_hi=1100 미체결).
        sell 후 holdings = 원래 - holdings/n_sell_orders.
        """
        env = self._env_with_n_buys(N_SPLITS * N_BUY)
        threshold = env.cycle_slot_size / self.AVG_SELL
        assert env.holdings > threshold

        holdings_before = env.holdings
        expected_sell_qty = holdings_before / N_SELL
        expected_remaining = holdings_before - expected_sell_qty

        env._process_fills(
            next_high=1060.0,   # sell_lo(1050) 체결, sell_hi(1100) 미체결
            next_low=1000.0,    # buy_hi=999.9, next_low=1000 > 999.9 → buy 미체결
            next_price=self.SELL_LO,
            buy_hi=999.9,
            buy_lo=899.0,
            sell_lo=self.SELL_LO,
            sell_hi=self.SELL_HI,
        )
        assert env.holdings == pytest.approx(expected_remaining, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SELL 우선 원칙 (step 통합 테스트)
# ─────────────────────────────────────────────────────────────────────────────

class TestSellFirstPrinciple:
    """
    같은 봉에서 sell + buy 동시 체결 가능할 때 sell이 먼저 처리되어야 한다.

    시나리오:
      스텝 0: next 봉 low=500 → buy_hi(≈999.9) 체결 → 포지션 매수
      스텝 1: next 봉 high=2000, low=500
              → sell_lo(≈1000.1) 체결 (sell 먼저)
              → holdings=0 → 사이클 종료
              → buy_hi(≈999.9) 체결 → 새 사이클 시작
    """

    def _build_env(self):
        n = WARMUP + 5
        prices = [PRICE] * n
        highs  = [PRICE] * n
        lows   = [PRICE] * n

        # 스텝 0의 다음 봉: low 낮게 → buy_hi 체결
        highs[WARMUP + 1] = PRICE
        lows[WARMUP + 1]  = 500.0

        # 스텝 1의 다음 봉: high 높게 + low 낮게 → sell + buy 동시
        highs[WARMUP + 2] = 2000.0
        lows[WARMUP + 2]  = 500.0

        df = _df_rows(prices, highs, lows)
        env = BTCGridTradingEnv(df, _cfg())
        env.reset(seed=0)
        return env

    def test_cycle_ends_before_new_buy(self):
        """sell이 먼저 처리되어 사이클 종료 후, buy로 새 사이클이 시작돼야 한다."""
        env = self._build_env()

        # 스텝 0: buy_hi 체결
        env.step(_act(0.0, 0.0))
        assert env.holdings > 0.0

        cycles_before = len(env.completed_cycles)

        # 스텝 1: sell + buy 동시 → sell 우선 → 사이클 종료 → buy 신규
        env.step(_act(0.0, 0.0))

        assert len(env.completed_cycles) > cycles_before, \
            "SELL 우선 처리: 사이클이 종료됐어야 함"
        assert env.in_cycle is True, \
            "SELL 후 BUY 발생: 새 사이클이 시작됐어야 함"

    def test_n_trades_increments_correctly(self):
        """같은 봉에서 sell + buy 2회 체결 → n_trades 누적."""
        env = self._build_env()
        env.step(_act(0.0, 0.0))   # 스텝 0: buy 1회
        trades_after_step0 = env.n_trades

        env.step(_act(0.0, 0.0))   # 스텝 1: sell + buy = 2회 이상
        assert env.n_trades > trades_after_step0 + 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Divergence — last_avg_price 활용
# ─────────────────────────────────────────────────────────────────────────────

class TestDivergence:
    def test_zero_before_any_trade(self):
        """거래 이력 없음 (에피소드 초반) → divergence = 0."""
        env = _env()
        obs = env._get_observation()
        assert obs[1] == pytest.approx(0.0)

    def test_uses_avg_price_while_holding(self):
        """보유 중: divergence = (avg_price - price) / avg_price."""
        env = _env()
        _buy(env, 1000.0)
        obs = env._get_observation()
        price = float(env.df.iloc[WARMUP]["close"])
        expected = (env.avg_price - price) / env.avg_price
        assert obs[1] == pytest.approx(expected, abs=1e-5)

    def test_uses_last_avg_price_after_full_sell(self):
        """
        전량 청산 후: divergence = (last_avg_price - price) / last_avg_price.
        df close=900, buy_price=1000 → last_avg_price≈1000
        divergence = (1000 - 900) / 1000 = 0.1
        """
        # df close=900 (≠ 매수가 1000)
        df = _df(price=900.0)
        env = BTCGridTradingEnv(df, _cfg())
        env.reset(seed=0)
        env.current_step = WARMUP

        env._execute_buy(fill_price=1000.0)   # avg_price ≈ 1000
        last_avg = env.avg_price
        env._execute_sell(1100.0, env.holdings, 1100.0)

        obs = env._get_observation()
        price = float(env.df.iloc[WARMUP]["close"])   # 900
        expected = (last_avg - price) / last_avg      # (1000-900)/1000 = 0.1
        assert obs[1] == pytest.approx(np.clip(expected, -5.0, 5.0), abs=1e-4)

    def test_divergence_nonzero_after_sell_when_prices_differ(self):
        """전량 청산 후에도 divergence ≠ 0 (last_avg_price != current_price)."""
        df = _df(price=900.0)
        env = BTCGridTradingEnv(df, _cfg())
        env.reset(seed=0)
        env.current_step = WARMUP
        env._execute_buy(fill_price=1000.0)
        env._execute_sell(1100.0, env.holdings, 1100.0)

        obs = env._get_observation()
        assert obs[1] != pytest.approx(0.0, abs=1e-4)

    def test_last_avg_price_branch_used_not_zero_branch(self):
        """
        last_avg_price > 0 분기가 사용되는지 확인.
        last_avg_price 설정 후 obs[1] != 0 이어야 한다 (price != last_avg_price 조건).
        """
        df = _df(price=800.0)
        env = BTCGridTradingEnv(df, _cfg())
        env.reset(seed=0)
        env.current_step = WARMUP

        env._execute_buy(fill_price=1000.0)
        assert env.last_avg_price == 0.0     # 아직 매도 전

        env._execute_sell(1050.0, env.holdings, 1050.0)
        assert env.last_avg_price > 0.0      # 청산 후 last_avg_price 저장 확인

        obs = env._get_observation()
        # price=800, last_avg≈1000 → divergence = (1000-800)/1000 = 0.2
        assert obs[1] != pytest.approx(0.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Reward 구조
# ─────────────────────────────────────────────────────────────────────────────

class TestReward:
    def test_no_trade_reward_zero(self):
        """
        보유 없고 체결도 없는 스텝: equity 불변 → reward ≈ 0.
        (high=low=price → 어떤 체결도 발생하지 않음)
        """
        df = _df(price=1000.0, high_offset=0.0, low_offset=0.0)
        env = BTCGridTradingEnv(df, _cfg())
        env.reset(seed=0)

        _, reward, _, _, _ = env.step(_act(0.0, 0.0))
        assert reward == pytest.approx(0.0, abs=1e-8)

    def test_reward_equals_equity_change_normalized(self):
        """
        체결이 있는 스텝:
        reward = (equity_after - equity_before) / start_capital - fee × n_trades

        next_low=999.5: buy_hi(999.9) 체결, buy_lo(999.0) 미체결 → n_trades=1 확정.
        """
        n = WARMUP + 5
        prices = [PRICE] * n
        highs  = [PRICE] * n
        lows   = [PRICE] * n
        # buy_hi = 1000*(1-0.0001) = 999.9, buy_lo = 1000*(1-0.001) = 999.0
        # next_low=999.5: 999.5 <= 999.9 → buy_hi 체결, 999.5 > 999.0 → buy_lo 미체결
        lows[WARMUP + 1] = 999.5

        df = _df_rows(prices, highs, lows)
        env = BTCGridTradingEnv(df, _cfg())
        env.reset(seed=0)

        equity_before = env.cash   # holdings=0 이므로 equity = cash
        n_trades_before = env.n_trades
        _, reward, _, _, _ = env.step(_act(0.0, 0.0))
        equity_after = env.cash + env.holdings * PRICE
        n_trades_step = env.n_trades - n_trades_before   # 실제 체결 수

        expected = (equity_after - equity_before) / INITIAL_CASH - FEE * n_trades_step
        assert reward == pytest.approx(expected, rel=1e-4)

    def test_no_cycle_bonus_on_cycle_end(self):
        """
        사이클 종료 스텝에서도 보너스가 없어야 한다 (bonus=0 설계).

        스텝 0: next_low=999.5 → buy_hi(999.9)만 체결 (1회)
        스텝 1: next_high=1000.5 → sell_lo(≈1000.1) 체결,
                sell_hi(avg_price*1.001 ≈ 999.9*1.001 ≈ 1000.9) 미체결 (1회)
                next_low=1000.0 > buy_hi(999.9) → buy 미체결
        """
        n = WARMUP + 5
        prices = [PRICE] * n
        highs  = [PRICE] * n
        lows   = [PRICE] * n
        lows[WARMUP + 1]  = 999.5    # 스텝 0: buy_hi(999.9)만 체결
        highs[WARMUP + 2] = 1000.5   # 스텝 1: sell_lo(1000.1) 체결, sell_hi(≈1000.9) 미체결
        lows[WARMUP + 2]  = 1000.0   # buy 미체결 (1000 > 999.9)

        df = _df_rows(prices, highs, lows)
        env = BTCGridTradingEnv(df, _cfg())
        env.reset(seed=0)

        env.step(_act(0.0, 0.0))   # 스텝 0: buy_hi 1회

        equity_before = env.cash + env.holdings * PRICE
        n_trades_before = env.n_trades
        _, reward, _, _, _ = env.step(_act(0.0, 0.0))   # 스텝 1: sell_lo 1회
        equity_after = env.cash + env.holdings * PRICE
        n_trades_step = env.n_trades - n_trades_before

        # 사이클 종료 확인
        assert len(env.completed_cycles) == 1

        # reward = equity_change / start_capital - fee × n_trades (보너스 없음)
        expected = (equity_after - equity_before) / INITIAL_CASH - FEE * n_trades_step
        assert reward == pytest.approx(expected, rel=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 10. gymnasium env_checker
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvChecker:
    def test_env_checker_passes(self):
        """gymnasium 공식 env_checker 통과."""
        from src.utils.config import load_config
        cfg = load_config("config/experiment_config.yaml")
        df  = pd.read_parquet("data/processed/btc_val.parquet")
        env = BTCGridTradingEnv(df, cfg)
        check_env(env)   # 예외 없으면 통과
