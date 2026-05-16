"""
scripts/eval_atr_test.py

ATR 고정 시스템을 Val/Test 세트에서 직접 평가.
trading_env.py의 _process_fills / _execute_buy / _execute_sell 로직을 정확히 재현.

direction multiplier (선택):
    trend_window : 사용할 trend 컬럼 윈도우 (시간 단위, e.g. 720)
    k            : direction 강도 계수 (0이면 기존 ATR과 동일)
    - 하락(trend < 0): buy gap *= (1 + k × |trend|)  → 더 깊은 낙폭 요구
    - 상승(trend > 0): sell gap *= (1 + k × trend)   → 더 높은 수익 요구
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils.config import load_config
from src.evaluation.metrics import compute_all


def run_atr_fixed(df: pd.DataFrame, cfg: dict,
                  trend_window: int | None = None,
                  k: float = 0.0) -> dict:
    coefs     = cfg["environment"]["formula_coefs"]
    A_b, C_b  = coefs["A_b"], coefs["C_b"]
    A_s, C_s  = coefs["A_s"], coefs["C_s"]
    n_splits  = cfg["environment"]["n_splits"]
    n_buy     = cfg["environment"]["n_buy_orders"]
    fee_rate  = cfg["environment"]["transaction_cost"]
    init_cash = cfg["environment"]["initial_cash"]

    # 포트폴리오 상태
    cash             = init_cash
    holdings         = 0.0
    avg_price        = 0.0
    in_cycle         = False
    cycle_slot_size  = 0.0   # env: cash / n_splits (사이클 시작 시 확정)
    per_order_size   = 0.0
    cycle_budget_rem = 0.0
    cycle_start_cash = init_cash
    cycle_start_step = 0

    equity_list = [cash]
    n_trades    = 0
    cycles      = []

    df = df.reset_index(drop=True)
    for i in range(len(df) - 1):
        price = float(df.at[i, "close"])
        atr_r = float(df.at[i, "volatility_raw"])
        nh    = float(df.at[i + 1, "high"])
        nl    = float(df.at[i + 1, "low"])
        nc    = float(df.at[i + 1, "close"])

        # ── direction multiplier ──────────────────────────────────
        if trend_window is not None and k > 0.0:
            col = f"trend_{trend_window}h_raw"
            trend_val = float(df.at[i, col])
            buy_mult  = 1.0 + k * max(0.0, -trend_val)  # 하락일수록 buy gap ↑
            sell_mult = 1.0 + k * max(0.0,  trend_val)  # 상승일수록 sell gap ↑
        else:
            buy_mult = sell_mult = 1.0

        # ── 주문 가격 계산 (formula_coefs 직접 사용) ──────────────
        sell_m = price     * (1 + atr_r * A_s * sell_mult)
        ref    = avg_price if avg_price > 0.0 else price
        sell_c = ref       * (1 + atr_r * C_s * sell_mult)
        buy_hi = price     * (1 - atr_r * A_b * buy_mult)
        buy_lo = price     * (1 - atr_r * C_b * buy_mult)

        # ── threshold_btc: 전량/분할 청산 기준 ───────────────────
        ref_price     = avg_price if avg_price > 0.0 else price
        threshold_btc = (cycle_slot_size / ref_price
                         if ref_price > 0.0 else 0.0)

        # ── 1. SELL 처리 (sell_market, sell_cost 각각 독립 실행) ──
        for sell_level in [sell_m, sell_c]:
            if holdings > 0.0 and nh >= sell_level:
                sell_qty = (holdings
                            if holdings <= threshold_btc
                            else holdings / n_splits)
                sell_qty = min(sell_qty, holdings)
                if sell_qty > 0.0:
                    proceeds = sell_qty * sell_level   # 지정가에 체결
                    fee      = proceeds * fee_rate
                    cash    += proceeds - fee
                    holdings -= sell_qty
                    n_trades += 1

                    if holdings < 1e-10:
                        # 전량 청산 → 사이클 종료
                        pnl_pct = (cash - cycle_start_cash) / cycle_start_cash
                        cycles.append({
                            "pnl_pct":     pnl_pct,
                            "cycle_hours": max(i - cycle_start_step, 1),
                            "n_trades":    1,
                        })
                        holdings         = 0.0
                        avg_price        = 0.0
                        in_cycle         = False
                        cycle_start_cash = cash

        # ── 2. BUY 처리 ──────────────────────────────────────────
        # n_buy_orders=2 → [buy_hi, buy_lo] 선형 보간
        if n_buy == 1:
            buy_prices = [buy_hi]
        else:
            buy_prices = [buy_hi + k / (n_buy - 1) * (buy_lo - buy_hi)
                          for k in range(n_buy)]

        for bp in buy_prices:
            if nl <= bp:
                # 사이클 최초 매수: 예산 확정
                if holdings == 0.0 and not in_cycle:
                    in_cycle         = True
                    cycle_start_cash = cash
                    cycle_start_step = i
                    cycle_slot_size  = cash / n_splits
                    per_order_size   = cycle_slot_size / n_buy
                    cycle_budget_rem = cash

                if cycle_budget_rem >= per_order_size:
                    spend    = per_order_size
                    fee      = spend * fee_rate
                    buy_qty  = (spend - fee) / bp   # 지정가에 체결
                    prev     = holdings
                    holdings += buy_qty
                    avg_price = (avg_price * prev + bp * buy_qty) / holdings
                    cash     -= spend
                    cycle_budget_rem -= spend
                    n_trades += 1

        equity_list.append(cash + holdings * nc)

    equity = pd.Series(equity_list, dtype=float)
    m = compute_all(equity, init_cash, n_trades, cycles)
    return m


def main():
    cfg     = load_config()
    df_val  = pd.read_parquet("data/processed/btc_val.parquet")
    df_test = pd.read_parquet("data/processed/btc_test.parquet")

    coefs = cfg["environment"]["formula_coefs"]
    tw    = coefs.get("trend_window", None)
    k     = float(coefs.get("k", 0.0))

    # ── exp026 ATR (direction 없음, k=0) ─────────────────────────
    # exp026 원래 계수로 복원해서 비교 기준선 계산
    from copy import deepcopy
    cfg_026 = deepcopy(cfg)
    cfg_026["environment"]["formula_coefs"].update({
        "A_b": 1.9211, "C_b": 5.7188, "A_s": 0.6875, "C_s": 9.6726,
    })
    cfg_026["environment"]["n_splits"] = 7

    print("=" * 60)
    print("exp026 ATR (direction 없음, n_splits=7)")
    print("=" * 60)
    mv26 = run_atr_fixed(df_val,  cfg_026, trend_window=None, k=0.0)
    mt26 = run_atr_fixed(df_test, cfg_026, trend_window=None, k=0.0)
    print(f"  {'':12s}  {'Val':>10s}  {'Test':>10s}")
    print(f"  {'Return':12s}  {mv26['total_return_pct']:>9.2f}%  {mt26['total_return_pct']:>9.2f}%")
    print(f"  {'Sharpe':12s}  {mv26['sharpe_ratio']:>10.3f}  {mt26['sharpe_ratio']:>10.3f}")
    print(f"  {'MDD':12s}  {mv26['max_drawdown_pct']:>9.2f}%  {mt26['max_drawdown_pct']:>9.2f}%")
    print(f"  {'Trades':12s}  {mv26['n_trades']:>10d}  {mt26['n_trades']:>10d}")

    # ── exp027 ATR+direction ──────────────────────────────────────
    print()
    print("=" * 60)
    print(f"exp027 ATR+direction (tw={tw}h, k={k:.4f}, n_splits={cfg['environment']['n_splits']})")
    print("=" * 60)
    mv27 = run_atr_fixed(df_val,  cfg, trend_window=tw, k=k)
    mt27 = run_atr_fixed(df_test, cfg, trend_window=tw, k=k)
    print(f"  {'':12s}  {'Val':>10s}  {'Test':>10s}")
    print(f"  {'Return':12s}  {mv27['total_return_pct']:>9.2f}%  {mt27['total_return_pct']:>9.2f}%")
    print(f"  {'Sharpe':12s}  {mv27['sharpe_ratio']:>10.3f}  {mt27['sharpe_ratio']:>10.3f}")
    print(f"  {'MDD':12s}  {mv27['max_drawdown_pct']:>9.2f}%  {mt27['max_drawdown_pct']:>9.2f}%")
    print(f"  {'Trades':12s}  {mv27['n_trades']:>10d}  {mt27['n_trades']:>10d}")

    # ── 개선 요약 ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("개선 요약 (exp027 - exp026)")
    print("=" * 60)
    print(f"  Val  Sharpe: {mv26['sharpe_ratio']:.3f} → {mv27['sharpe_ratio']:.3f}  "
          f"({mv27['sharpe_ratio']-mv26['sharpe_ratio']:+.3f})")
    print(f"  Test Sharpe: {mt26['sharpe_ratio']:.3f} → {mt27['sharpe_ratio']:.3f}  "
          f"({mt27['sharpe_ratio']-mt26['sharpe_ratio']:+.3f})")


if __name__ == "__main__":
    main()
