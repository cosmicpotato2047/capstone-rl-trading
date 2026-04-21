"""
scripts/eval_atr_test.py

ATR 고정 시스템(exp023 계수)을 Val/Test 세트에서 직접 평가.
trading_env.py의 _process_fills / _execute_buy / _execute_sell 로직을 정확히 재현.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils.config import load_config
from src.evaluation.metrics import compute_all


def run_atr_fixed(df: pd.DataFrame, cfg: dict) -> dict:
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

        # ── 주문 가격 계산 (formula_coefs 직접 사용) ──────────────
        sell_m = price     * (1 + atr_r * A_s)
        ref    = avg_price if avg_price > 0.0 else price
        sell_c = ref       * (1 + atr_r * C_s)
        buy_hi = price     * (1 - atr_r * A_b)
        buy_lo = price     * (1 - atr_r * C_b)

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

    print("=" * 50)
    print("ATR 고정 시스템 (exp023 계수) 평가")
    print("=" * 50)
    coefs = cfg["environment"]["formula_coefs"]
    print(f"A_b={coefs['A_b']:.4f}  C_b={coefs['C_b']:.4f}")
    print(f"A_s={coefs['A_s']:.4f}  C_s={coefs['C_s']:.4f}")
    print(f"n_splits={cfg['environment']['n_splits']}")
    print()

    mv = run_atr_fixed(df_val, cfg)
    print("[Val 2021~2023H1]")
    print(f"  Return : {mv['total_return_pct']:.2f}%")
    print(f"  Sharpe : {mv['sharpe_ratio']:.3f}")
    print(f"  MDD    : {mv['max_drawdown_pct']:.2f}%")
    print(f"  Trades : {mv['n_trades']}")
    print()

    mt = run_atr_fixed(df_test, cfg)
    print("[Test 2023H2~2026]")
    print(f"  Return : {mt['total_return_pct']:.2f}%")
    print(f"  Sharpe : {mt['sharpe_ratio']:.3f}")
    print(f"  MDD    : {mt['max_drawdown_pct']:.2f}%")
    print(f"  Trades : {mt['n_trades']}")


if __name__ == "__main__":
    main()
