"""
live_trading/bot.py
─────────────────────────────────────────────────────────────────────────────
메인 루프. 1시간봉 완성 시마다 실행.

실행:
  cd live_trading
  cp .env.example .env          # API 키 입력
  python bot.py                 # testnet 모드 (config.yaml exchange.testnet: true)

RL 전환:
  config.yaml 에서 formula.mode: rl, rl_model_path 설정 후 재실행
"""

from __future__ import annotations
import os
import sys
import time
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

# 프로젝트 루트를 sys.path에 추가 (src/ import 위해)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from live_trading.exchange import (
    build_exchange, fetch_ohlcv, compute_atr,
    get_balance, place_limit_order, cancel_order, fetch_open_orders,
)
from live_trading.formula import get_action, compute_order_prices
from live_trading.state_tracker import load_state, save_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "bot.log"),
    ],
)
log = logging.getLogger(__name__)


def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_rl_model(cfg: dict):
    """mode=rl 일 때 SB3 모델 로드."""
    model_path = cfg["formula"].get("rl_model_path")
    if not model_path:
        raise ValueError("rl_model_path가 config.yaml에 설정되지 않았습니다.")
    from stable_baselines3 import PPO
    return PPO.load(Path(__file__).parent / model_path)


def cancel_all_open_orders(ex, symbol: str, state) -> None:
    """열린 주문 전량 취소."""
    for oid in list(state.open_order_ids):
        cancel_order(ex, oid, symbol)
    state.open_order_ids.clear()


def emergency_stop(ex, symbol: str, state, price: float) -> None:
    """MDD 한도 초과 — 주문 취소 후 전량 매도."""
    log.warning("!!! 긴급 정지: MDD 한도 초과. 전량 청산 시작.")
    cancel_all_open_orders(ex, symbol, state)
    if state.holdings_btc > 0:
        ex.create_market_sell_order(symbol, state.holdings_btc)
        log.warning(f"시장가 전량 매도: {state.holdings_btc:.6f} BTC @ ~{price:.2f}")
    log.warning("봇 종료.")
    sys.exit(1)


def tick(cfg: dict, ex, state, model=None) -> None:
    """1시간봉 1회 실행 로직."""
    symbol    = cfg["trading"]["symbol"]
    timeframe = cfg["trading"]["timeframe"]
    atr_win   = cfg["trading"]["atr_window"]

    # 1. 캔들 + ATR 갱신
    df = fetch_ohlcv(ex, symbol, timeframe, limit=atr_win + 10)
    price     = float(df["close"].iloc[-1])
    atr_ratio = compute_atr(df, window=atr_win)

    # 2. equity / MDD 체크
    eq = state.equity(price)
    if eq > state.peak_equity:
        state.peak_equity = eq
    mdd = state.max_drawdown(price)
    log.info(f"price={price:.2f}  equity={eq:.4f}  MDD={mdd:.2%}  "
             f"cash={state.cash:.4f}  BTC={state.holdings_btc:.6f}")

    if mdd >= cfg["trading"]["max_drawdown_stop"]:
        emergency_stop(ex, symbol, state, price)

    # 3. 체결 동기화 — 거래소의 실제 체결 내역으로 상태 업데이트
    _sync_fills(ex, symbol, state, price, cfg)

    # 4. action 계산
    agg, pt = get_action(cfg, state=_build_state_vector(state, price, atr_ratio), model=model)

    # 5. 주문 가격 계산
    orders = compute_order_prices(price, atr_ratio, agg, pt, state.avg_price, cfg)
    log.info(f"action=({agg:.3f},{pt:.3f})  "
             f"buy_hi={orders['buy_hi']:.2f}  buy_lo={orders['buy_lo']:.2f}  "
             f"sell_mkt={orders['sell_market']:.2f}  sell_cost={orders['sell_cost']:.2f}")

    # 6. 기존 주문 취소 후 재발행 (매 봉마다 갱신)
    cancel_all_open_orders(ex, symbol, state)
    _place_orders(ex, symbol, state, orders, price, cfg)

    save_state(state)


def _sync_fills(ex, symbol: str, state, price: float, cfg: dict) -> None:
    """
    열린 주문 목록을 확인해 체결된 주문의 상태를 반영.
    TODO: 완전한 사이클 로직은 구현 예정.
    """
    still_open = []
    for oid in state.open_order_ids:
        try:
            o = ex.fetch_order(oid, symbol)
            if o["status"] == "closed":
                filled_qty  = float(o["filled"])
                filled_price = float(o["average"] or o["price"])
                fee = filled_qty * filled_price * cfg["trading"]["fee_rate"]
                if o["side"] == "buy":
                    cost = filled_qty * filled_price + fee
                    state.cash -= cost
                    # avg_price 갱신
                    total_btc = state.holdings_btc + filled_qty
                    if total_btc > 0:
                        state.avg_price = (
                            state.avg_price * state.holdings_btc + filled_price * filled_qty
                        ) / total_btc
                    state.holdings_btc = total_btc
                    if not state.in_cycle:
                        state.in_cycle = True
                        state.cycle_start_cash = state.cash + state.holdings_btc * price
                    log.info(f"매수 체결: {filled_qty:.6f} BTC @ {filled_price:.2f}")
                else:  # sell
                    proceeds = filled_qty * filled_price - fee
                    state.cash += proceeds
                    state.holdings_btc -= filled_qty
                    if state.holdings_btc <= 1e-8:
                        state.holdings_btc = 0.0
                        state.avg_price    = 0.0
                        state.in_cycle     = False
                        state.completed_cycles += 1
                    log.info(f"매도 체결: {filled_qty:.6f} BTC @ {filled_price:.2f}")
            elif o["status"] == "open":
                still_open.append(oid)
        except Exception as e:
            log.warning(f"주문 {oid} 조회 실패: {e}")
            still_open.append(oid)
    state.open_order_ids = still_open


def _place_orders(ex, symbol: str, state, orders: dict, price: float, cfg: dict) -> None:
    """매수·매도 지정가 주문 발행."""
    t = cfg["trading"]
    n_buy  = t["n_buy_orders"]
    n_sell = t["n_sell_orders"]
    slot   = state.cash / t["n_splits"]
    per_buy_usdt = slot / n_buy

    # 매수: 현금이 충분할 때만
    for buy_price in [orders["buy_hi"], orders["buy_lo"]]:
        if state.cash >= per_buy_usdt and buy_price > 0:
            qty = (per_buy_usdt / buy_price) * (1 - t["fee_rate"])
            try:
                o = place_limit_order(ex, symbol, "buy", round(buy_price, 2), round(qty, 6))
                state.open_order_ids.append(o["id"])
            except Exception as e:
                log.warning(f"매수 주문 실패 ({buy_price:.2f}): {e}")

    # 매도: 보유 중일 때만
    if state.holdings_btc > 1e-8:
        sell_qty = state.holdings_btc / n_sell
        for sell_price in [orders["sell_market"], orders["sell_cost"]]:
            if sell_price > price:
                try:
                    o = place_limit_order(ex, symbol, "sell", round(sell_price, 2), round(sell_qty, 6))
                    state.open_order_ids.append(o["id"])
                except Exception as e:
                    log.warning(f"매도 주문 실패 ({sell_price:.2f}): {e}")


def _build_state_vector(state, price: float, atr_ratio: float):
    """
    RL 모드용 state 벡터 (placeholder).
    RL 재설계 완료 후 7D 벡터로 교체 예정.
    현재는 fixed 모드에서만 사용되지 않으므로 None 반환.
    """
    return None


def main():
    load_dotenv(Path(__file__).parent / ".env")
    cfg   = load_config()
    ex    = build_exchange(cfg)
    state = load_state(cfg["trading"]["initial_cash"])
    model = load_rl_model(cfg) if cfg["formula"]["mode"] == "rl" else None

    log.info(f"봇 시작 — mode={cfg['formula']['mode']}, testnet={cfg['exchange']['testnet']}")
    log.info(f"초기 상태: cash={state.cash:.4f}, BTC={state.holdings_btc:.6f}")

    while True:
        try:
            tick(cfg, ex, state, model)
        except KeyboardInterrupt:
            log.info("사용자 중단. 열린 주문 취소 후 종료.")
            cancel_all_open_orders(ex, cfg["trading"]["symbol"], state)
            save_state(state)
            break
        except Exception as e:
            log.error(f"tick 오류: {e}", exc_info=True)

        # 다음 봉까지 대기 (약 1시간)
        time.sleep(3600)


if __name__ == "__main__":
    main()
