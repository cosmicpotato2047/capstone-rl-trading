"""
live_trading/formula.py
─────────────────────────────────────────────────────────────────────────────
주문 간격(gap) 계산.
mode: fixed  → config의 고정 계수 + 고정 action 사용
mode: rl     → RL 모델이 현재 state를 보고 action 결정

RL 재설계 완료 후 config.yaml의 mode와 rl_model_path만 바꾸면 전환 완료.
"""

from __future__ import annotations
import numpy as np


def get_action(cfg: dict, state: np.ndarray | None = None, model=None) -> tuple[float, float]:
    """
    Returns (aggressiveness, profit_target) in [0, 1].

    fixed 모드: config 고정값 반환
    rl 모드:    RL 모델이 state를 보고 action 결정
    """
    mode = cfg["formula"]["mode"]
    if mode == "fixed":
        return cfg["formula"]["aggressiveness"], cfg["formula"]["profit_target"]
    elif mode == "rl":
        if model is None:
            raise ValueError("mode=rl 이지만 model이 None입니다.")
        if state is None:
            raise ValueError("mode=rl 이지만 state가 None입니다.")
        action, _ = model.predict(state[np.newaxis, :], deterministic=True)
        agg = float(np.clip(action[0][0], 0.0, 1.0))
        pt  = float(np.clip(action[0][1], 0.0, 1.0))
        return agg, pt
    else:
        raise ValueError(f"알 수 없는 formula.mode: {mode}")


def compute_order_prices(
    price: float,
    atr_ratio: float,
    aggressiveness: float,
    profit_target: float,
    avg_price: float,
    cfg: dict,
) -> dict:
    """
    ATR 비례 스케일링으로 지정가 주문 가격 4개 계산.

    Parameters
    ----------
    price        : 현재 가격
    atr_ratio    : ATR(168) / price  (전처리 컬럼 volatility_raw)
    aggressiveness, profit_target : [0, 1] 행동값
    avg_price    : 현재 평균 매수가 (미보유 시 price 사용)
    cfg          : config.yaml 딕셔너리
    """
    c = cfg["formula"]

    buy_hi_gap      = atr_ratio * (c["A_b"] + aggressiveness * c["B_b"])
    buy_lo_gap      = atr_ratio * (c["C_b"] + aggressiveness * c["D_b"])
    sell_market_gap = atr_ratio * (c["A_s"] + profit_target  * c["B_s"])
    sell_cost_gap   = atr_ratio * (c["C_s"] + profit_target  * c["D_s"])

    ref_price = avg_price if avg_price > 0 else price

    return {
        "buy_hi":      price     * (1 - buy_hi_gap),
        "buy_lo":      price     * (1 - buy_lo_gap),
        "sell_market": price     * (1 + sell_market_gap),
        "sell_cost":   ref_price * (1 + sell_cost_gap),
    }
