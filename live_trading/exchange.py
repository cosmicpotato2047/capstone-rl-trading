"""
live_trading/exchange.py
─────────────────────────────────────────────────────────────────────────────
ccxt Binance 래퍼.
잔고 조회, 캔들 수신, 지정가 주문 생성/취소.
"""

from __future__ import annotations
import os
import ccxt
import pandas as pd
import numpy as np


def build_exchange(cfg: dict) -> ccxt.Exchange:
    api_key    = os.environ[cfg["exchange"]["api_key_env"]]
    api_secret = os.environ[cfg["exchange"]["api_secret_env"]]

    ex = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "options": {"defaultType": "spot"},
    })
    if cfg["exchange"]["testnet"]:
        ex.set_sandbox_mode(True)

    return ex


def fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """최근 limit 개 캔들 반환 (timestamp, open, high, low, close, volume)."""
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").astype(float)
    return df


def compute_atr(df: pd.DataFrame, window: int = 168) -> float:
    """ATR(window) / 현재 종가 반환 (= volatility_raw)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    return float(atr / close.iloc[-1])


def get_balance(ex: ccxt.Exchange) -> dict:
    """{'USDT': float, 'BTC': float} 반환."""
    bal = ex.fetch_balance()
    return {
        "USDT": float(bal["free"].get("USDT", 0.0)),
        "BTC":  float(bal["free"].get("BTC",  0.0)),
    }


def place_limit_order(
    ex: ccxt.Exchange,
    symbol: str,
    side: str,       # "buy" | "sell"
    price: float,
    amount: float,   # BTC 수량
) -> dict:
    """지정가 주문 생성. 주문 딕셔너리 반환."""
    return ex.create_limit_order(symbol, side, amount, price)


def cancel_order(ex: ccxt.Exchange, order_id: str, symbol: str) -> None:
    try:
        ex.cancel_order(order_id, symbol)
    except ccxt.OrderNotFound:
        pass  # 이미 체결·취소된 주문


def fetch_open_orders(ex: ccxt.Exchange, symbol: str) -> list[dict]:
    return ex.fetch_open_orders(symbol)


def fetch_order(ex: ccxt.Exchange, order_id: str, symbol: str) -> dict:
    return ex.fetch_order(order_id, symbol)
