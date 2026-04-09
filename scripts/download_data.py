"""
BTC/USDT 1시간봉 데이터 다운로드 (Binance, ccxt)

사용법:
    python scripts/download_data.py

출력:
    data/raw/btc_usdt_1h.csv
"""

import time
from pathlib import Path

import ccxt
import pandas as pd

from src.utils.config import load_config

# ── 설정 ──────────────────────────────────────────────────────
cfg = load_config()
SYMBOL     = cfg["data"]["symbol"]       # "BTC/USDT"
TIMEFRAME  = cfg["data"]["interval"]     # "1h"
SINCE      = cfg["data"]["train_start"]  # "2020-01-01"
OUTPUT     = Path("data/raw/btc_usdt_1h.csv")
LIMIT      = 1000                        # Binance 최대 요청 캔들 수
SLEEP_SEC  = 0.2                         # 요청 간격 (rate limit 방지)


def fetch_all_candles(exchange: ccxt.Exchange, since_str: str) -> list:
    """since_str(ISO 날짜)부터 현재까지 1h 캔들 전체 수집."""
    since_ms = exchange.parse8601(f"{since_str}T00:00:00Z")
    all_candles: list = []
    request_count = 0

    print(f"다운로드 시작: {SYMBOL} {TIMEFRAME} / {since_str} ~ 현재")

    while True:
        candles = exchange.fetch_ohlcv(
            SYMBOL, TIMEFRAME, since=since_ms, limit=LIMIT
        )
        if not candles:
            break

        all_candles.extend(candles)
        request_count += 1
        last_ts = candles[-1][0]
        last_dt = pd.to_datetime(last_ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")
        print(f"  [{request_count:>3}회] {len(all_candles):>6}개 수집 | 마지막: {last_dt}")

        # 마지막 캔들의 다음 ms부터 재요청
        since_ms = last_ts + 1

        # limit보다 적게 반환 → 더 이상 데이터 없음
        if len(candles) < LIMIT:
            break

        time.sleep(SLEEP_SEC)

    return all_candles


def save_csv(candles: list) -> pd.DataFrame:
    """캔들 리스트 → DataFrame 변환 후 CSV 저장."""
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    # 중복 제거
    df = df[~df.index.duplicated(keep="last")]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT)
    return df


def main():
    exchange = ccxt.binance({"enableRateLimit": True})
    candles  = fetch_all_candles(exchange, SINCE)
    df       = save_csv(candles)

    print(f"\n완료: {len(df):,}개 캔들 저장 → {OUTPUT}")
    print(f"기간: {df.index[0]} ~ {df.index[-1]}")
    print(f"컬럼: {list(df.columns)}")


if __name__ == "__main__":
    main()
