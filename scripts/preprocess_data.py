"""
BTC/USDT 1h 데이터 전처리 및 분할

입력:  data/raw/btc_usdt_1h.csv
출력:  data/processed/btc_train.parquet
       data/processed/btc_val.parquet
       data/processed/btc_test.parquet  ← 봉인, 학습 중 열람 금지

수행 작업:
    1. ATR(168) 계산
    2. log_price, volatility 계산
    3. rolling z-score 정규화 (window=168)
    4. NaN 행 제거 (warmup 기간)
    5. train / val / test 분할 후 parquet 저장
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config

# ── 설정 ──────────────────────────────────────────────────────
cfg        = load_config()
RAW_PATH   = Path("data/raw/btc_usdt_1h.csv")
OUT_DIR    = Path("data/processed")

TRAIN_END  = cfg["data"]["train_end"]   # "2022-12-31"
VAL_START  = cfg["data"]["val_start"]   # "2023-01-01"
VAL_END    = cfg["data"]["val_end"]     # "2023-12-31"
TEST_START = cfg["data"]["test_start"]  # "2024-01-01"

LOOKBACK   = cfg["indicators"]["price_lookback"]   # 168
ATR_PERIOD = cfg["indicators"]["atr_period"]       # 168
ZSCORE_WIN = cfg["indicators"]["zscore_window"]    # 168


# ── 지표 계산 함수 ─────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range (ATR) 계산."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def compute_log_price(close: pd.Series, lookback: int) -> pd.Series:
    """log(price / rolling_mean) — 최근 평균 대비 현재가 수준."""
    return np.log(close / close.rolling(lookback).mean())


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score 정규화. 미래 데이터 누수 없음."""
    mean = series.rolling(window).mean()
    std  = series.rolling(window).std()
    return (series - mean) / (std + 1e-8)


# ── 메인 ──────────────────────────────────────────────────────

def main():
    # 1. 원본 로드
    print(f"로드: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, index_col="timestamp", parse_dates=True)
    df = df.sort_index()
    print(f"  {len(df):,}개 캔들 | {df.index[0]} ~ {df.index[-1]}")

    # 2. 지표 계산
    print("\n지표 계산 중...")
    atr        = compute_atr(df, ATR_PERIOD)
    log_price  = compute_log_price(df["close"], LOOKBACK)
    volatility = atr / df["close"]

    # 3. Rolling z-score 정규화
    df["log_price"]         = log_price
    df["volatility_raw"]    = volatility
    df["zscore_log_price"]  = rolling_zscore(log_price,  ZSCORE_WIN)
    df["zscore_volatility"] = rolling_zscore(volatility, ZSCORE_WIN)

    # 4. NaN 제거 (warmup: 첫 168봉 × 2 = 336봉 이후부터 유효)
    before = len(df)
    df = df.dropna()
    print(f"  NaN 제거: {before - len(df)}행 제거 → {len(df):,}행 남음")

    # 5. 컬럼 정리 (환경에서 필요한 컬럼만 유지)
    keep_cols = ["open", "high", "low", "close", "volume",
                 "log_price", "volatility_raw",
                 "zscore_log_price", "zscore_volatility"]
    df = df[keep_cols]

    # 6. 분할
    train = df[df.index <= TRAIN_END]
    val   = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test  = df[df.index >= TEST_START]

    print(f"\n데이터 분할:")
    print(f"  Train : {len(train):>6,}개 | {train.index[0].date()} ~ {train.index[-1].date()}")
    print(f"  Val   : {len(val):>6,}개 | {val.index[0].date()} ~ {val.index[-1].date()}")
    print(f"  Test  : {len(test):>6,}개 | {test.index[0].date()} ~ {test.index[-1].date()} ← 봉인")

    # 7. parquet 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT_DIR / "btc_train.parquet")
    val.to_parquet(OUT_DIR / "btc_val.parquet")
    test.to_parquet(OUT_DIR / "btc_test.parquet")

    print(f"\n저장 완료 → {OUT_DIR}/")
    print("  btc_train.parquet")
    print("  btc_val.parquet")
    print("  btc_test.parquet  ← 학습 완료 전 열람 금지")

    # 8. 간단 통계 출력
    print(f"\nTrain 기초 통계:")
    print(train[["close", "zscore_log_price", "zscore_volatility"]].describe().round(4))


if __name__ == "__main__":
    main()
