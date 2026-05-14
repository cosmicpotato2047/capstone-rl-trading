# 현실적 체결 시뮬레이션 — Slippage, Partial Fill, Queue Position

> QuantConnect, NautilusTrader 등 상용 백테스트 엔진의 표준 모델.
> 학술 참조: Almgren & Chriss (2001) Market Impact, Cont & Kukanov (2017) Order Routing.

## 요지

1. 1h OHLCV 봉만으로 체결을 시뮬레이션하는 것은 **본질적으로 근사**. 실 거래에서는 호가창 큐 위치가 결정적.
2. **3가지 핵심 모델**: (a) Slippage, (b) Partial Fill, (c) Queue Position.
3. 우리 exp026의 체결가 수정(`next_low/next_high` → 지정가)이 첫 단추. **다음 단계**는 슬리피지와 큐 모델 도입.

## 시뮬레이션 충실도 단계

| Level | 가격 데이터 | 체결 모델 | 적용 |
|---|---|---|---|
| **L0 (현재 우리)** | OHLCV 1h | 지정가가 high/low 사이면 체결 | 기본 백테스트 |
| **L1** | OHLCV + slippage | + 체결가에 noise (e.g., ±0.02%) | 보수적 백테스트 |
| **L2** | + Bid/Ask 시계열 | + 스프레드 차감, partial fill | 준현실 |
| **L3** | Full order book (L2 data) | + 큐 위치, market impact | 학술 표준 |
| **L4** | Tick-by-tick | + 레이턴시, 호가 변동 시뮬 | HFT 수준 |

우리 프로젝트 1h 봉 → L1까지가 현실적 목표.

## Slippage Model

### Linear Model
```
slippage = β × (order_size / avg_volume)
체결가 = limit_price ± slippage  (방향: 손실 측)
```

### Volume Participation Model (QuantConnect default)
```
participation = order_size / bar_volume
slippage_ratio = α × participation^γ   (γ ≈ 0.5~1.0)
slippage = price × slippage_ratio
```

→ 작은 주문(우리 케이스): slippage ≈ 0.01~0.05%
→ 큰 주문 또는 저유동성: 0.5%+

### 우리 적용 (가장 단순):
```python
# slippage_rate 추가 (0.02% default)
def execute_buy(limit_price, ...):
    actual_price = limit_price * (1 + slippage_rate)  # 매수는 더 비싸게
    spend = qty * actual_price + fee
    ...

def execute_sell(limit_price, ...):
    actual_price = limit_price * (1 - slippage_rate)  # 매도는 더 싸게
    proceeds = qty * actual_price - fee
    ...
```

## Partial Fill Model

### 트리거 조건 + 봉 거래량 기반
```python
# bar_volume = 봉의 거래량
# 우리 주문 크기 = order_qty

if next_low <= buy_limit:  # 지정가 도달
    fill_ratio = min(1.0, bar_volume × 0.001 / order_qty)
    # 봉 거래량의 0.1%까지만 우리가 차지 가능 가정
    actual_qty = order_qty * fill_ratio
    # 나머지는 다음 봉으로 이월 또는 취소
```

→ 우리 시스템에서는 사이클 슬롯 크기가 일반적으로 작음 (1000 USDT 미만)
→ 1h BTC 봉 거래량 보통 100만 USDT+ → partial fill 확률 낮음
→ **현재는 우선순위 낮음**. 자본 규모 증가 시 도입.

## Queue Position Model

호가창 정상 가격 도달 후 실제 체결까지의 확률 모델:

```python
# 호가창 도달 시점의 큐 길이 = q
# 우리 주문이 들어간 시점부터 q만큼 소진되어야 우리 차례
# 봉 내에서 소진 안 되면 미체결

fill_probability = min(1.0, cumulative_volume_at_price / queue_length)
```

→ L2 데이터 필요. 1h 봉 데이터로는 추정만 가능.
→ 보수적 가정: "가격이 우리 limit를 넘어서면 체결, 단순히 touch만 하면 50% 확률"

## 우리 프로젝트 진단 + 단계별 개선

### 현재 (L0)
```python
# trading_env.py:_process_fills
if next_low <= buy_hi:
    execute_buy(price=buy_hi)  # 지정가 자체로 체결
```

→ 슬리피지 0, partial fill 없음, queue position 0% (즉시 체결 가정).

### Step 1: Slippage 추가 (D-LV1)
```python
# config
slippage_rate: 0.0002  # 0.02%

# _execute_buy
actual_price = limit_price * (1 + self.slippage_rate)
```

→ exp026 best params로 재학습 → 어떤 결과 변화?
→ ATR 시스템: Sharpe 1.978 → 어디로?
→ RL: Sharpe 0.896 → 어디로?

### Step 2: 봉 내 가격 경로 추정 (D-LV1.5)
```python
# 봉 내 가격 변동을 sigmoid 또는 random walk로 시뮬
# 지정가 도달 시점에 따라 체결 확률 결정

if next_low <= buy_limit <= next_high:
    # 가격이 봉 내에서 limit를 spent_time만큼 머물렀다 추정
    # 그 시간 안에 queue가 소진될 확률
    fill_prob = sigmoid((next_low - buy_limit) / atr)
    if random() < fill_prob:
        execute_buy(price=buy_limit + slippage)
```

→ 결정론적 시뮬레이션 → 확률적 시뮬레이션. 여러 seed 평가 필수.

### Step 3: Paper trading로 sim2real gap 측정 (D-LV2)
- 1주 paper trade → 실제 Sharpe, MDD 측정
- Sim Sharpe와 비교 → gap 계산
- 큰 gap이면 → Step 1, 2 파라미터 재조정

## 학술적 참조

- **Almgren & Chriss (2001)** _Optimal execution of portfolio transactions._ → 시장 충격 모델의 정석. 우리는 작아서 직접 적용 어렵지만 개념 차용 가능.
- **Cont & Kukanov (2017)** _Optimal order routing._ → 다중 거래소 환경에서 partial fill 동적 처리.
- **Stoikov & Sağlam (2009)** _Option market making under inventory risk._ → 마켓 메이커가 직면하는 adverse selection.

## 우리 시스템의 최우선 액션 (다음 실험에 즉시)

1. **slippage_rate 파라미터 추가** (1일):
   - `config/experiment_config.yaml`: `slippage_rate: 0.0002` 추가
   - `trading_env.py:_execute_buy/_sell`: 체결가 보정
   - tests 갱신

2. **exp026 재실행**:
   - ATR 시스템: 새 슬리피지 환경에서 Sharpe 변화
   - RL: 재학습 후 변화

3. **Paper trading 인프라 본격 시작** (`live_trading/`):
   - 이미 폴더 구조 있음
   - Binance testnet 연결
   - 1주 forward test → 실 sim gap 측정

## 백링크

- [[sim2real_finance]] — Sim2real gap의 일반론
- [[reward_hacking]] — 비현실적 체결이 hacking 채널
- [[avellaneda_stoikov_2008]] — 마켓 메이킹과 inventory risk

## 출처

- [QuantConnect — Slippage docs](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/slippage/key-concepts)
- [NautilusTrader Backtesting docs](https://nautilustrader.io/docs/latest/concepts/backtesting/)
- [Backtesting Limitations: Slippage and Liquidity](https://www.luxalgo.com/blog/backtesting-limitations-slippage-and-liquidity-explained/)
- [Why Backtesting Environments Differ from Live Markets](https://algobulls.com/blog/algo-trading/backtesting-technical-factor)
