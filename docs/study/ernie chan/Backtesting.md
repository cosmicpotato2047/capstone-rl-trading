Chapter 3 "Backtesting"은 **약 50페이지(p.33-82, 2판 기준)** 분량으로, Chan 본인이 "이 책에서 가장 기술적인 챕터 중 하나"라고 한 부분입니다. 사용자가 지적한 5가지 주제가 거의 다 여기 들어있습니다.

## 챕터 구조 (페이지 기준, 2판)

|섹션|페이지|내용|
|---|---|---|
|§3.1 Common Backtesting Platforms|p.34-40|Excel/MATLAB/Python/R/QuantConnect/Blueshift 비교|
|§3.2 Finding and Using Historical Databases|p.40-46|데이터 품질 점검|
|§3.2.1 Split·Dividend 조정|p.41|(BTC 무관)|
|§3.2.2 Survivorship Bias Free|p.44|★|
|§3.2.3 **High/Low Data 사용 여부**|p.46|★★ **(해상도 이슈 핵심)**|
|§3.3 Performance Measurement|p.47-56|★ Sharpe, drawdown, MAR ratio|
|§3.4 Common Backtesting Pitfalls to Avoid|p.57-71||
|§3.4.1 **Look-Ahead Bias**|p.58|★★|
|§3.4.2 Data-Snooping Bias|p.59-71|★ (가장 긴 단일 섹션)|
|§3.5 **Transaction Costs**|p.72-76|★★|
|§3.6 Strategy Refinement|p.77-82||

§3.1은 도구 소개라 **건너뛰어도 됩니다**(yfinance + 자체 Python env 사용 중). 나머지는 모두 RL env 정확성과 직결됩니다.

---

## ① Look-Ahead Bias (§3.4.1, p.58)

**책의 정의**: 시점 t의 의사결정에 t+1 이후에야 알 수 있는 정보를 사용하는 것. **"가장 흔한 백테스트 버그"** 라고 Chan이 명시.

**Chan이 드는 흔한 사례들**:

- 당일 종가를 사용해서 당일 매매 결정
- 종목 split이 미래에 일어났는데 그 조정가를 과거 시점 의사결정에 사용
- "전체 기간 평균"으로 정규화 (rolling이 아닌 global statistic)

**RL 프로젝트에서의 위험 지점**:

1. **상태 정규화**: `log_price = log(price) / global_mean(log_price)` 같은 정규화는 **즉시 look-ahead**. 반드시 **rolling window** 또는 **expanding window with min_periods**로 계산.
2. **volatility 계산**: centered window는 미래를 본다. **반드시 trailing window** (예: `df['volatility'] = df['return'].rolling(window=24, min_periods=24).std().shift(0)` — shift 부호 검증 필수).
3. **divergence 상태변수**: "현재가 vs 평균"의 평균이 미래 데이터를 포함하는지 점검. EMA, SMA 모두 trailing이어야 함.
4. **보상 계산 시점**: action_t를 취한 후 t+1 가격으로 평가. action_t 결정에 price_t+1이 들어가면 안 됨. **PPO env의 step() 함수 시점 정렬**을 단위 테스트로 검증해야 함.
5. **숨은 함정**: `df = df.dropna()` 같은 전처리에서 미래의 nan 조건이 과거 인덱스에 영향 주는 경우.

**디버깅 팁**: Chan 자신도 추천하는 방법으로, **백테스트 결과가 너무 좋으면 99% look-ahead 버그**입니다. Sharpe > 3은 의심해야 함.

---

## ② Survivorship Bias (§3.2.2, p.44)

**책의 정의**: 살아남은 종목만 데이터셋에 있어서 실제보다 수익률이 과대평가되는 편향.

**프로젝트 적용**: BTC 단일 자산이라 직접적으로는 무관. **하지만 두 가지 변형이 있습니다**:

1. **Regime survivorship**: 강세장 기간에만 학습 → "BTC가 항상 회복한다"는 잘못된 사전믿음을 학습. 2017-현재 전 구간 사용 시 **각 regime의 비중**을 의식. 강세장 30%, 약세장 30%, 횡보장 40% 정도가 균형.
2. **거래소 survivorship**: yfinance의 BTC-USD는 사실 **Coinbase 데이터** (또는 합성). FTX, Mt.Gox 등 사라진 거래소의 데이터는 빠져있음. 학술 프로젝트 수준에선 큰 문제 아니지만 디펜스에서 질문 받을 수 있음.

---

## ③ High/Low Data 사용 (§3.2.3, p.46) — 해상도 이슈의 핵심

**책의 핵심 경고**: **"High와 Low는 순간 가격이고, 그 가격에 실제로 체결됐다는 보장이 없다."** Open/Close는 마감 메커니즘이 있어 비교적 신뢰 가능하지만, High/Low는 1틱 스파이크일 수 있고, 그 가격에 매수/매도 의지가 충분히 있었다는 보장도 없음.

**프로젝트 적용** (이전 작업의 그 이슈):

- 이전에 `grid_drift_engine.py`가 실제 트레이더보다 underperform한 원인을 **일중/일봉 해상도 갭**으로 진단하셨던 부분, **Chan이 정확히 그 이슈를 §3.2.3에서 다룹니다**.
- 1시간봉으로 그리드 봇 백테스트 시 **체결 가정**이 큰 문제:
    - **Naive 가정**: 1시간 동안 high가 그리드 위에 닿으면 매도 체결됨 → **너무 낙관적**
    - **현실**: 그리드 위에 limit sell 주문을 미리 걸어둬야 체결됨. 가격이 잠깐 스쳐도 호가창 깊이가 부족하면 미체결
- **Chan의 권장 보수적 접근**: high/low 기반 체결 가정을 피하고 **OHLC 평균** 또는 **VWAP** 사용. 더 보수적으로는 **다음 봉 open**으로 체결.
- **PPO env에서의 구현**:
    - 비관적 모델: action_t의 그리드 주문은 **t+1 봉의 open 또는 close**에서만 체결
    - 중간 모델: high/low 기반 체결하되 **slippage 페널티** 추가
    - **반드시 둘 다 시뮬레이션**해서 sensitivity 보고. 디펜스에서 강력한 무기.
- 가능하면 1분봉 또는 tick 데이터로 보정 백테스트를 **한 번이라도** 돌려보면, 1시간봉 백테스트의 신뢰구간을 추정할 수 있음.

---

## ④ Transaction Costs (§3.5, p.72-76)

**책의 핵심 주장**: 거래비용은 **4가지 구성요소**:

1. **Commission** (수수료) — BTC 거래소: maker 0.02-0.04%, taker 0.04-0.1%
2. **Bid-Ask Spread** — 절반 비용으로 계산 (mid 기준)
3. **Slippage** — 주문 사이즈와 시장 유동성에 따라
4. **Market Impact** — 큰 주문 시 가격 자체 이동 (소액 그리드 봇은 무시 가능)

**Chan의 구체적 권고**:

- **수수료가 단방향 0.05%면 왕복 0.1%**. Sharpe 계산 시 직접 차감.
- **Sharpe ratio가 비용 차감 후 음수로 가는 전략은 폐기**.
- **고빈도 전략일수록 비용 가정에 민감** — sensitivity analysis 필수.

**PPO env 구현 권장**:

```
# 매 거래 시
gross_pnl = (price_now - entry_price) * position
commission_cost = abs(trade_size) * price_now * 0.0008  # 왕복 보수적 가정
slippage_cost = abs(trade_size) * price_now * 0.0005    # 추가 보수
net_pnl = gross_pnl - commission_cost - slippage_cost
reward = net_pnl  # 이 값이 reward에 들어가야 함
```

**그리드 봇 특수 위험**: 그리드는 거래수가 많음 → 비용이 가장 잘 드러나는 전략 유형. **비용 0%일 때 Sharpe와 비용 0.1% 일 때 Sharpe를 둘 다 보고**해야 합니다. 두 결과 차이가 클수록 전략은 fragile.

---

## ⑤ Sharpe & Drawdown 계산 (§3.3, p.47-56)

이 챕터에서 Chan이 **가장 자세히 다루는 단일 주제**입니다. 발췌하면:

### Sharpe Ratio의 미묘한 점들

Chan은 Sharpe ratio, maximum drawdown, MAR ratio를 가장 중요한 성과 지표 세 가지로 꼽으며, CAGR(연간복리수익률)은 분모 정의가 모호해서 권장하지 않는다고 명시합니다.

**핵심 미묘점들**:

1. **무위험 수익률 차감 여부**: 자본을 투입하는 전략은 차감, dollar-neutral 전략은 차감 안 함. **BTC 그리드 봇은 자본 투입 전략 → 차감.** 단, BTC 변동성 대비 무위험 수익률은 미미해서 결과 차이 거의 없음.
2. **Annualization factor**:
    - 주식 일봉: √252
    - **BTC는 24/7 거래** → 일봉 기준 **√365**
    - **1시간봉 기준** → **√(365 × 24) = √8760 ≈ 93.6**
    - 이 부분 자주 틀립니다. 디펜스에서 검증 질문 받기 쉬움.
3. **MAR ratio = CAGR / |max drawdown|**: Chan은 이를 보조 지표로 권장하며, 레버리지에 비교적 독립적이라 비교가 쉽다고 강조합니다.

### Drawdown 계산

- **Max drawdown** = 자산 곡선의 최대 peak-to-trough 손실률
- **Max drawdown duration** = 신고점 갱신까지 걸린 최대 일수
- 두 가지 모두 보고. 실전 사용 가능성은 duration이 좌우.

Chan은 자신의 강의 슬라이드에서 "연 30% 수익이지만 Sharpe 0.7, MDD duration 2년인 전략을 백테스트할 가치가 있는가?"라는 질문에 "No"라고 답합니다. 낮은 Sharpe(<1)와 긴 MDD duration은 성과 일관성이 없다는 신호이며, 높은 평균 수익은 우연(overfitting)일 수 있기 때문입니다.

**RL 보상함수 설계로의 함의**:

- **Sharpe 또는 differential Sharpe ratio**를 직접 reward로 사용하는 방법이 RL 트레이딩 논문에서 표준화되어 가고 있음 (Moody & Saffell, 1998 differential Sharpe). 단순 pnl reward보다 학습 안정성과 실전 성능이 좋다는 보고 다수.
- **Drawdown 페널티**: `reward = pnl - λ · dd_penalty` 형태에서 λ는 hyperparameter. λ=0과 λ=클 때를 비교하는 ablation 실험을 디펜스 자료로.

---

## 졸업 프로젝트용 Chapter 3 체크리스트

이 챕터를 다 읽고 RL env 코드를 점검할 때, **단위 테스트로 만들 수 있는 항목 7개**:

1. **Look-ahead 테스트**: 같은 timestep까지의 데이터만 잘라서 학습/추론 → 결과가 동일한지
2. **정규화 누수 테스트**: `df.iloc[:t+1]` 통계만으로 state 생성되는지
3. **체결 모델 sensitivity**: optimistic vs pessimistic 체결 모델로 동일 strategy 백테스트
4. **거래비용 sensitivity**: 비용 0%, 0.05%, 0.1%, 0.2% 별 Sharpe 비교
5. **Annualization 검증**: 동일 전략을 1시간봉/일봉으로 돌려서 annualized Sharpe 일치 확인
6. **MDD 검증**: 단순 buy-and-hold BTC의 MDD가 알려진 값(2018년 -84%, 2022년 -77%)과 일치하는지
7. **Walk-forward 검증**: train/test 시점 분할 시 미래 데이터 누수 없음

이 7개 통과하지 못하면 RL 결과는 신뢰할 수 없습니다. **Chan의 Chapter 3 메시지를 한 줄로 요약하면: "백테스트 결과가 너무 좋으면 코드를 의심하라."** 이 한 문장이 졸업 프로젝트 디펜스에서 "내 결과가 신뢰할 만하다"를 증명하는 가장 강력한 자세입니다.