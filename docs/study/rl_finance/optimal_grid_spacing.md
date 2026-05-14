# Optimal Grid Spacing & Volatility Harvesting

> 실용 가이드와 학술 연구의 결합. AS framework의 단순화된 grid 적용.

## 요지

1. **그리드 봇 = 변동성 수확(volatility harvesting)** 메커니즘. 횡보 변동성에서 사이클 수익을 누적.
2. **최적 그리드 간격 = f(변동성, 수수료, 손익분기점, 위험 허용도)**.
3. 너무 좁으면: 잦은 거래 + 수수료에 잠식. 너무 넓으면: 체결 거의 없음 + 사이클 미완료.

## 그리드 트레이딩의 손익 분포

### 횡보장 (Best case)
```
가격 평균 P, 진폭 σ
그리드 간격 g (P × g%)
사이클당 수익: g - 2 × fee
사이클 빈도: ~ σ² / g²   (Brownian motion에서)

기대 수익률 ∝ (σ² / g²) × (g - 2f)
             = σ² × (1/g - 2f/g²)

최대화:  dπ/dg = 0
        → -1/g² + 4f/g³ = 0
        → g* = 4 × f
```

→ **최적 그리드 간격은 약 4배 수수료**. fee=0.05%면 g* ≈ 0.2%.
→ 단, 이건 평균 수익 최대화. Sharpe 최대화는 다름.

### 추세장 (Worst case)
- 가격이 한 방향으로 가면 그리드 봇 손실 누적
- 매수 후 가격 하락 → 미실현 손실 증가
- → **그리드 봇은 short volatility position과 등가**
- → 큰 추세가 발생할 때 무한 손실 가능 (자본 소진까지)

## ATR 비례 그리드 (우리 방식)

```
g(t) = k × ATR(t) / price(t)

장점:
- 변동성 변화에 자동 적응
- 저변동성: 좁은 그리드 → 사이클 빈도 유지
- 고변동성: 넓은 그리드 → 손실 위험 완화
```

vs **고정 % 그리드** (e.g., 1%):
- 단순, 직관적
- 변동성 변화 무시 → 저변동성에서 비효율, 고변동성에서 위험

vs **Geometric grid** (등비 간격):
- BTC처럼 가격 자체가 변동성에 비례할 때 적합
- 현재 우리 시스템과 거의 동치 (ATR ∝ price인 경향)

## 그리드 폭 (Width) 결정

```
total_width = N × g  (N = 그리드 레벨 수)

너무 좁음: 가격이 grid 밖으로 벗어나면 사이클 미완료 (BTC 강세장 2024)
너무 넓음: 자본 분산 → 사이클당 수익 작음

권장: total_width ≈ price × ATR × √(holding_period)
      (시계열 변동성을 cover하는 범위)
```

우리 시스템: N = 2 (buy_market, buy_avg) + 2 (sell_market, sell_cost)
실질적 width = `C_b × ATR` (가장 멀리 있는 buy_avg)

## 학술 vs 실용 차이

| 항목 | Avellaneda-Stoikov (학술) | 실용 그리드 봇 |
|---|---|---|
| Spread 결정 | HJB로 closed-form | 휴리스틱 (ATR × k) |
| Inventory 처리 | reservation price 조정 | 평단가 기준 sell_cost |
| 시간 의존성 | 명시 | 무시 |
| 종료 조건 | T 명시 | 무한 horizon |
| 거래 빈도 | 틱 단위 | 시간봉 |
| Risk aversion | γ 명시 | implicit (PPO 또는 휴리스틱) |

→ 우리 시스템은 **실용 그리드 봇 + RL 학습**. 학술 모델보다 단순.

## 그리드 봇의 본질적 한계

### 1. Short volatility position
- 평균적 수익 + 가끔 큰 손실 (LTCM 분포)
- 추세장에서 자본 소진까지 손실
- 방어: stop-loss, trend filter, 자본 한도 (Kelly criterion)

### 2. 자본 효율성
- N개 그리드에 자본 분산 → 한 사이클 수익이 작음
- 레버리지 사용 시 inventory risk 확대

### 3. Sim2Real gap
- 1h 봉 시뮬레이션 → 호가창 큐 위치 무시
- 실거래에서 partial fill, 우선순위 손실 발생

## 우리 시스템의 grid spacing 결정 방식 진단

```python
# exp026 ATR 최적값 (Bayesian)
A_b = 1.921    # buy_market gap = 1.92 × ATR
C_b = 5.719    # buy_avg gap   = 5.72 × ATR
A_s = 0.688    # sell_market gap = 0.69 × ATR
C_s = 9.673    # sell_cost gap  = 9.67 × ATR
n_splits = 3
```

### 해석
- **buy_market = 1.92 × ATR**: 약 1.92×0.5% = 1% 아래에서 첫 매수 (BTC 1h 평균 ATR ~0.5%)
- **buy_avg = 5.72 × ATR**: 약 2.9% 아래에서 두 번째 매수 (큰 조정 대기)
- **sell_market = 0.69 × ATR**: 약 0.34% 위에서 매도 (빠른 수익 실현)
- **sell_cost = 9.67 × ATR**: 평단가 +4.8% (큰 랠리 전용)

→ **비대칭 구조**: buy_avg는 깊게 (5.7×), sell_cost는 더 깊게 (9.7×)
→ Asymmetric grid가 BTC 강세장 추세에 적응

### 손익분기점 분석
```
수수료 0.05% × 2 (왕복) = 0.1%
slippage 0.02% × 2 = 0.04% (현재 무시 중)
총 비용 = 0.1~0.14%

최소 사이클 수익 (sell_market):
  0.69 × ATR ≈ 0.34% > 0.14%  ✓ 손익분기 통과

하지만:
  최악의 경우 buy_avg에서 매수 + sell_market에서 매도
  순수익 = sell_market - buy_avg = 0.69 - 5.72 = -5.03 × ATR  ← 손실!
```

→ buy_avg에서 매수한 경우 sell_market으로 청산하면 손실
→ sell_cost (평단가 + 9.67×ATR)로 청산해야 수익
→ 평단가가 buy_market + buy_avg 평균 ≈ 3.8 × ATR 아래
→ sell_cost 도달 가격 = 평단가 + 9.67×ATR ≈ +5.9×ATR (현재가 기준)
→ 큰 랠리(+3%) 필요. 그래서 사이클이 길어짐.

## 즉시 액션 (이론 ↔ 실험 연결)

1. **우리 grid의 학술적 정당화 작성**:
   - 4개 주문이 AS framework의 2개 quote를 확장한 것
   - sell_market = AS의 ask quote
   - sell_cost = inventory-aware reservation price 조정
   - buy_market/buy_avg = bid를 두 레벨로 분산 (Cartea-Jaimungal 변형)

2. **Sharpe 최적화 vs 평균 수익 최적화 명시**:
   - 단순 g* = 4f는 평균 최적
   - 우리 Optuna는 Sharpe 최적 → 다른 균형점
   - 학술 비교 시 baseline으로 g* 그리드 사용 가능

3. **Short volatility position 위험 명시 (논문)**:
   - 우리 결과는 BTC 강세장에 유리한 시기
   - 2022 폭락 같은 시나리오에서 어떻게 행동?
   - exp020 5개 episode 평가가 이 위험을 다 잡지 못함

## 백링크

- [[avellaneda_stoikov_2008]] — 학술적 정식 모델
- [[inventory_risk_adverse_selection]] — 그리드의 두 위험
- [[volatility_modeling]] — σ² / ATR 추정
- [[realistic_execution_simulation]] — 실거래 갭

## 출처

- [Best Grid Bot Settings (WunderTrading)](https://wundertrading.com/journal/en/learn/article/best-grid-bot-settings)
- [Grid Trading Strategy 2025 (Zignaly)](https://zignaly.com/crypto-trading/algorithmic-strategies/grid-trading)
- [Finding Optimal Grid Bot Spacing (Gainium)](https://gainium.io/help/grid-step)
- [Bitsgap Grid Bot Guide](https://bitsgap.com/crypto-trading-bot/grid-bot)
