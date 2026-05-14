# Avellaneda-Stoikov — High-Frequency Trading in a Limit Order Book (2008)

> Avellaneda, M., & Stoikov, S. (2008). _High-Frequency Trading in a Limit Order Book._ Quantitative Finance, 8(3), 217–224.

## 요지

1. **마켓 메이킹 = 그리드 트레이딩의 학술적 조상.** 양방향 지정가를 동시에 걸어 spread를 수익으로 가져가는 구조.
2. **두 가지 핵심 위험**: (a) **Inventory risk** — 보유 포지션이 가격 역행으로 손실, (b) **Adverse selection** — 정보 우위 거래자가 우리 가격에 체결할 때 손실.
3. **최적 해**: bid/ask spread = function(volatility, time, inventory, risk aversion). Closed-form 존재.
4. 그리드 봇은 이 framework의 단순화된 특수 케이스 (시간 의존성 무시, 고정 inventory target = 0).

## 모형 설정

### 가격 동역학
```
dS_t = σ dW_t        (Brownian motion mid-price)
```

### 거래량 동역학
```
M_t^a = Poisson process — 매도(ask) 체결
M_t^b = Poisson process — 매수(bid) 체결

체결 강도(intensity):
  λ^a(δ_a) = A · exp(-k · δ_a)    δ_a = ask - mid (스프레드 절반)
  λ^b(δ_b) = A · exp(-k · δ_b)    δ_b = mid - bid

스프레드 좁힐수록 → 체결 확률 ↑
스프레드 넓힐수록 → 체결 확률 ↓
```

### 마켓 메이커의 가치 함수
```
u(s, x, q, t) = sup E[ -exp(-γ X_T) | S_t=s, X_t=x, Q_t=q ]

s: 현재 mid-price
x: 현금
q: inventory
γ: risk aversion (CARA utility)
T: 종료 시간
```

→ HJB equation을 풀어 closed-form 해.

## 최적 quote (핵심 결과)

```
Reservation price (마켓 메이커의 "공정 가격"):
  r(s, q, t) = s  -  q · γ · σ² · (T - t)

Optimal spread:
  δ^a + δ^b = γ · σ² · (T - t)  +  (2/γ) · log(1 + γ/k)
```

### 결과 해석

1. **Reservation price r은 mid-price와 다르다.**
   - inventory q > 0이면 r < mid (재고 줄이고 싶음 → 매도 favorable)
   - inventory q < 0이면 r > mid

2. **Bid quote**: r - δ^b
3. **Ask quote**: r + δ^a

→ **inventory가 쌓이면 매도 쪽 가격을 낮춰 더 자주 체결시킴**. "skewed quoting"

4. **Spread 크기는 변동성 σ² + 시간 (T-t) + risk aversion γ**의 함수.
   - 변동성 ↑ → spread ↑ (위험 보상 필요)
   - 종료 임박 (T-t → 0) → spread → 상수 (inventory risk 사라짐)
   - γ ↑ → spread ↑ (보수적)

## 우리 그리드 봇과의 비교

| 측면 | Avellaneda-Stoikov | 우리 시스템 |
|---|---|---|
| 거래 빈도 | 고빈도 (틱 단위) | 1h 봉 |
| 주문 개수 | 2 (bid + ask) | 4 (buy_market, buy_avg, sell_market, sell_cost) |
| Inventory 처리 | r 가격 조정 | sell_cost (평단가 기준)로 자동 청산 유도 |
| 변동성 사용 | σ²로 spread 결정 | ATR/price로 gap 결정 |
| 시간 의존성 | 종료 시간 T 명시 | 무한 horizon, T 무시 |
| Risk aversion | γ (CARA) | 명시적 모델 없음 (PPO가 암묵적 학습) |
| Adverse selection | 모형에 직접 없음 (확장 모델에 있음) | 무시 |

→ **우리 시스템은 AS 모델의 "단순화 + 확장":**
- 단순화: 시간 의존성 무시, 단일 inventory target
- 확장: 4개 주문, 평단가 기반 sell_cost (AS에는 없는 개념)

## 우리 시스템의 학술적 위치

```
Avellaneda-Stoikov (2008)
  ↓ 단순화 (시간 의존성 제거)
Cartea-Jaimungal (2014) 변형
  ↓ 더 단순화 + 시간봉 사용
Hummingbot, 3Commas grid bots
  ↓ ATR 비례 + 학습 기반 계수
우리 시스템 (RL + ATR 그리드)
```

→ **학술 논문 작성 시 AS를 reference로 인용하면 그리드 봇의 이론적 정당성 확보.**
→ Naive grid bot보다 우리 시스템이 발전한 측면: ATR 비례 = 변동성 적응 (AS의 σ²-dependence 흉내).

## ATR 비례 공식의 AS적 해석

```
우리: buy_gap = ATR/price × coef

AS:    spread ≈ γ σ² (T - t) + constant
       ≈ γ σ² × time_horizon  (T = ∞이면 발산 → 시간을 average horizon으로 대체)
```

→ ATR ≈ σ √Δt, ATR/price ≈ relative volatility
→ 두 공식 모두 변동성에 비례. 우리는 risk aversion γ를 implicit (계수 coef로 흡수).

## RL과 AS의 결합 (학술적 frontier)

최근 연구들이 AS framework + RL을 결합:

- **Spooner et al. (2018)** _Market Making via RL_ — DDPG로 AS의 closed-form 해 학습
- **Cartea et al. (2015)** _Algorithmic and High-Frequency Trading_ — AS 모델 + 강화학습 확장
- **Sadighian (2019)** _Deep RL for Market Making in Cryptocurrencies_ — BTC/USDT 마켓 메이킹 PPO

→ **우리 프로젝트가 학술적으로 위치할 자리**:
- "AS framework를 단순화 (시간 의존성 제거) + 비대칭 주문 구조 (sell_market/sell_cost) + RL로 계수 학습"

## 우리 프로젝트와의 즉시 활용

1. **논문/디펜스 도입부**:
   - "그리드 트레이딩은 Avellaneda-Stoikov 마켓 메이킹의 단순화된 변형이다"
   - 학술적 뿌리 확립

2. **Inventory skewing 적용 가능**:
   - 현재 sell_cost가 평단가 기준 → 이미 부분적으로 inventory-aware
   - 더 발전: holdings가 많을수록 sell gap 축소 (빨리 청산)
   - exp020 budget_fraction 시도와 연결될 수 있음

3. **Risk aversion γ를 명시적 hyperparameter로**:
   - 현재 γ는 PPO 학습에 implicit
   - DSR + prospect theory ([[prospect_theory]]) 와 결합하면 explicit γ 도입

4. **시간 의존성 무시의 정당화**:
   - 우리는 무한 horizon 가정 → AS의 시간 항 사라짐
   - 단, episode 종료 (TimeLimit)는 인공적 → 학술적 약점

## 백링크

- [[inventory_risk_adverse_selection]] — AS framework의 두 핵심 위험
- [[optimal_grid_spacing]] — AS의 실용적 단순화
- [[hierarchical_rl_trading]] — Market making이 hierarchy의 자연스런 예
- [[volatility_modeling]] — σ² 추정의 발전 (GARCH, realized vol)

## 출처

- [Avellaneda & Stoikov (2008) Cornell PDF](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf)
- [Hummingbot Guide to AS Strategy](https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/)
- [Optimal Market Making (Udit Samani)](https://uditsamani.com/avellaneda-stoikov/)
- [Optiver's Market Making Engine (Medium)](https://medium.com/@navnoorbawa/optivers-3-5b-market-making-engine-avellaneda-stoikov-inventory-optimization-at-scale-a28fede5a85a)
- [Optimal HFT Market Making (Stanford)](https://stanford.edu/class/msande448/2018/Final/Reports/gr5.pdf)
