# 공식 정의 — ATR 고정 vs RL 버전

두 시스템은 **동일한 ATR 비례 구조**를 공유한다.  
차이는 오직 하나: 계수를 Bayesian이 고정하느냐(ATR 버전), RL이 동적으로 결정하느냐(RL 버전).

---

## 공통 구조

### 기본 변수

```python
price     = 현재 봉 close
atr_ratio = ATR(168) / price        # 변동성 척도 (≈ 0.002 ~ 0.020)
avg_price = 보유 중 평단가 (미보유 시 price fallback)
```

### 주문 체결 방식

```
다음 봉 high/low 기준:
  next_low  <= buy_hi  → next_low  가격으로 매수 체결
  next_low  <= buy_lo  → next_low  가격으로 매수 체결
  next_high >= sell_market → next_high 가격으로 매도 체결
  next_high >= sell_cost   → next_high 가격으로 매도 체결

sell 우선 원칙: 같은 봉에서 sell과 buy 동시 조건 충족 시 sell 먼저 처리
```

### 주문 가격 계산

```python
buy_hi      = price     * (1 - buy_hi_gap)
buy_lo      = price     * (1 - buy_lo_gap)
sell_market = price     * (1 + sell_market_gap)
sell_cost   = avg_price * (1 + sell_cost_gap)
```

### 예산/사이클 구조

```python
# 사이클 시작 (holdings=0 → 첫 매수 체결)
cycle_slot_size  = cash / n_splits           # n_splits=4
per_order_size   = cycle_slot_size / n_buy_orders  # n_buy_orders=2
cycle_budget_remaining = cash

# 사이클 종료 (holdings → 0)
# 통계 기록 후 다음 사이클 대기
```

---

## 시스템 A — ATR 고정 (규칙 기반)

계수를 Optuna Bayesian 최적화로 1회 확정, 이후 고정 사용.

### 공식

```python
# Buy side (계수 고정)
buy_hi_gap  = atr_ratio * A_b   # = atr_ratio * 0.285
buy_lo_gap  = atr_ratio * C_b   # = atr_ratio * 5.223

# Sell side (계수 고정)
sell_market_gap = atr_ratio * A_s   # = atr_ratio * 0.05
sell_cost_gap   = atr_ratio * C_s   # = atr_ratio * 2.5
```

### 최적 계수 (BTC 1h, Optuna Trial #42 기준)

| 계수 | 값 | 의미 |
|---|---|---|
| A_b | **0.285** | 매수 상단: 현재가 -0.285 ATR |
| C_b | **5.223** | 매수 하단: 현재가 -5.223 ATR |
| A_s | **0.05** | 매도 즉시: 현재가 +0.05 ATR (매우 타이트) |
| C_s | **2.5** | 매도 원가: 평단가 +2.5 ATR |

### 왜 A_s=0.05가 최적인가

빠른 매도(작은 매도 간격) → 짧은 사이클 → 복리 사이클 수 극대화.
사이클당 수익 0.08~0.14%, 하루 수십 사이클 → 복리 누적이 핵심.

### 성능 (BTC)

| 데이터셋 | Sharpe | MDD |
|---|---|---|
| Val (2021~2023H1) | 45.390 | 1.94% |
| Test (2023H2~2026) | 41.769 | 1.26% |

**Action 없음. 매 스텝 동일한 계수 사용.**

---

## 시스템 B — RL 버전 (학습 기반, exp022 예정)

동일한 ATR 비례 구조를 유지하되, 계수를 RL이 매 스텝 동적으로 결정.

### Action Space

```python
action_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

action[0] = aggressiveness  # buy gap 계수 결정
action[1] = profit_target   # sell gap 계수 결정
```

### 공식

```python
# Buy side (RL이 계수 결정)
buy_hi_gap  = atr_ratio * (0.05 + aggressiveness * 0.55)  # [0.05, 0.60] × ATR
buy_lo_gap  = atr_ratio * (1.0  + aggressiveness * 9.0)   # [1.0, 10.0] × ATR

# Sell side (RL이 계수 결정)
sell_market_gap = atr_ratio * (0.01 + profit_target * 0.29)  # [0.01, 0.30] × ATR
sell_cost_gap   = atr_ratio * (0.5  + profit_target * 5.5)   # [0.5,  6.0] × ATR
```

### ATR 고정 최적값이 RL 범위 어디에 위치하는가

| 계수 | ATR 최적값 | RL 범위 | 대응 action 값 |
|---|---|---|---|
| buy_hi (A_b=0.285) | 0.285 | [0.05, 0.60] | aggressiveness ≈ **0.43** |
| buy_lo (C_b=5.223) | 5.223 | [1.0, 10.0] | aggressiveness ≈ **0.47** |
| sell_market (A_s=0.05) | 0.05 | [0.01, 0.30] | profit_target ≈ **0.14** |
| sell_cost (C_s=2.5) | 2.5 | [0.5, 6.0] | profit_target ≈ **0.36** |

ATR 최적값이 RL 범위의 중간(0.14~0.47)에 위치 → RL이 ATR 최적을 찾거나 더 나은 값을 탐색 가능.

### RL이 적응해야 하는 것

ATR 비례 구조를 유지하면서도, RL이 같은 ATR 수준에서 **어떤 계수 배율을 선택할지**를 state(trend, volatility, position)를 보고 결정해야 한다.

- 하락장 (trend_1w 음수): 작은 aggressiveness (낮은 매수 빈도) + 작은 profit_target (빠른 매도)?
- 상승장 (trend_1w 양수): 큰 aggressiveness (적극 매수) + 큰 profit_target (추세 편승)?
- 횡보 (trend_1w ≈ 0): 중간값?

ATR 버전은 이 판단을 하지 않는다. 변동성(ATR 크기)만 반영하고 추세 방향은 무시.

### 예상 성능 (BTC)

미지수. 가능한 시나리오:
- **RL ≈ ATR**: BTC에서도 ATR 최적이 이미 최선 → 계수 고정이 충분
- **RL > ATR**: trend 신호를 활용해 ATR 고정보다 나은 계수 조합 발견
- **RL < ATR**: 학습 불안정, ATR 최적 계수를 초과/미달

---

## 두 시스템 비교 요약

| | ATR 고정 | RL |
|---|---|---|
| 계수 결정 | Bayesian 1회 최적화, 고정 | 매 스텝 동적 결정 |
| State 활용 | 없음 (공식만) | trend, volatility, position |
| 학습 필요 | 없음 | PPO 1M 스텝 |
| 장점 | 안정적, 해석 가능, 빠름 | 레짐 적응 가능성 |
| 단점 | trend 방향성 무시 | 과적합 위험, 학습 불안정 |
| BTC 성능 | Test Sharpe 41.769 | Test Sharpe 42.090 (≈ 동일) |

---

## 다자산 확장 시 예상 차이

| 자산 | ATR 강점 | RL 기회 | 예상 우위 |
|---|---|---|---|
| BTC | 변동성 자체가 전략 전부 | 없음 (이미 확인) | ATR = RL |
| 주식 | 일반 변동성 포착 | 실적 발표, 갭, 섹터 레짐 | **RL 가능성** |
| 외환 | 일반 변동성 포착 | 금리 결정, 고용지표 | 중간 |
| 금/원자재 | 단기 변동성 | 계절성, 공급 충격 | **RL 가능성** |

공식 계수(A_b, C_b, A_s, C_s)는 자산별로 Bayesian 재최적화 필요.

---

## 파일 위치

| 구현 | 경로 |
|---|---|
| 환경 (공통) | `src/env/trading_env.py` |
| ATR 고정 설정 | `config/experiment_config.yaml` |
| RL 실험 (exp022) | `experiments/exp022_rl_coef/` (예정) |
| 비교 스크립트 | `scripts/compare_atr_vs_rl.py` (예정) |
