# 공식 정의 — ATR 고정 vs RL 버전

두 시스템은 **동일한 ATR 비례 구조**를 공유한다.  
차이는 오직 하나: 계수를 Bayesian이 고정하느냐(ATR 버전), RL이 동적으로 결정하느냐(RL 버전).

> **본 문서의 위치 (2026-05-14 기준)**:
> - 본 문서는 Phase 2 (ATR vs RL 비교) 시점의 공식 정의.
> - 현 시점(Phase 3)에서는 두 시스템의 비교가 **§4 Negative finding (RQ-1)** 으로 정리되고,
>   본 논문의 메인은 **reward 변형 비교 (RQ-2, exp032)** 로 이동.
> - 본 문서의 ATR 시스템은 Phase 3 실험들의 **고정 baseline** 으로 계속 사용된다.
> - 단일 기준점: [`PROJECT_GOAL.md`](PROJECT_GOAL.md).

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

### 실제 성능 (BTC, exp022 결과)

exp022 결과로 위 시나리오는 답이 나왔다.

| 시스템 | Val Sharpe | Test Sharpe |
|---|---|---|
| ATR 고정 (Bayesian Trial #42) | 45.390 | 41.769 |
| RL exp020 (budget_fraction) | 45.390 | 42.090 |
| Fixed [1.0, 0.0] (RL 수렴 값 고정) | **45.390** (RL과 동일) | 41.769 |

→ **RL과 ATR이 사실상 동등** (Val Sharpe 완전 일치, Test ±0.32 이내).
→ **시나리오 1 (RL ≈ ATR) 채택.**

> **체결가 버그 수정 후 (exp026)**:
> 위 결과는 next_low/next_high 체결의 favorable bias 영향. 지정가 체결로 수정 후:
> - ATR Val Sharpe 1.978 (Test 0.935)
> - RL Val Sharpe 0.896 (Test 0.009)
> - ATR > RL 로 결과 역전.
>
> **그러나 exp027_rl asymmetric reward 도입 후**:
> - RL Test Sharpe 1.955 (ATR 0.935의 2배)
> - **→ Reward 설계가 RL 알파의 핵심 채널**이라는 가설의 사전 증거.
> - 본 졸업 논문의 메인 주제.

---

## Phase 3 — Reward 변형이 결정한다 (현 주제)

Phase 2의 발견 "RL ≈ ATR" 은 **Symmetric reward + ATR 비례 공식의 조합** 에서만 성립.

Phase 3에서는 동일 공식 구조를 유지하되 **reward 함수만 변경** 하여 RL이 ATR을 초과하는 조건을 탐색.

### Reward Variant (exp032 메인 비교)

| 코드 | 정의 | 출처 |
|---|---|---|
| `sym` | `(equity_t - equity_{t-1}) / start_capital` | Phase 1~2 baseline |
| `asym` | `sym` if ≥0 else `β * sym` (β=2.0) | exp027_rl, Kahneman-Tversky 단순화 |
| `dsr` | Differential Sharpe Ratio | Moody & Saffell (2001) |
| `pt` | `sign(x) * abs(x)^α * (1 or λ)` (α=0.88, λ=2.25) | Kahneman & Tversky (1979) |

→ 각 variant × 5 seeds × Val 평가 → CPCV 분포 → DSR 검증.

---

## 두 시스템 비교 요약 (Phase 2 시점 결론, 본 논문 §4 Negative finding 근거)

| | ATR 고정 | RL (Symmetric reward) |
|---|---|---|
| 계수 결정 | Bayesian 1회 최적화, 고정 | 매 스텝 동적 결정 |
| State 활용 | 없음 (공식만) | trend, volatility, position |
| 학습 필요 | 없음 | PPO 1M 스텝 |
| 장점 | 안정적, 해석 가능, 빠름 | 레짐 적응 가능성 |
| 단점 | trend 방향성 무시 | 과적합 위험, 학습 불안정 |
| BTC 성능 (체결가 수정 후) | Test Sharpe 0.935 | Test Sharpe 0.009 (Symmetric) → **1.955 (Asymmetric, exp027_rl)** |

→ Reward 변형으로 RL이 ATR 명확히 초과 가능 → 본 논문 §5 Positive finding.

---

## 다자산 확장 (본 논문 범위 외) ⛔

본 졸업 논문에서는 **자산 확장을 명시적으로 제외**.

이전 검토 (Phase 2 시점, 참고용으로만 보관):

| 자산 | ATR 강점 | RL 기회 | 예상 우위 |
|---|---|---|---|
| BTC | 변동성 자체가 전략 전부 | 없음 (이미 확인) | ATR = RL → Reward 변형 시 RL 우위 |
| 주식 | 일반 변동성 포착 | 실적 발표, 갭, 섹터 레짐 | (가설, 본 논문에서 검증 안 함) |
| 외환 | 일반 변동성 포착 | 금리 결정, 고용지표 | (가설, 본 논문에서 검증 안 함) |
| 금/원자재 | 단기 변동성 | 계절성, 공급 충격 | (가설, 본 논문에서 검증 안 함) |

본 논문 §8 Discussion에서 "확장 시사점"으로만 짧게 언급.

---

## 파일 위치

| 구현 | 경로 |
|---|---|
| 환경 (공통) | `src/env/trading_env.py` |
| ATR 고정 설정 | `config/experiment_config.yaml` |
| RL 실험 (Phase 2) | `experiments/exp020_*`, `experiments/exp022_rl_coef/` |
| ATR 최적화 (Phase 2) | `experiments/exp023_atr_optuna/`, `experiments/exp026_atr_limitfill/` |
| Reward 변형 (Phase 3) | `experiments/exp032_reward_variants/` (예정) |
| 비교 스크립트 | `scripts/eval_atr_test.py`, `scripts/eval_test.py` |
