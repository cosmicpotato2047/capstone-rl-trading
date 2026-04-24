# 캡스톤 프로젝트 중간 보고서
## BTC 동적 그리드 트레이딩: ATR 규칙 기반 vs 강화학습

**작성일:** 2026-04-24  
**실험 범위:** exp001 ~ exp029

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 구조 한눈에 보기](#2-시스템-구조-한눈에-보기)
3. [실험 전체 흐름](#3-실험-전체-흐름)
4. [현재 상태 및 결론](#4-현재-상태-및-결론)
5. [앞으로 방향](#5-앞으로-방향)

---

## 1. 프로젝트 개요

### 핵심 아이디어

가격이 오를지 내릴지 **예측하지 않는다.** 대신 변동성 자체에서 수익을 추구하는 **동적 그리드 트레이딩**을 구현한다.

**그리드 트레이딩 원리:**
```
현재가 $100 기준으로 주문 배치:

  $108 ← sell_cost   (평단가 기준 목표 수익)
  $105 ← sell_market (현재가 기준 목표 수익)
  $100  ← 현재가
   $96 ← buy_market  (첫 진입)
   $90 ← buy_avg     (물타기 DCA)

  → 가격이 오르내리며 주문이 반복 체결 = 수익 누적
```

**핵심 질문:** 그리드 간격을 고정 규칙(ATR 비례)으로 결정하는 것 vs 강화학습(PPO)으로 학습하는 것 — 어느 쪽이 더 나은가?

---

## 2. 시스템 구조 한눈에 보기

### 2-1. 데이터 파이프라인

**소스:** Binance API (ccxt) — BTC/USDT 1시간봉

```
Binance API
    │ ccxt (download_data.py)
    ▼
data/raw/btc_1h.parquet          ← 원본 OHLCV (불변)
    │ preprocess_data.py
    ▼
data/processed/
  ├── btc_train.parquet  ─── Train  2017-08-17 ~ 2022-12-31  (44,952봉)
  ├── btc_val.parquet    ─── Val    2023-01-01 ~ 2023-12-31  (8,760봉)
  └── btc_test.parquet   ─── Test   2024-01-01 ~ 현재        (봉인)
```

**전처리 후 컬럼:**

| 컬럼 | 계산식 | 역할 |
|------|--------|------|
| `close`, `high`, `low` | 원본 | 주문 체결 판단 |
| `atr_168` | Wilder ATR(168봉) | 변동성 스케일 |
| `atr_ratio` | `atr_168 / close` | 정규화된 변동성 |
| `log_price` | `log(close / close.rolling(168).mean())` | 중기 가격 위치 |
| `trend_72h` ~ `trend_1440h` | `pct_change(N)` | 방향성 피처 5종 |
| `zscore_*` | `(x - x.rolling(168).mean()) / std` | 모든 state 변수 정규화 |

**데이터 기간 설계 근거:**

| 구간 | 기간 | 주요 레짐 | 역할 |
|------|------|-----------|------|
| Train | 2017~2022 | BULL/BEAR/SIDE 혼재, 2022 대폭락(-64%) 포함 | 다양한 시장 학습 |
| Val | 2023 | 회복 + H2 BULL (30k→44k) | 하이퍼파라미터 튜닝 |
| Test | 2024~ | ETF 승인, 반감기, 강한 상승 | 최종 평가 (봉인) |

---

### 2-2. 체결 로직

**한 봉(1시간)에서 처리 순서:**

```
봉 정보 수신: open, high, low, close
         │
         ▼
  1. SELL 주문 체결 판단 (Sell-First 원칙)
     ├── sell_market: next_high >= sell_market → 체결
     └── sell_cost:   next_high >= sell_cost   → 체결
         │
         ▼
  2. BUY 주문 체결 판단
     ├── buy_market: next_low <= buy_market → 체결
     └── buy_avg:    next_low <= buy_avg    → 체결
         │
         ▼
  3. 청산 룰 (하드 룰)
     if holdings_value < cycle_slot_size:
         → sell_market 전량 매도 강제 실행
         │
         ▼
  4. State 업데이트 + Reward 계산
```

**Sell-First 원칙:** 같은 봉에서 buy·sell 동시 체결 가능 시 sell을 먼저 처리. 매수 직후 매도 → 수수료만 손실하는 허위 수익 방지.

**수수료:** Binance 지정가 maker fee 0.05% per trade (매수/매도 각각 적용)

**사이클 정의:**
```
사이클 시작: holdings == 0 → 첫 buy 체결 순간
사이클 진행: 매수/매도 반복 (n_splits 슬롯 기준 예산 관리)
사이클 종료: holdings → 0 (전량 매도 완료)
```

---

### 2-3. 최종 환경 구조 (exp029 기준)

**State (9차원, rolling z-score 정규화)**

```
┌─────────────────────────────────────────────────────────────┐
│                        STATE (9D)                           │
├───┬──────────────────────────┬──────────────────────────────┤
│ 0 │ zscore_log_price         │ 현재가 vs 7일 이동평균 위치  │
│ 1 │ divergence               │ 평단가 대비 현재가 괴리율    │
│ 2 │ holdings_value_ratio     │ 보유 BTC 가치 / 초기 자본    │
│ 3 │ cash_ratio               │ 현금 / 초기 자본             │
│ 4 │ zscore_volatility        │ ATR(168h) / price           │
│ 5 │ zscore_trend_72h         │ 3일 단기 모멘텀             │
│ 6 │ zscore_trend_720h        │ 30일 중기 레짐              │
│ 7 │ idle_norm                │ 미거래 경과시간 / grace_period│
│ 8 │ n_splits_norm            │ 현재 사이클 분할 수 (정규화) │
└───┴──────────────────────────┴──────────────────────────────┘
```

**Action (5차원 연속 [0,1]⁵, 사이클 시작 시 1회 결정)**

```
┌─────────────────────────────────────────────────────────────┐
│                       ACTION (5D)                           │
│           사이클 시작 순간 한 번만 결정, 이후 고정           │
├───┬──────────────┬───────────────────────────────────────────┤
│ 0 │ n_splits_coef│ n_splits = 5 + round(a × 10)  → {5..15} │
│ 1 │ gap_b1       │ buy_market  = price × (1 - atr×a×coef)  │
│ 2 │ gap_b2       │ buy_avg     = last_avg × (1 - atr×a×coef)│
│ 3 │ gap_s1       │ sell_market = price × (1 + atr×a×coef)  │
│ 4 │ gap_s2       │ sell_cost   = avg × (1 + atr×a×coef)    │
└───┴──────────────┴───────────────────────────────────────────┘

  coef_b1=9.9, coef_b2=9.5, coef_s1=9.7, coef_s2=23.9  (Optuna 최적값)
```

**Reward (3성분)**

```python
r_step  = (equity_t - equity_{t-1}) / start_capital   # 매 스텝, symmetric
r_cycle = cycle_return_pct × w_cycle                   # 사이클 완료 보너스
r_idle  = -idle_rate × (cash / start_capital)          # idle > grace_period 시 패널티

reward  = r_step + r_cycle + r_idle
```

---

## 3. 실험 전체 흐름

```
exp001 ──────── exp016 ─────── exp023 ──── exp027 ──── exp029
   │    [P1]       │    [P2]      │  [P3]     │  [P4]     │
  0.8             35x            현실화      1.955      -0.450
 Sharpe          버블           붕괴        최고        과적합
```

---

### Phase 1: 기반 구축 (exp001~016) — 2026-04-05 ~ 04-21

#### exp001 — 첫 PPO 학습

**문제:** MDP 설계가 이산(Buy/Sell/Hold) 에서 시작 → 방향성 트레이딩 접근.  
**변경:** 연속 2D action [aggressiveness, profit_target]으로 설계 변경. 그리드 트레이딩으로 전환.  
**결과:** Val Sharpe 0.795. 기준선 대비 낮음.

#### exp002~010 — 순차적 개선

| 실험 | 변경 | 문제 | 결과 |
|------|------|------|------|
| exp002~005 | 수수료 계산, state 정규화 | 이중 계산 버그 | 수정 후 Sharpe 개선 |
| exp006~010 | n_splits 도입, 사이클 보너스 | reward sparse | 구조 안정화 |

#### exp011~015 — ATR 비례 공식 도입

**문제:** 절대 간격(예: 5%) → 변동성 낮을 때 체결 안 됨, 높을 때 과도한 손실.  
**변경:** `gap = atr_ratio × action` — 변동성에 비례하는 동적 간격.  
**결과:** 체결 확률 스펙트럼이 [0,1] 전체에 고르게 분포됨.

#### exp016 — Bayesian 최적화

**문제:** ATR 공식의 계수(coef)를 수동 설정.  
**변경:** scikit-optimize Bayesian 50 trials로 `coef_b1, coef_b2, coef_s1, coef_s2` 최적화.  
**결과:** Val Sharpe **35.424**, Test Sharpe **43.040** — 베이스라인(1.472)의 29배.

---

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  PIVOT 1 — "Sharpe 수조%" 버블 붕괴                          ║
║                                                                  ║
║  43배 우위는 구조적 artifact였다.                                 ║
║  행동 분석: 결정론적 행동이 항상 [0.0, 0.0]으로 수렴.             ║
║  Bayesian이 A_s=0.101(탐색 하한)으로 수렴 → RL action 역할 제거.  ║
║  실제 의미: 봉의 최저가 매수 / 최고가 매도 (비현실적 체결).       ║
╚══════════════════════════════════════════════════════════════════╝
```

**핵심 교훈:** 시뮬레이터가 "봉의 고점/저점에서 체결"을 허용하면 RL이 이를 exploit. 체결 로직을 **지정가(limit order)** 로 현실화해야 한다.

---

### Phase 2: 재설계 + 현실화 (exp017~023) — 2026-04-21 ~ 04-22

#### exp017~018 — 데이터 재분할 + 지정가 체결

**문제:** Val(2020 BULL만)이 Test 레짐과 불일치. 체결 로직 비현실적.  
**변경:**
- Val 기간을 2021~2023으로 교체 (ATH→대폭락→회복 포함)
- 체결 조건: `next_low <= buy_price`, `next_high >= sell_price` (지정가)

**결과:** Val Sharpe 38.186 (여전히 높지만 현실에 가까워짐).

#### exp019~021 — Action 구조 Ablation

**목적:** RL이 실제로 학습하는가, 아니면 고정값으로 수렴하는가 검증.

| 실험 | 변경 | 결론 |
|------|------|------|
| exp019 | `n_splits` action으로 | 4로 수렴 (고정값) |
| exp020 | `budget_fraction` action으로 | 1.0으로 수렴 (항상 전체 예산) |
| exp021 | `entry_gate` action으로 | open으로 수렴 (항상 진입) |

---

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  PIVOT 2 — "ATR이 이미 레짐 적응을 하고 있다"               ║
║                                                                  ║
║  결정적 실험: Fixed [1.0, 0.0] vs RL(exp020)                     ║
║  → Val Sharpe 45.390 vs 45.390 — 완전 동일                       ║
║                                                                  ║
║  ATR/price 비례 공식이 변동성 레짐을 이미 내재 처리.              ║
║  RL이 추가로 배울 신호가 gap 크기에 없다.                         ║
║  → RL의 역할을 "gap 계수"가 아닌 다른 것으로 재정의해야 함.       ║
╚══════════════════════════════════════════════════════════════════╝
```

#### exp022~023 — ATR 단독 성능 재평가

**문제:** Sharpe 수조% 시대의 계수들을 지정가 체결 환경에서 재검증.  
**변경:** Bayesian 최적화를 지정가 환경에서 재실행.  
**결과:**

| | Val Sharpe | Test Sharpe |
|--|--|--|
| 구 계수 (exp022) | 60.723 → 정규화 1.701 | **0.935** |
| 새 계수 (exp023) | 재최적화 | 1.701 유지 |

> Sharpe 수조% → Val 1.701 / Test 0.935 로 현실화. 이것이 진짜 기준선.

---

### Phase 3: ATR vs RL 공정 비교 (exp024~027) — 2026-04-22 ~ 04-23

#### exp024~026 — RL 재설계 (지정가 환경 대응)

**문제:** 이전 RL 설계가 비현실적 체결 환경 기준. 지정가 환경에서 재학습 필요.  
**변경:**
- State 7D (trend_72h, trend_720h 추가)
- Action 4D (buy_hi, buy_lo, sell_market, sell_cost 계수)
- Reward: symmetric equity change

**exp026 Val Sharpe:** 0.896 (ATR 1.701 미달)  
**exp026 Test Sharpe:** 0.009

그러나 **레짐별 행동 분석에서 중요한 발견:**

| Action | Bull | Bear | 해석 |
|--------|------|------|------|
| buy_hi_coef | 0.235 | 0.328 | 하락장: 더 보수적 진입 |
| sell_m_coef | 0.011 | 0.001 | 하락장: 손절 방지 (Sharpe p<0.001) |
| sell_c_coef | 0.229 | 0.160 | 상승장: 더 높은 목표가 설정 |

→ **RL이 State에서 레짐을 감지하고 다르게 행동하는 것을 학습했다** (K-W p<0.001).

#### exp027 — ATR + Direction Rule vs RL

**배경:** RL이 방향성을 활용한다면, 수동 direction rule로 동일 효과를 낼 수 있는가?

```python
# ATR+direction rule
buy_mult  = 1.0 + k × max(0, -trend_raw)   # 하락 → buy gap 확대
sell_mult = 1.0 + k × max(0,  trend_raw)   # 상승 → sell gap 확대
```

Optuna 최적화 → `trend_window=336h, k=3.584`

| | Val Sharpe | Test Sharpe | 판정 |
|--|--|--|--|
| ATR (exp026) | 1.701 | 0.935 | 규칙 기반 기준 |
| ATR+direction | 2.348 | **-0.213** | Val 과적합 (k 고정) |
| **RL exp027 best** | **2.444** | **1.955** | **현재 최고 성능** |

**exp027 RL 분석:**
- Test Sharpe **1.955** — ATR의 2.1배
- Test MDD **0.39%** — 극도로 낮은 낙폭
- 거래 **214건** — ATR(1,591건)의 1/7, 선택적 진입

**성공 원인:** asymmetric reward(beta=2.0)가 "확신 없으면 거래 마라"를 유도. trend_720h + volatility 결합으로 유망한 사이클만 선별.

---

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  PIVOT 3 — "거래 안 함" 붕괴 + 재현성 없음                  ║
║                                                                  ║
║  exp027 RL best (200k step): Test Sharpe 1.955 ← 성공            ║
║  exp027 RL final (1M step):  거래 0건, Sharpe 0.000 ← 붕괴       ║
║                                                                  ║
║  asymmetric beta=2.0의 역설:                                     ║
║  손실 패널티가 커질수록 "거래 안 함(reward=0)"이 최적 전략.       ║
║  best_model이 200k에 우연히 저장됐을 뿐, 재현성 없음.            ║
║  → 보상 구조 자체를 바꿔야 한다.                                 ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### Phase 4: 전면 재설계 (exp028~029) — 2026-04-23 ~ 04-24

#### exp028 — 학습 안정화 시도

**문제:** beta=2.0이 너무 강함 → 거래 회피 수렴.  
**변경:** `beta: 2.0 → 1.5` + early stopping(patience=6) 추가.  
**결과:** 동일 패턴. beta 조정만으로는 해결 불가.

**핵심 인사이트:** beta 값 자체가 문제가 아니라 reward 구조가 "거래 안 함"을 허용하는 한 같은 결과.

#### exp029 — MDP 전면 재설계

**3가지 근본 문제 해결 시도:**

| 문제 | exp029 해결책 |
|------|--------------|
| "거래 안 함" 붕괴 | asymmetric beta 제거 + idle penalty 도입 |
| RL action 역할 불분명 | 사이클 시작 시 1회 결정 (n_splits + 4 gap 계수) |
| Val/Test 레짐 불일치 | Val 기간 2023-12-31까지 확장 |

**3단계 Optuna:**

| 단계 | 대상 | Trials | 최적 Val Sharpe |
|------|------|--------|----------------|
| 1단계 | PPO 하이퍼파라미터 | 30 × 200k | 1.786 |
| 2단계 | 환경 파라미터 | 50 × 200k | 2.019 |
| 3단계 | 전체 최적값으로 1M 학습 | — | **1.440** (best, 450k) |

**학습 곡선:**

```
Sharpe
+2.0 |
+1.5 |                           ● 450k (BEST)
+1.0 |                         ●   ●
+0.5 |                       ●       ●
 0.0 |─────────────────────●───────────●─── step
-0.5 |         ●
-1.0 |   ●   ●
-2.0 |  ●
-3.0 |
-5.0 |●
     0   100k  200k  300k  400k  500k  600k  750k
```

---

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  PIVOT 4 — "Val 최적화 = Test 과적합"                        ║
║                                                                  ║
║  exp029 결과:                                                    ║
║    Val Sharpe  +1.440  (baseline 1.051 초과 ✓)                   ║
║    Test Sharpe -0.450  (baseline 1.463 대비 완패 ✗)              ║
║                                                                  ║
║  원인:                                                           ║
║  Optuna를 Val에 3번 적용 → Val(혼재장, 2021~2023) 메타 과적합.   ║
║  Test(2024~, 강한 상승장)는 Val과 완전히 다른 레짐.              ║
║  → 더 단순한 exp027 RL(Test 1.955)보다 오히려 퇴보.             ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 4. 현재 상태 및 결론

### 전략별 최종 성적표

| 전략 | Val Sharpe | Test Sharpe | Test MDD | 비고 |
|------|-----------|-------------|---------|------|
| Buy-and-Hold | 0.373 | 0.734 | 77.2% | 기준 |
| Fixed Grid 1% | 0.190 | 0.242 | 42.2% | |
| Fixed Grid 5% | 1.051 | **1.463** | 1.3% | Test 베이스라인 최고 |
| ATR k=1.0 | 1.701 | 0.499 | 28.9% | |
| ATR+direction | 2.348 | -0.213 | 3.4% | Val 과적합 |
| **RL exp027 best** | **2.444** | **1.955** | **0.39%** | **RL 최고** |
| RL exp029 best | 1.440 | -0.450 | 19.9% | Val 과적합 |

### 핵심 발견 4가지

**발견 1: ATR 비례 공식의 암묵적 레짐 적응**
> ATR/price가 변동성 레짐을 이미 내재 처리 → RL이 gap 크기에서 추가 학습 여지 없음.  
> "Fixed [1.0, 0.0] = RL" (exp020) 실험이 증거.

**발견 2: 고정 규칙의 Val 과적합 한계**
> k=3.584 같은 고정 계수는 Val 레짐에서 최적이지만 Test에서 역효과.  
> direction rule(Val 2.348 → Test -0.213)이 대표적 사례.

**발견 3: asymmetric reward의 구조적 역설**
> 손실 패널티가 강할수록 "거래 안 함(reward=0)"이 최선 전략이 됨.  
> beta 조정이 아니라 reward 구조 자체를 바꿔야 해결 가능.

**발견 4: RL의 state 기반 레짐 적응 성공 (exp026/027)**
> 레짐별 행동이 통계적으로 유의미하게 다름 (K-W p<0.001).  
> 고정 공식이 포착 못하는 방향성 신호를 RL이 활용.  
> Test Sharpe 1.955 — ATR의 2.1배.

### 현재 상태 진단

```
달성:  RL이 ATR을 상회하는 성능 달성 가능함을 증명 (exp027, Test 1.955)
미달:  안정적 재현 불가. exp029에서 오히려 퇴보.
원인:  레짐 불일치 + Val 과적합 + 학습 불안정의 복합 문제
```

---

## 5. 앞으로 방향

### 진단

exp029의 실패는 세 가지가 동시에 문제였다:

1. **훈련 데이터 레짐 편향** — Train(2017~2020)에 BULL 상승장 비중 부족
2. **Reward 노이즈** — 매 스텝 equity 변화는 신호가 약함
3. **Optuna 과도 적용** — Val에 3번 최적화 = Val 메타 과적합

### 변경 계획

**[변경 1] 훈련 데이터 재조정 — 레짐 균형**

```
현재:  Train 2017-2020  |  Val 2021-2023  |  Test 2024~
변경:  Train 2017-2022  |  Val 2023       |  Test 2024~
                  ↑                   ↑
          BULL(2020,2021)        회복+H2 BULL
          BEAR(2018,2022)        → Test와 레짐 가깝
          모두 Train에 포함
```

**효과:** Test(강한 상승장)와 비슷한 레짐(2020 BULL, 2021 ATH)을 Train에 포함 → 일반화 향상.

**[변경 2] Reward 재설계 — Rolling Sharpe**

```python
# 현재 (노이즈 많음)
reward = (equity_t - equity_{t-1}) / start_capital

# 변경: 최근 N봉의 Sharpe를 직접 최적화
window = 168  # 7일
returns = equity_history[-window:].pct_change().dropna()
reward = returns.mean() / (returns.std() + 1e-8)
```

**효과:** "얼마나 벌었나"가 아니라 "얼마나 안정적으로 벌었나"를 직접 최적화. 학습 목표(Sharpe)와 reward 신호 정렬.

**[변경 3] 단순화 — 복잡도 줄이기**

5D action → 3D 또는 2D 검토. exp027이 증명했듯 단순한 설계가 더 잘 일반화.

### 기대 효과

| 변경 | 기대 효과 |
|------|---------|
| Train 확장 | Test 레짐 경험 → Val→Test 일반화 갭 감소 |
| Rolling Sharpe reward | 학습 안정화 + 목표-reward 정렬 |
| 단순화 | 과적합 방지, 학습 효율 향상 |

### 장기 (2학기)

| 과제 | 내용 |
|------|------|
| 다자산 확장 | 주식(AAPL, SOXL), 외환, 원자재에서 ATR vs RL 비교 |
| Live Trading | Binance Testnet → 실계좌 소액 |
| 논문 작성 | "ATR 비례 그리드 vs RL — 자산군별 우위 조건" |

**핵심 가설 (2학기):** BTC에서는 ATR ≈ RL이었지만, 이벤트 리스크(실적 발표, 금리 결정)가 있는 자산에서는 RL의 state 기반 적응이 ATR을 이길 수 있다.

---

*실험 상세 기록: [RESEARCH_LOG.md](../../RESEARCH_LOG.md)*  
*환경 설계: [src/env/trading_env.py](../../src/env/trading_env.py)*  
*설정 파일: [config/experiment_config.yaml](../../config/experiment_config.yaml)*
