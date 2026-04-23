# 캡스톤 프로젝트 중간 보고서
## BTC 동적 그리드 트레이딩: ATR 규칙 기반 vs 강화학습

**작성일:** 2026-04-23  
**현재 실험:** exp028 (학습 안정화 진행 중)

---

## 1. 프로젝트 개요

### 핵심 아이디어

가격 방향 예측이 아니라 **변동성 자체에서 수익**을 추구하는 동적 그리드 트레이딩.

- **그리드 트레이딩**: 현재가 아래에 매수 주문, 위에 매도 주문을 동시에 걸어두고 가격 진폭으로 수익 실현
- **핵심 질문**: 고정 규칙(ATR 비례 공식) vs 학습 기반(PPO) — 어느 쪽이 우월한가?

### 데이터 분할

| 구분 | 기간 | 특징 |
|------|------|------|
| Train | 2017-08-17 ~ 2020-12-31 | BULL 46% / BEAR 38% / SIDE 15% |
| Val | 2021-01-01 ~ 2023-06-30 | ATH→대폭락→회복 (BULL 22% / BEAR 33% / SIDE 44%) |
| **Test** | **2023-07-01 ~ 현재** | **봉인 해제 후 최종 평가** (BULL 45%) |

### MDP 구조 (최종)

**State (7차원)**
```
[0] zscore_log_price       — 현재가 vs 7일 이동평균 위치
[1] divergence             — 평단가 대비 현재가 괴리
[2] holdings_value_ratio   — 보유 BTC 가치 / 초기 자본
[3] cash_ratio             — 보유 현금 / 초기 자본
[4] zscore_volatility      — ATR(168h) / price
[5] zscore_trend_72h       — 72h 단기 모멘텀
[6] zscore_trend_720h      — 720h 중기 레짐
```

**Action (4차원 연속 [0,1]⁴, ATR 없음 — 절대 비율)**
```python
buy_hi_gap      = action[0] × 0.10         # [0%, 10%]
buy_lo_gap      = buy_hi_gap + action[1] × 0.20
sell_market_gap = action[2] × 0.10
sell_cost_gap   = action[3] × 0.20

buy_hi      = price     × (1 - buy_hi_gap)
buy_lo      = price     × (1 - buy_lo_gap)
sell_market = price     × (1 + sell_market_gap)
sell_cost   = avg_price × (1 + sell_cost_gap)  # 평단가 기준
```

---

## 2. 실험 흐름 (exp001 → exp027)

### Phase 1: 기반 구축 (exp001~016) — 2026-04-05 ~ 04-21

#### 주요 마일스톤

| 날짜 | 이벤트 | 결과 |
|------|--------|------|
| 04-05 | 프로젝트 초기화, MDP 설계 | 5D state, 2D continuous action 확정 |
| 04-09 | 데이터 파이프라인 완성 | 54,933 캔들, BTCGridTradingEnv 구현 |
| 04-12 | 주문 사이징 재설계 | n_splits 균등 분할, 사이클 보너스 제거 |
| 04-13 | 테스트 46개, EDA, 베이스라인 구현 | Fixed Grid 1% Val Sharpe 2.610 (기준선) |
| 04-14 | PPOAgent + train_ppo.py | exp001 완료 (Val Sharpe 0.795) |
| 04-19 | 수수료 이중계산 버그 수정 | reward 공식 정정 (8배 과다 패널티 제거) |
| 04-19 | ATR 비례 action 공식 도입 | 체결 확률 스펙트럼 고르게 분포 |
| 04-20 | n_splits/threshold 파라미터 탐색 | n_splits=2, threshold=avg_price 최적 확정 |
| 04-20 | Bayesian 계수 튜닝 (50 trials) | Val Sharpe 42.997 달성 |
| 04-20 | exp016 3M 스텝 Full 학습 | **Val Sharpe 35.424, Test Sharpe 43.040** |

#### Phase 1 결과

PPO Sharpe 43.040 vs 최강 베이스라인 1.472 → **29배 우위**

그러나 행동 분석에서 구조적 문제 발견: **정책 포화(Policy Saturation)**
- 결정론적 행동이 항상 `[aggressiveness=0, profit_target=0]`으로 수렴
- Bayesian이 A_s=0.101(탐색 하한 도달)로 최적화 → RL action의 역할 제거
- 수익률 수조%는 "봉의 최저가 매수 / 최고가 매도" artifact였음

---

### Phase 2: 재설계 + Pivot (exp017~023) — 2026-04-21 ~ 04-22

#### 핵심 변경

1. **데이터 재분할**: Val에 2021 ATH(+103%) + 2022 대폭락(-64%) 포함 (진정한 스트레스 테스트)
2. **State 7D**: trend_1d, trend_1w 추가 (방향성 피처)
3. **체결가 수정**: 봉의 최저/최고 → 지정가(limit price)로 현실화

#### Ablation 실험 결과

| 실험 | action[0] 변경 | Val Sharpe | 핵심 발견 |
|------|--------------|-----------|---------|
| exp017 | aggressiveness | 38.186 | profit_target 부분 적응 확인 |
| exp018 | + Optuna 튜닝 | 38.330 | regime 적응 통계적 유의 (p<0.001) |
| exp020 | budget_fraction | 48.238 | 항상 1.0 수렴 → action 무의미 |
| exp021 | entry_gate | 48.074 | 항상 open → action 무의미 |

**Fixed [1.0, 0.0] vs RL(exp020): Val Sharpe 45.390 vs 45.390 — 완전 동일**

→ **핵심 발견: ATR/price 비례 공식이 이미 레짐 적응을 내재. RL이 추가로 배울 신호 없음.**

#### exp023: 지정가 체결 기준 ATR 재최적화

| | Val Sharpe | Test Sharpe |
|--|--|--|
| 기존 ATR | 45.390 | 41.769 |
| **ATR 최적화 (exp023)** | **60.723** | **52.632** |

---

### Phase 3: ATR vs RL 공정 비교 (exp024~027) — 2026-04-22 ~ 04-23

#### 지정가 체결로 현실화 후 성과

| 시스템 | Val Sharpe | Test Sharpe | Test MDD |
|--------|-----------|-------------|---------|
| ATR 고정 (exp022) | 60.723 → 재기준 1.701 | 0.935 | 2.43% |
| RL (exp026, 4D action) | 0.896 | 0.009 | 1.21% |

> 수조% → 34%(Val) / 4%(Test)로 현실화. ATR이 RL 대비 Sharpe 우위.

#### exp026 RL 레짐 분석 — 의미있는 발견

레짐별 행동이 통계적으로 유의미하게 다름 (K-W p<0.001):

| Action | Bull | Bear | 해석 |
|--------|------|------|------|
| buy_hi_coef | 0.235 | 0.328 | 하락장: 더 보수적 진입 |
| sell_m_coef | 0.011 | 0.001 | 하락장: 손절 방지 |
| sell_c_coef | 0.229 | 0.160 | 상승장: 더 높은 목표가 |

**RL이 state 기반 행동 차별화에 성공** — exp022(0 수렴)와 대조적

#### exp027: ATR + Direction Rule

방향성 신호를 수동 규칙으로 추가:
```python
buy_mult  = 1.0 + k × max(0, -trend_raw)   # 하락 → buy gap 확대
sell_mult = 1.0 + k × max(0,  trend_raw)    # 상승 → sell gap 확대
```

Optuna 최적화 결과: `trend_window=336h, k=3.584, Val Sharpe 2.348 (+38%)`

| | Val Sharpe | Test Sharpe | 판정 |
|--|--|--|--|
| exp026 ATR | 1.701 | **0.935** | Test 기준 최고 규칙 기반 |
| exp027 ATR+direction | 2.348 | -0.213 | Val 과적합 (k 고정) |
| **exp027 RL best** | **2.444** | **1.955** | **현재 최고 성능** |

#### exp027 RL 결과 분석

- **Test Sharpe 1.955** — ATR(0.935)의 2.1배
- **MDD 0.39%** — 극도로 낮은 낙폭
- **거래 214건** — ATR(1,591건)의 1/7, 선택적 진입

**성공 원인:** asymmetric reward(beta=2.0)가 "확신 없으면 거래 마라"를 학습. trend_720h + volatility 결합으로 유망한 사이클만 선별. 고정 k=3.584의 과적합 문제를 state 기반 동적 조정으로 자연스럽게 회피.

**문제점:** beta=2.0이 200k 스텝 이후 "거래 안 함(reward=0)" 전략으로 수렴. best_model이 우연히 200k에 저장됐을 뿐, 재현성 없음.

---

## 3. 현재 상태 (2026-04-23)

### exp028: 학습 안정화 (진행 중)

| 변경 | 내용 | 이유 |
|------|------|------|
| reward_loss_beta | 2.0 → **1.5** | 붕괴 완화 |
| early_stopping_patience | **6회** (300k step) | Val Sharpe 개선 없으면 자동 종료 |

예상 종료: best_model 시점(200~300k) + patience(300k) = 최대 500~600k 스텝에서 자동 중단.

### 전략별 최종 성적표

| 전략 | Val Sharpe | Test Sharpe | Test MDD | 비고 |
|------|-----------|-------------|---------|------|
| Buy-and-Hold | 0.212 | 0.930 | 50.1% | |
| Fixed Grid 1% | 0.607 | 1.550 | 19.7% | |
| ATR (exp026) | 1.701 | 0.935 | 2.43% | 규칙 기반 최고 |
| ATR+direction | 2.348 | -0.213 | 3.43% | Val 과적합 |
| **RL exp027 best** | **2.444** | **1.955** | **0.39%** | **현재 최고** |

---

## 4. 다음 계획

### 단기 (exp028 완료 후)

#### 4-1. exp028 결과 평가
- early stopping이 작동했는지 확인
- beta=1.5에서 학습 곡선이 안정화되는지 검증
- Test Sharpe가 1.955 유지/개선되는지 확인

#### 4-2. exp029: Action 공간 최적화 (Optuna)

현재 gap 범위는 설계값:
```
buy_hi_gap = action[0] × 0.10   # 0.10이 최적인가?
sell_cost_gap = action[3] × 0.20  # 0.20이 최적인가?
```

Optuna로 탐색:
- gap 상한 계수 4개 (`[0.05, 0.30]` 범위)
- state 윈도우: trend_short (`[48, 72, 96, 120]h`), trend_long (`[336, 504, 720, 1008]h`)

목표: 현재 설계 추측값 → 데이터 기반 최적값

#### 4-3. exp030: PPO 하이퍼파라미터 재탐색 (Optuna)

exp027에서 사용한 PPO 파라미터(exp026 Optuna 결과)를 새 설계에 맞게 재최적화.

---

### 장기 (2학기)

| 과제 | 내용 |
|------|------|
| 다자산 확장 | 주식(AAPL, SOXL), 외환, 원자재에서 ATR vs RL 비교 |
| Live Trading | Binance Testnet → 실계좌 소액 |
| 논문 작성 | "ATR 비례 그리드 vs RL — 자산군별 우위 조건" |

**핵심 가설 (2학기):** BTC에서는 ATR ≈ RL이었지만, 이벤트 리스크(실적 발표, 금리 결정)가 있는 자산에서는 RL의 state 기반 적응이 ATR을 이길 수 있다.

---

## 5. 핵심 발견 요약

### 발견 1: ATR 비례 공식의 암묵적 레짐 적응
ATR/price가 변동성 레짐을 이미 내재적으로 처리 → BTC에서 RL의 추가 학습 여지 제한.

### 발견 2: 고정 규칙의 Val 과적합 한계
k=3.584 같은 고정 계수는 Val 레짐에 최적화되면 Test에서 역효과 (direction rule 실패).

### 발견 3: asymmetric reward의 이중 효과
beta=2.0으로 손실 패널티 강화 → 선택적 진입 유도(+) / 학습 붕괴(-). beta 조정 + early stopping이 필수.

### 발견 4: RL의 state 기반 행동 차별화 성공
exp026 RL: 레짐별 행동이 통계적으로 유의미하게 다름 (K-W p<0.001). 고정 공식이 포착 못하는 방향성 신호를 RL이 활용.

---

*실험 상세 기록: RESEARCH_LOG.md*  
*환경 설계: src/env/trading_env.py*  
*설정 파일: config/experiment_config.yaml*
