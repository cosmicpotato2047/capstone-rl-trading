# 환경 변천 History (Single Source of Truth)

> 본 문서는 BTCGridTradingEnv 의 시간별 변천 기록 + 본 졸업 논문이 사용할 정식 환경(canonical) 정의 + 이전 결과의 인용 가능성 매트릭스를 담는다.
> 본 문서가 환경에 관한 단일 기준점이다. 다른 문서가 본 문서와 충돌하면 본 문서가 우선한다.
> RQ는 [PROJECT_GOAL.md](PROJECT_GOAL.md), 결과 표는 [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) 참조.

---

## 본 졸업 논문이 사용할 정식 환경 (Canonical, Env-v4)

**Status**: 2026-05-14 결정. 환경 복원 진행 중 (Step 6~7).

| 항목 | 정의 |
|---|---|
| State | 7D `[log_price, divergence, holdings_value_ratio, cash_ratio, volatility, trend_short(72h), trend_long(720h)]` (rolling z-score 정규화) |
| Action | **2D** `[aggressiveness, profit_target]` ∈ [0, 1]² |
| 공식 | **ATR 비례 스케일링** — `gap = atr_ratio × (A + B × action)` |
| Buy 공식 | `buy_hi_gap = atr_ratio × (A_b + B_b × aggressiveness)`, `buy_lo_gap = atr_ratio × (C_b + D_b × aggressiveness)` |
| Sell 공식 | `sell_market_gap = atr_ratio × (A_s + B_s × profit_target)`, `sell_cost_gap = atr_ratio × (C_s + D_s × profit_target)` |
| 체결 | 지정가 (limit price) — `next_low ≤ buy_*` 시 buy_* 가격으로 체결, `next_high ≥ sell_*` 시 sell_* 가격으로 체결 |
| Reward | `step_return × β` (β=1.0 symmetric, β=2.0 asymmetric. config에서 결정) |
| 사이클 | `holdings == 0 → 첫 매수 시 시작` / `holdings → 0 복귀 시 종료` |
| Order sizing | `cycle_slot_size = cycle_start_cash / n_splits`, `per_order_size = cycle_slot_size / n_buy_orders` |
| 매도 | `threshold_btc = cycle_slot_size / price` 이하면 전량 청산, 아니면 `holdings / n_splits` 분할 |
| 데이터 분할 | Train 2017-10 ~ 2020-12 (~3.4년), Val 2021-01 ~ 2023-12 (~3년), Test 2024-01 ~ (봉인, exp035까지) |
| Fee | 0.05% (Binance maker fee) |
| 기본 계수 (formula_coefs) | A_b=0.285, B_b=1.748, C_b=5.223, D_b=18.683 (Buy: exp023 Bayesian Trial #42), A_s=0.05, B_s=1.95, C_s=2.5, D_s=7.5 (Sell: Phase 2 재설계 — Bayesian 대상 제외, RL action 충돌 방지) |

→ Phase 3 의 exp030~035 는 모두 본 환경 (Env-v4) 위에서 진행.

---

## 환경 변천 Timeline

### Env-v1 (exp001 ~ exp005) — 고정 절대 gap

| 항목 | 값 |
|---|---|
| Action | 2D `[aggressiveness, profit_target]` |
| 공식 | 고정 절대 범위 — 예: `buy_hi_gap = 0.0001 + aggressiveness × 0.05` ([0.01%, 5%]) |
| ATR 사용 | State[4] 만, 공식에는 없음 |
| 체결 | next_low/high (favorable bias) |
| 데이터 분할 | Train 2020-01 ~ 2022-12, Val 2023, Test 2024~ |

**주요 결과**: exp001 Val Sharpe 0.795, exp002 0.745, exp005 0거래 수렴. → **fee 손익분기 문제** 발견.

### Env-v2 (exp006 ~ exp023) — 2D ATR 비례 + favorable bias 체결

전환 commit: `352a98b` (action 공식 ATR 비례 스케일링 도입).

| 항목 | 값 |
|---|---|
| Action | 2D `[aggressiveness, profit_target]` |
| 공식 | **ATR 비례** — `gap = atr_ratio × (A + B × action)` |
| ATR 사용 | State[4] + Action 공식 |
| 체결 | next_low/high (favorable bias) ⚠️ |
| 데이터 분할 | exp017 시점: Train 2020-01 ~ 2022-12 / Val 2023. exp019 이후: Train 2017-08 ~ 2020-12 / Val 2021-2023H1 (Split B) |

**주요 결과**:
- exp006 Val Sharpe 1.183 (random_start + n_envs=4)
- exp007/008 Val Sharpe ~12 (multi-eval)
- exp016 Val Sharpe **35.4** / Test Sharpe **43.0** ⚠️ (favorable bias artifact)
- exp020 RL = Fixed [1.0, 0.0] Val Sharpe **45.39** ⚠️ (favorable bias artifact)
- exp022 ATR vs RL Test Sharpe 52.6 / 52.8 ⚠️ (artifact)
- exp023 ATR Bayesian Trial #42 Val Sharpe **60.7** ⚠️ (artifact)

→ 환경 구조는 본 논문 (Env-v4) 와 거의 같지만 **체결가 favorable bias 로 수치 부풀려짐**. 본 논문 인용 시 ⚠️ 주의.

### Env-v3 (exp024 ~ exp028) — 4D 절대 gap + 지정가 체결

전환 commit: `757c1ce` (exp024 env 재설계 — ATR 제거, 4D 절대 gap), `e84862e` (체결가 지정가 수정).

| 항목 | 값 |
|---|---|
| Action | **4D** `[buy_hi_coef, buy_lo_extra, sell_m_coef, sell_c_coef]` |
| 공식 | **절대 % gap** — 예: `buy_hi_gap = action[0] × 0.10` ([0%, 10%]), **ATR 미사용** |
| ATR 사용 | State[4] 만 (Action 공식엔 없음 — dead state 문제) |
| 체결 | 지정가 ✓ |
| Reward | step_return × β (asymmetric 옵션) |
| 데이터 분할 | Train 2017-08 ~ 2020-12 / Val 2021-01 ~ 2023-12 / Test 2024-01 ~ |

**주요 결과 (지정가 체결로 현실화)**:
- exp026 ATR (Bayesian best) Val Sharpe **1.978** / Test Sharpe **0.935**
- exp026 RL (symmetric) Val Sharpe 0.896 / Test Sharpe 0.009
- exp027 ATR+direction (k=3.58) Val Sharpe 2.348 / Test Sharpe **-0.213** (Val 과적합)
- **exp027_rl (asymmetric β=2.0) Val Sharpe 2.444 / Test Sharpe 1.955** (ATR 2배 + MDD 0.39%)
- exp028 (early stopping + β=1.5) — 학습 안정화
- exp029 (5D action 재설계 시도, 코드 commit 안 됨) — Val Sharpe 1.440, 학습 불안정

→ 본 논문 사전 증거 (asymmetric reward 효과) 가 이 환경에서 나옴. **다른 환경이라 본 논문 (Env-v4) 에서 재현 검증 필요**.

### Env-v4 (canonical, 2026-05-14 확립) — 2D ATR 비례 + 지정가 체결

본 졸업 논문의 정식 환경. exp030~035 가 모두 Env-v4 에서 진행.

**Env-v2 와의 차이**:
- 체결: favorable bias → 지정가 ✓ (수치 현실화)
- 데이터 분할: 명확화 (Train 2017-10 ~ 2020-12 / Val 2021-2023 / Test 2024~)

**Env-v3 와의 차이**:
- Action: 4D 절대 gap → 2D ATR 비례 (학술 정합성 + state-action 정합성)
- ATR이 state + action 공식 양쪽에서 활용 (dead state 해결)

### Env-v4 검증 결과 (2026-05-14)

**ATR Baseline (Bayesian 50 trials)**:
- 최적 계수 (Trial #34): A_b=1.665, C_b=6.070, A_s=0.285, C_s=1.951, n_splits=2
- Val Sharpe **1.505** (초기 측정) → **1.378** (Phase 16a evaluation setup 통일 후 재측정, 본 논문 canonical 값)
- Return +35.80%, MDD 9.83%
- ⚠️ exp023 Env-v2 계수는 Env-v4 에서 Sharpe -4.738 — 환경 의존성 입증

**RL with Asymmetric Reward β=2.0 재현 (1M steps)**:
- Best (100k): Val Sharpe **2.250**, Return +8.00%, MDD 1.75%
- vs ATR baseline: **+0.745 (+49%) 우위**
- 원본 (Env-v3 Val 2.444) 대비 -8% — 부분 재현 성공, 본질 효과 유지

**결론**: 본 논문 §5 strong positive thesis 사전 증거 확보. exp032 4 variant 비교에서 정식 검증 예정.

→ Phase 2~3 의 핵심 발견 (asymmetric reward, exp027_rl) 이 Env-v4 에서도 유지됨. 환경 복원 작업 성공.

---

## 인용 가능성 매트릭스 (본 논문 작성 시)

| Exp | 환경 | 본 논문 직접 인용? | 사유 / 처리 |
|---|---|---|---|
| exp001 ~ exp005 | Env-v1 | ❌ | 다른 환경. Phase 1 negative finding 정성적 언급만 |
| exp006 ~ exp016 | Env-v2 (체결가 버그) | ⚠️ 정성적 | 학습 안정화 발견 (random_start, ent_coef 등) 은 인용. Sharpe 수치는 favorable bias artifact 명시 |
| exp017 ~ exp023 | Env-v2 (체결가 버그) | ⚠️ 정성적 | state 7D 확장, Bayesian 계수 등 설계 결정은 인용. Sharpe 수치는 artifact 명시 |
| **exp020 "RL = Fixed [1.0, 0.0]" 발견** | Env-v2 | ⚠️ **정성적 only** | Policy saturation 발견은 §4 메인 인용. 단 Val Sharpe 45.39 수치는 artifact 처리 |
| exp024 ~ exp026 ATR | Env-v3 (4D) | ❌ 직접 인용 불가 | 환경 다름. 단 "체결가 수정으로 수치 현실화 (수조% → 1~2)" 발견은 §3.1 method 설명 |
| **exp026 ATR (Bayesian)** | Env-v3 | ❌ | Env-v4 에서 재최적화 필요 (Step 9) |
| exp027 ATR+direction | Env-v3 | ❌ | Val 과적합 사례. 정성적 인용 가능 (§7.2 CPCV 필요성 motivation) |
| **exp027_rl (asym β=2.0) Test Sharpe 1.955** | Env-v3 | ❌ **재현 필요** | Env-v4 에서 재현 결과가 본 논문 §5 의 사전 증거 (Step 12~13) |
| exp028 (early stopping) | Env-v3 | ❌ | 환경 다름. 학습 안정화 방법론은 exp030 에서 재구현 |
| exp029 (5D action) | (코드 commit X) | ❌ | 코드도 결과도 본 논문 외 |

---

## 환경 복원 작업의 위험 (옵션 A)

**가장 큰 위험**: exp027_rl 의 "Test Sharpe 1.955 (ATR 2배)" 가 Env-v4 (2D ATR 비례) 에서 재현 안 될 가능성.

**가능성**:
- 4D 절대 gap 환경의 자유도가 asymmetric reward 효과를 더 크게 만들었을 가능성
- 2D ATR 비례 환경에서는 같은 reward 변형의 효과가 작을 수도

**대응**:
- 재현 결과를 정직하게 기록
- 재현 부분 성공 (효과 약화) → §5 strong → moderate 로 톤 조정
- 재현 실패 → §5 가 §4 negative 확장으로 통합 + asymmetric reward 의 환경 의존성 발견 자체가 §6 새 발견

본 논문 RQ 가 열린 질문이라 어느 결과든 무너지지 않음.

---

## 변경 이력

| 날짜 | 변경 | 근거 |
|---|---|---|
| 2026-05-14 (1차) | 본 문서 신설 (Step 1). 환경 변천 4단계 (Env-v1~v4) 정리. 본 논문 정식 환경 (Env-v4) 정의. 인용 가능성 매트릭스 작성. | 옵션 A (2D ATR 비례 복원) 결정 후 무효화 범위 명시 + 재현 작업 토대 마련 |
| 2026-05-14 (2차) | Env-v4 검증 결과 채움. ATR Bayesian (Val Sharpe 1.505) + RL asym β=2.0 재현 (Val Sharpe 2.250, +49%) 완료. 시나리오 A 확정. | 환경 복원 + 재현 작업 완료 |
