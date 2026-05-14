# 프로젝트 진행 계획 — Phase 3 (Reward Design 본격 탐구)

> 작성: 2026-05-13. 개정: 2026-05-14 (RQ 확정, 자산 확장 제외 반영).
> **단일 기준점**: [[PROJECT_GOAL]]. 본 문서는 그 RQ의 실험적 실행 계획.

## RQ 재확인

> **BTC 그리드 트레이딩에서 reward 설계가 RL 정책의 알파에 어떤 영향을 미치는가?
> 특히, 어떤 reward 함수 하에서 RL이 ATR 규칙 기반을 초과하며 (혹은 초과하지 못하며), 그 메커니즘은 무엇인가?**

- 자산: BTC/USDT 1h 단일 (확장 X)
- 메인 챕터: **exp032** (4가지 reward 비교) — RQ에 답하는 핵심 실험
- 보조 챕터: exp030~031 (방법론 기반), exp033~035 (Robustness 검증)
- ⚠️ RQ는 열린 질문 — 사전 증거(exp027_rl)는 가설 H1~H4로 흡수, RQ는 어느 결과든 살아남는 형태

## 현재 상태

- exp029 학습 불안정 보류 (Val Sharpe 1.440 best, 450k peak 후 oscillation)
- exp027_rl asymmetric reward의 발견: Test Sharpe 1.955 (ATR 0.935 대비 2배) — **메인 가설의 사전 증거**
- 중간 보고서 완성, Test set 봉인 유지
- 이론 노트 24개 작성 ([[00_overview]])

---

## 실험 시리즈 개요

| Exp | 목적 | 논문 챕터 | 기간 | 의존성 |
|---|---|---|---|---|
| exp030 | PPO 학습 안정화 | Method §3.3 | 1주 | — |
| exp031 | BC warm-start | Method §3.4 | 1주 | exp030 |
| exp031b | (조건부) CQL + mixed | Method §3.4 | +2~3주 | exp031 결과 부족 시 |
| **exp032a** | **각 variant reward hyperparameter 튜닝 (공정 비교 보장)** | **Method §3.5** | 1~2주 | exp030 |
| **exp032b** | **4 variant full 비교 + effect size 분석** | **§5 Positive finding** | 1~2주 | exp032a |
| **exp032c** | **메커니즘 분석 (counterfactual, SHAP, mediation)** | **§6 Mechanism** | 1주 | exp032b |
| exp033 | Slippage + DR | §7.1 Robustness | 1주 | exp032b best variant |
| exp034 | CPCV 6-fold + DSR | §5, §7.2 | 1~2주 | exp032b |
| exp035 | Test set 봉인 해제 | §7.3 | 당일 | exp034 통과 |

**총 예상**: 8~11주 (조건부 exp031b 시 +2~3주). 학기 일정 안.

---

## exp030 — 학습 안정화 패키지 (1주)

**목적**: exp029의 oscillation 해결. baseline Sharpe 1.5+ 안정 도달.

**변경 사항** ([[policy_gradient_stabilization]]):
1. PPO 하이퍼파라미터 보수화:
   - LR linear schedule: 3e-4 → 1e-5
   - `target_kl = 0.02` (early stop)
   - `ent_coef` annealing: 0.01 → 0.001
   - `clip_range`: 0.367 → 0.2 (exp029 Optuna 값이 너무 큼)
   - `n_steps`: 1024 → 4096

2. Reward 점검 ([[reward_shaping_ng1999]]):
   - `r_idle`을 potential-based 형식으로 재작성
   - `r_cycle` outlier clipping

3. Early stopping 강화:
   - patience=10 (현재 6)
   - min_delta=0.05

**성공 기준**: Val Sharpe ≥ 1.5, 학습 후반(700k+) Sharpe ± 0.3 이내 안정성.

**산출물**: `experiments/exp030_stabilization/`, `config/exp030_config.yaml`, 학습 곡선.

---

## exp031 — BC Warm-Start (Action Bias Init): **Negative Result, 폐기 (2026-05-14)**

**원래 의도**: ATR 정책을 PPO 출발선으로 → 학습 초반 random 정책 낭비 회피.

**시도한 접근 (Action Bias Initialization)**:
- PPO `policy.action_net.bias` 를 `[-10, -10]` (action ≈ 0) → 학습 정체
- bias 약화 `[-3, -3]` 도 학습 정체 (Sharpe 1.526 정확히 동일)

**원인** (SB3 PPO + Box action_space 의 알고리즘 특성):
```
deterministic action = clip(raw_mean, 0, 1)
```
- bias 가 음수면 raw_mean 항상 음수 영역 → clip 으로 action 항상 0
- 학습 mean 을 음수 → 양수로 옮기는 reward signal 약함 → 학습 진행 X

→ 본 환경 / PPO baseline / 후속 exp 에 영향 **없음** (exp030 random init 으로 정상 학습됨 — best Sharpe 1.974).

**결정**: exp031 폐기, exp032 (reward variant 비교) 로 직행.

**본 논문에서의 처리**: §3.4 Method 또는 §8 Discussion 에서 한 줄
> "Action bias initialization 을 BC 의 단순화 형태로 시도했으나 SB3 PPO + Box action_space 의 clipping 특성으로 학습이 진행되지 않음. 정석 BC pretrain 또는 SAC 등 알고리즘 변경이 future work."

**Future work (선택)**:
- 정석 BC pretrain — `imitation` library 활용
- SAC 전환 — squashed Gaussian + auto ent_coef 가 본 환경에 적합할 수도
- 둘 다 본 논문 범위 외

---

## exp031b (조건부) — CQL Warm-Start with Mixed Dataset (2~3주)

**발동 조건**: exp031 결과가 "ATR 출발선(Val Sharpe ~1.98) 근처에서 멈춤". 데이터셋 한계 신호.

**변경 사항** ([[cql_kumar_2020]]):
1. Mixed dataset 구성 (~12,000 trajectory):
   - ATR 변형 5종 (Optuna top-5 trials) — 5000개
   - Fixed Grid 1%/2%/5% — 각 1000개
   - ATR Grid k=0.5/1.0/2.0 — 각 1000개
   - 랜덤 정책 — 1000개
2. CQL pretrain via `d3rlpy.algos.CQL`:
   - α auto-tune (Lagrangian dual)
   - CQL(ρ) for continuous action
   - 100 epochs
3. CQL 정책 → PPO/SAC online fine-tune

**비교 옵션**: IQL (Kostrikov 2021)도 함께 시도하면 풍부한 ablation.

**성공 기준**: exp031 BC 대비 Val Sharpe +0.3 이상.

---

## exp032 — Reward 정식화 비교 (본 논문 메인)

**목적**: 4가지 reward 변형이 RL 정책에 만드는 차이를 측정 + 메커니즘 분석.
**이게 본 논문의 §5 (Positive finding) + §6 (Mechanism) 본체.**

### 4가지 Reward Variant

```python
# 1. Symmetric (baseline, 현 r_step)
r_sym = (equity_t - equity_{t-1}) / start_capital

# 2. Asymmetric (exp027_rl 방식)
delta = (equity_t - equity_{t-1}) / start_capital
r_asym = delta if delta >= 0 else β * delta   # β: 튜닝 대상

# 3. Differential Sharpe Ratio (Moody & Saffell 2001)
A_t = A_{t-1} + η * (R_t - A_{t-1})          # mean EMA
B_t = B_{t-1} + η * (R_t² - B_{t-1})         # 2nd moment EMA
r_dsr = (B_{t-1}*ΔA - 0.5*A_{t-1}*ΔB) / (B_{t-1} - A_{t-1}²)^1.5
# η: 튜닝 대상

# 4. Prospect-theoretic (Kahneman-Tversky 1979)
delta = (equity_t - equity_{t-1}) / start_capital
r_pt = sign(delta) * abs(delta)^α * (1 if delta >= 0 else λ)
# α, λ: 튜닝 대상 (Kahneman 1979 실증치 α=0.88, λ=2.25 시드값)
```

→ **각 variant의 hyperparameter는 exp032a에서 튜닝** (공정 비교 보장).
   [[hyperparameter_parity]] 참조.

---

## exp032a — Variant별 Reward Hyperparameter 튜닝 (공정 비교 보장)

**목적**: 각 variant가 자신의 best hyperparameter로 비교되도록.
임의값으로 비교하면 variant 차이가 아닌 hyperparameter 차이를 측정하게 됨.

**왜 필수**: ATR baseline은 Bayesian-optimized (150 trials). Reward variant도 동일한 정성으로 튜닝되지 않으면 체계적 불공정 비교. [[hyperparameter_parity]] 의 약점 분석 참조.

**탐색 공간**:

| Variant | Hyperparameter | 탐색 범위 |
|---|---|---|
| `sym` | (없음) | — (baseline) |
| `asym` | β | [1.0, 4.0] |
| `dsr` | η (EMA decay) | [1/720, 1/24] (월~일 horizon) |
| `pt` | α, λ | α ∈ [0.5, 1.0], λ ∈ [1.0, 4.0] |

**실험 설계**:
- 각 variant Optuna TPE 30 trials × 200k steps (sym은 skip)
- 동일 PPO hyperparameter (exp030 결과)
- 단일 seed (튜닝 단계)
- Sampler: TPE, Pruner: MedianPruner

**산출물**:
- `experiments/exp032a_reward_tuning/{asym,dsr,pt}/optuna_study.db`
- `config/exp032b_{variant}_config.yaml` — variant별 best hyperparameter 고정 config

**Compute**: 3 variant × 30 trial × 200k = 18M steps (1~2주)

---

## exp032b — Full 비교 + Effect Size 분석 (§5 Positive finding 메인)

**목적**: 확정된 best hyperparameter로 4 variant 본격 비교.

**실험 설계**:
- exp032a best hyperparameter 사용 (sym은 없음)
- 각 variant × 5 random seeds × 1M steps
- 동일 PPO hyperparameter
- 평가: Val set, multi-episode (random_start 5회)

**평가 (강화 버전, [[effect_size_rliable]] 참조)**:

기존 plan + 추가:
1. **기본 통계**: Val Sharpe / MDD / 거래수 / 사이클 (mean ± std)
2. **Effect size**: Cohen's d (모든 pair)
3. **rliable IQM + stratified bootstrap 95% CI**
4. **Probability of Improvement**: P(variant_A > variant_B)
5. **Performance Profile**: 임계값별 성능 분포
6. **BEST (Bayesian)**: variant 쌍별 차이의 95% HDI

**보고 형식**:

| Pair | mean diff | Cohen's d | 95% CI | P(A>B) | BEST HDI |
|---|---|---|---|---|---|
| asym vs sym | +0.42 | 0.85 | [0.31, 0.53] | 0.92 | [0.28, 0.55] |
| pt vs sym | +0.45 | 0.90 | [0.34, 0.56] | 0.94 | [0.31, 0.58] |
| dsr vs sym | +0.18 | 0.41 | [0.05, 0.31] | 0.74 | [0.06, 0.30] |
| asym vs pt | -0.03 | 0.05 | [-0.12, 0.06] | 0.48 | [-0.13, 0.08] |

→ 디펜스에서 "통계 robust한가" 질문 즉답 가능.

**가설 (RQ-2, 가설 정직 유지)**:

| 가설 | 예측 (사전 증거 강도별) |
|---|---|
| H2a | `asym`, `pt`가 `sym`보다 우수 (사전 증거: exp027_rl Test Sharpe 1.955 vs 0.935) |
| H2b | `dsr`은 학습 안정적, 절대 성능은 `asym`/`pt`와 비슷 (사전 증거 없음, 가설) |
| H3a | 우수 variant는 거래 빈도 낮음 (사전 증거: exp027_rl 214건 vs ATR 1591건) |
| H3b | 우수 variant는 bear regime에서 보수적 행동 (사전 증거 없음, 가설) |

⚠️ **RQ 자체는 단정하지 않음** — 결과가 위 가설 예측을 부정해도 그것 자체가 의미 있는 발견 ("Reward 형식은 알파를 좌우하지 않는다, ATR이 이미 흡수했다"). 본 논문의 §4 Negative finding 확장으로 처리.

**산출물**:
- `experiments/exp032b_reward_comparison/{sym,asym,dsr,pt}/seed_{0..4}/`
- 비교표 + 학습 곡선 4-panel plot + Performance Profile plot
- `effect_size_analysis.ipynb`

**Compute**: 4 variant × 5 seeds × 1M = 20M steps (1~2주)

---

## exp032c — Mechanism Analysis (§6 RQ-3 답변)

**목적**: 우위가 발견된다면 **어떻게** reward가 정책 행동을 변화시켰는지 정량화.
없으면 "왜 차이가 없는지" 분석.

**분석 도구** ([[causal_counterfactual_rl]] 참조):

1. **Counterfactual policy comparison**:
   - 동일 trajectory에서 4 variant의 action 차이
   - state feature와의 회귀 → 차이의 driver 식별
   
2. **SHAP feature attribution**:
   - 각 variant best 정책의 SHAP value
   - variant 간 attribution 패턴 비교

3. **Mediation analysis**:
   - Reward → {거래 빈도, 사이클 승률, 보유 기간} → Sharpe
   - 직접/간접 효과 분해

4. **Policy distance**:
   - variant pairwise KL divergence (또는 deterministic L2)
   - t-SNE 시각화

5. **Regime별 행동 분석** (기존 plan 유지):
   - Bull/Bear/Sideways별 action 분포
   - Kruskal-Wallis 통계 검증

**산출물**:
- `notebooks/mechanism_analysis.ipynb`
- §6 figures (counterfactual heatmap, SHAP summary, mediation diagram, policy t-SNE, regime action plot)

**Compute**: 학습 없음, 분석만. 수일.

---

**전체 참조 노트**: [[differential_sharpe_moody2001]], [[prospect_theory]], [[reward_shaping_ng1999]], [[reward_hacking]], [[hyperparameter_parity]], [[effect_size_rliable]], [[causal_counterfactual_rl]]

---

## exp033 — Slippage + Domain Randomization (1주)

**목적**: Sim2Real gap 축소 + 학습 robustness.

**변경 사항** ([[realistic_execution_simulation]], [[curriculum_learning]]):
1. `slippage_rate` 파라미터 (0.02% default) → `trading_env.py:_execute_buy/_sell` 적용
2. Domain randomization:
   - fee_rate ∈ [0.04%, 0.08%]
   - slippage_rate ∈ [0.0%, 0.05%]
3. exp032 best variant로 재학습 (variant 1개로 좁힘)

**성공 기준**:
- 슬리피지 0.02% 환경에서 Val Sharpe ≥ 1.0 (variant best 대비 25% 이내 감소)
- DR 환경에서 학습 수렴

**산출물**: `experiments/exp033_slippage_dr/`, 슬리피지 적용 코드, 비교 plot.

---

## exp034 — CPCV + DSR 통계 검증 (1주, 컴퓨팅 비용 큼)

**목적**: 단일 walk-forward 대신 다중 path 평가로 견고성 확인.

**작업** ([[walk_forward_cv]], [[bayesian_optimization_tpe]]):
1. 6-fold 분할 인프라 (Train+Val 안에서, 2017-08 ~ 2023-12)
2. exp033 best 모델을 모든 fold 조합 (15 paths) 으로 학습+평가
3. Sharpe 분포 분석 (mean, std, 5% CVaR)
4. **DSR 계산** — 다중검정 보정 후 진짜 알파인지

### CPCV 분할 (6-fold)

```
F1: 2017-08 ~ 2018-12  (BTC 초기, ICO 버블 전후)
F2: 2019-01 ~ 2020-04  (회복 + 코로나 충격)
F3: 2020-05 ~ 2021-08  (강세장 시작 ~ 첫 ATH)
F4: 2021-09 ~ 2022-12  (조정 + 폭락)
F5: 2023-01 ~ 2023-06  (회복기)
F6: 2023-07 ~ 2023-12  (안정)

각 trial:
  Train = 4 folds (예: F1,F2,F4,F5)
  Test  = 2 folds (예: F3,F6)
→ C(6,2) = 15 paths
```

### Embargo
- State에 168봉 lookback → 각 fold 사이에 7일 embargo

### 성공 기준
- Mean Sharpe ≥ exp032 결과
- 5% CVaR Sharpe ≥ 0.5
- DSR p-value < 0.05

**참조 노트**: [[walk_forward_cv]], [[bayesian_optimization_tpe]], [[gort_2022_crypto_overfitting]]

**산출물**: `scripts/cpcv_eval.py`, fold 분할 인프라, 15 paths 결과 CSV, DSR 계산 노트북.

---

## exp035 — Test Set 최종 평가 (당일)

**전제**: exp034 CPCV 통과 (mean Sharpe ≥ 1.5, DSR p < 0.05).

**작업**:
1. exp034 best variant + best model로 Test (2024+) 평가 (1회)
2. Sharpe, MDD, Calmar 보고
3. CPCV 분포 mean ± 2σ 안에 들어오는지 확인

**성공 기준**:
- Test Sharpe가 CPCV 분포 안 (일반화 검증)
- Test Calmar > 1.0

**산출물**: `experiments/exp035_test_eval/`, `test_eval_results.yaml`, 최종 figure.

---

## Phase 3 마무리 — 논문 작성 (1~2주)

### 논문 구조 (PROJECT_GOAL의 9장 구조)

| 챕터 | 출처 |
|---|---|
| 1. Introduction | PROJECT_GOAL의 RQ + 가설 |
| 2. Background | [[avellaneda_stoikov_2008]], [[zhang_zohren_roberts_2020]], [[gort_2022_crypto_overfitting]] |
| 3. Method | 환경 + ATR baseline + PPO + 4 reward 변형 |
| 4. Negative finding (RQ-1) | exp020~022 결과 (RESEARCH_LOG) |
| **5. Positive finding (RQ-2)** | **exp032 결과 (메인)** |
| 6. Mechanism (RQ-3) | exp032 행동 분석 |
| 7. Robustness (RQ-4) | exp033 (Slippage), exp034 (CPCV), exp035 (Test) |
| 8. Discussion | Reward 채널 이유 + 한계 + 자산 확장 시사 (단순 언급) |
| 9. Conclusion | — |

### 핵심 인용 (이미 학습 노트로 정리됨)

[[avellaneda_stoikov_2008]], [[zhang_zohren_roberts_2020]], [[gort_2022_crypto_overfitting]],
[[differential_sharpe_moody2001]], [[prospect_theory]], [[reward_shaping_ng1999]], [[ppo_schulman_2017]],
[[walk_forward_cv]], [[bayesian_optimization_tpe]]

### 약점 보강 체크리스트

- [x] 단일 분할 → CPCV (exp034)
- [x] 임의 hyperparameter → DSR (exp034)
- [x] 체결 시뮬레이션 단순화 → Slippage + DR (exp033)
- [x] Reward 임의 선택 → 학술 출처 인용 (exp032)
- [ ] Live trading sim2real gap (Phase 5, 선택)

---

## 자산 확장 (이번 논문 범위 외) ⛔

**ROADMAP의 Phase 4~6 (주식/FX/원자재) 는 본 졸업 논문에 포함하지 않음.**

이유:
- BTC 하나로도 RQ 검증 충분
- 자산별 인프라 재설계 시간 6주+ → 깊이 손해
- 사용자 개인 운용은 별도 진행 (시장수익률 추종)

논문 §8 Discussion에서 "확장 시사점"으로만 짧게 언급.

---

## 위험 및 대응

| 위험 | 가능성 | 대응 |
|---|---|---|
| exp030 안정화 실패 | 중 | SAC로 알고리즘 전환 ([[ddpg_continuous_control]]) |
| exp031 BC가 local optima | 중 | Entropy 일시 증가, exp031b로 분기 |
| exp032 reward 4종 차이 없음 | 낮음 | β/λ/α 추가 sweep |
| CPCV 컴퓨팅 부족 | 높음 | Fold 6→3 축소 또는 fold당 짧은 학습 |
| Test 결과가 CPCV 분포 벗어남 | 중 | regime shift 인정, 추가 분석 + 디펜스에서 한계 명시 |
| Paper trading gap 큼 (Phase 5 시) | 높음 | Slippage 모델 재조정, exp033 재실행 |

---

## 메타 원칙 (이번 정리에서 도출 + 유지)

1. **단일 hyperparameter, 단일 분할, 단일 reward는 위험** → 다중 평가 path.
2. **학술 인용은 디펜스 무기** → "임의 hyperparameter"가 아닌 "Kahneman-Tversky λ".
3. **Sim2Real gap은 본질적** → Paper trading으로 측정 (선택).
4. **Reward design이 알파의 핵심 채널** → 본 논문의 단일 contribution.
5. **자산 확장 전에 BTC 완결성** → 이번 논문은 BTC 완결만.
6. **RQ에서 벗어나는 작업은 사용자 합의 후** (CLAUDE.md 절대 금지 규칙 3).

---

## 참조 노트 인덱스

- [[PROJECT_GOAL]] — 단일 기준점
- [[00_overview]] — 학습 노트 허브

본 계획에서 참조한 학습 노트:
- [[policy_gradient_stabilization]] — exp030
- [[offline_rl_warm_start]], [[cql_kumar_2020]] — exp031, exp031b
- [[differential_sharpe_moody2001]], [[prospect_theory]], [[reward_shaping_ng1999]], [[reward_hacking]] — exp032 이론 출처
- [[hyperparameter_parity]] — **exp032a 공정 비교**
- [[effect_size_rliable]] — **exp032b 통계 검증**
- [[causal_counterfactual_rl]] — **exp032c 메커니즘 분석**
- [[realistic_execution_simulation]], [[sim2real_finance]], [[curriculum_learning]] — exp033
- [[walk_forward_cv]], [[bayesian_optimization_tpe]], [[gort_2022_crypto_overfitting]] — exp034
- [[avellaneda_stoikov_2008]], [[zhang_zohren_roberts_2020]], [[ppo_schulman_2017]] — 논문 배경
- [[risk-adjusted return]] (기존), [[The Dangers of Backtesting]] (기존, López de Prado) — 평가 방법론
