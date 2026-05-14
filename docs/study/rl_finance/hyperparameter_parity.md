# Hyperparameter Parity — Variant 간 공정 비교의 전제

> Cawley & Talbot (2010). _On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation._ JMLR.
> Henderson et al. (2018). _Deep Reinforcement Learning that Matters._ AAAI.
> Andrychowicz et al. (2020). _What Matters in On-Policy RL? A Large-Scale Empirical Study._

## 요지

1. **공정 비교의 본질적 요구사항**: 두 알고리즘/reward를 비교할 때, **각각 자신의 최적 hyperparameter** 로 비교해야 한다. 한쪽만 튜닝되면 결과는 hyperparameter 차이 vs 알고리즘 차이를 구분 못 함.
2. **Nested CV** 가 표준 해법: 외부 fold로 generalization 추정, 내부 fold로 hyperparameter 튜닝.
3. **DRL에서는 더 심각**: hyperparameter 민감도 + seed variance + reward shape이 결합되어 fair comparison이 어려움. Henderson 2018이 정식 비판.

## 핵심 문제 (Selection Bias)

```
Naive: 모든 algorithm을 한 hyperparameter 세팅으로 비교
       → 그 세팅이 algorithm A에 최적이면 A 부풀려짐

Selection bias: 한 dataset에서 hyperparameter 선택 + 같은 dataset에서 평가
       → 평가 점수가 generalization 추정 아닌 in-sample 최적치
```

## Nested Cross-Validation (정석)

```
Outer loop (generalization 추정):
  Fold 1, 2, ..., K
  
Inner loop (각 outer fold 안에서 hyperparameter 튜닝):
  내부 K-fold CV로 hyperparameter 선택
  
보고:
  Outer fold별 best hyperparameter + outer fold 평가 점수
  → Generalization 추정 = mean(outer scores)
  → Hyperparameter 변동성도 볼 수 있음 (fold마다 다를 수 있음)
```

→ Hyperparameter 선택 자체가 학습 procedure의 일부로 평가됨.
→ "이 알고리즘의 typical 성능" 을 정직하게 추정.

## DRL에서의 fair comparison (Henderson 2018)

### 문제 사례
> 같은 PPO 코드, 다른 hyperparameter → Sharpe 차이 2배.
> 같은 hyperparameter, 다른 random seed → Sharpe 차이 1.5배.
> 두 알고리즘 비교한다는 게 사실은 hyperparameter 비교일 수 있음.

### 권고
1. **각 알고리즘마다 동일한 compute budget 으로 hyperparameter 튜닝**
2. **최소 5~10 random seed** (3개는 부족)
3. **Sample efficiency vs final performance 분리 보고**
4. **Statistical test + effect size 동반**

## Andrychowicz 2020 — "What Matters in On-Policy RL"

50+ design choice를 체계적으로 ablation. 결론:
- **가장 중요**: advantage normalization, value clipping, large batch
- **중간**: GAE λ, learning rate, n_epochs
- **거의 무관**: optimizer choice, exact entropy bonus 값

→ 의미: PPO를 비교하려면 위 "가장 중요" 항목을 모든 variant에 똑같이 적용. 그 외는 variant별로 튜닝 가능.

## 우리 프로젝트 직결 — exp032 공정 비교 보장

### 현재 plan의 약점

```
exp032 (project_continuation_plan.md):
- 4 variant × 5 seeds × 1M steps
- 동일 PPO 하이퍼파라미터 (exp030 결과 사용)
- Variant별 reward hyperparameter: 임의값
  - asym: β = 2.0
  - dsr: η = 1/168
  - pt: α = 0.88, λ = 2.25
```

**문제**:
1. β=2.0이 asym 최적인지 모름 (Kahneman 1979가 2.25 보고, 우리 환경에서는?)
2. dsr의 η 선택 임의 (1주? 1일? 1개월?)
3. pt의 α, λ는 인간 행동 실증치 → RL에 그대로 적용 정당화 약함
4. ATR baseline은 Bayesian-optimized (150 trials), variant는 임의값 → **체계적 불공정**

→ exp032 결과가 "ATR > variant" 로 나와도 그게 진짜 negative finding인지 단순 hyperparameter 부족인지 구분 불가.

### 보강 제안 (exp032 단계 분리)

```
exp032a (variant별 hyperparameter 튜닝):
  각 variant에 대해 Optuna 30 trials × 200k steps
  Reward hyperparameter 탐색:
    sym: 없음 (baseline)
    asym: β ∈ [1.0, 4.0]
    dsr: η ∈ [1/720, 1/24] (월~일 EMA)
    pt: α ∈ [0.5, 1.0], λ ∈ [1.0, 4.0]
  → 각 variant의 best hyperparameter 확정

exp032b (확정 hyperparameter로 full 비교):
  각 variant × 5 seeds × 1M steps
  동일 PPO hyperparameter (exp030 결과)
  Reward hyperparameter는 exp032a best
  → Sharpe / MDD / 행동 통계 + Effect size 분석

exp032c (메커니즘 분석):
  exp032b best 정책으로 counterfactual + SHAP + mediation
  → RQ-3 답변
```

### Compute 비용
- exp032a: 4 variant × 30 trial × 200k = 24M steps (단 sym은 skip → 18M)
- exp032b: 4 variant × 5 seed × 1M = 20M steps
- exp032c: 분석만, 학습 없음
- **총 38M steps** (이전 plan 20M의 약 1.9배)

### 대안: 절약 모드

만약 compute가 부족하면:
- exp032a hyperparameter sweep을 작게 (15 trial × 200k = 9M)
- 또는 baseline literature 값으로 시작하고 한 axis만 sweep (β만, λ만 등)

### Hyperparameter 공정성의 ATR baseline 보존

ATR baseline (exp026 best, 150 Bayesian trials) 는 이미 강하므로 그대로 사용.
다만 논문에 "ATR baseline은 150 trials Bayesian, variant는 30 trials × 4 = 동일 compute" 같은 형태로 비교 가능성 명시.

## Nested CV와의 결합 (선택, exp034와 통합)

```
exp034 (CPCV 6-fold) 안에서:
  각 fold마다:
    Inner: Reward hyperparameter 탐색 (작은 fold)
    Outer: 그 hyperparameter로 평가
  → 각 fold의 best hyperparameter가 다를 수 있음
  → "hyperparameter 자체의 fold-by-fold 안정성" 도 평가
```

→ 가장 엄밀하나 compute 비용 큼 (15 paths × inner sweep).
→ 본 논문에선 exp032a를 단일 split에서 한 후 exp034에서 fixed hyperparameter로 CPCV 적용하는 게 현실적.

## 본 논문 활용

| 챕터 | 활용 |
|---|---|
| §3.5 Evaluation Methodology | Nested CV / hyperparameter parity 인용 (Cawley 2010, Henderson 2018) |
| §5 Positive finding | "Variant 비교에서 각각 hyperparameter 튜닝 후 비교" 명시 |
| §7.2 Robustness | Hyperparameter sensitivity analysis (variant별로 hyperparameter ± 30% 변동 시 결과 차이) |
| §8 Discussion | 한계: hyperparameter 공간 일부만 탐색, more thorough sweep은 future work |

## 백링크

- [[bayesian_optimization_tpe]] — Variant별 hyperparameter 탐색 (Optuna)
- [[walk_forward_cv]] — Nested CV 결합 가능성
- [[effect_size_rliable]] — 공정 비교 후 effect size 보고
- [[policy_gradient_stabilization]] — PPO hyperparameter는 공통, reward만 variant별

## 출처

- [Cawley & Talbot (2010) — On Over-fitting in Model Selection (JMLR)](https://www.jmlr.org/papers/v11/cawley10a.html)
- [Henderson et al. (2018) — Deep RL that Matters (AAAI)](https://arxiv.org/abs/1709.06560)
- [Andrychowicz et al. (2020) — What Matters in On-Policy RL?](https://arxiv.org/abs/2006.05990)
- [Nested CV (scikit-learn docs)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Nested CV introduction (Ploomber)](https://ploomber.io/blog/nested-cv/)
