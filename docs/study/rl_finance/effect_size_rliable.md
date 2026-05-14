# Effect Size & rliable — RL Variant 비교의 통계적 정직성

> Cohen (1988) — Statistical Power Analysis for the Behavioral Sciences. (Cohen's d 정의)
> Agarwal, Schwarzer, Castro, Courville, Bellemare (2021). _Deep Reinforcement Learning at the Edge of the Statistical Precipice._ NeurIPS Outstanding Paper.
> Kruschke (2013). _Bayesian Estimation Supersedes the t Test._ J. Experimental Psychology: General.

## 요지

1. **p-value만으로는 부족**. 충분히 많은 sample은 어떤 작은 차이도 "통계적으로 유의" 만든다. Effect size가 진짜 차이의 크기를 측정.
2. **DRL 평가의 위기**: 대부분 논문이 mean/median만 보고하고 stratified bootstrap CI 없음 → 재현성 위기. Agarwal 2021 (NeurIPS Outstanding) 가 정식 해법 제시 + `rliable` 라이브러리.
3. **Bayesian estimation (BEST)**: t-test의 Bayesian 대안. Credible interval로 effect의 분포 전체를 표현.

## Cohen's d (가장 단순한 effect size)

```
d = (mean_A - mean_B) / pooled_std

해석 (Cohen 1988):
  d ≈ 0.2 : small effect
  d ≈ 0.5 : medium effect
  d ≈ 0.8 : large effect

주의: 위 기준은 "심리학" 도메인용. 금융/RL에서 그대로 적용은 무리.
      도메인 specific benchmark가 필요.
```

→ p < 0.05 이지만 d = 0.05 라면 "통계적으론 유의하나 실질적으론 무의미".

## rliable (Agarwal 2021) — DRL 평가 표준 도구

### 핵심 문제
> Deep RL 논문 대부분이 mean/median을 단일 숫자로 보고. 3~5개 seed로 끝.
> 분산이 큰 결과를 단일 숫자로 압축 → 재현성 위기.

### 제안된 방법

| 도구 | 목적 |
|---|---|
| **Stratified Bootstrap CI** | seed × task 별 재샘플링으로 통계적 불확실성 정량화 |
| **Performance Profile** | 임계값 τ 위에서 성능 분포 (분포 전체 비교) |
| **IQM (Interquartile Mean)** | 상하위 25% 제외한 평균 → 이상치에 robust |
| **Probability of Improvement** | "A가 B보다 나을 확률" |
| **Optimality Gap** | 이상적 성능 대비 격차 |

### Probability of Improvement
```
P(X > Y) = Σ_{x,y} I(x > y) / (N × M)

여기서 X, Y는 각각 알고리즘 A, B의 score 분포 (seed × task)
```

→ p-value 대신 "A가 B보다 나을 확률"을 직접 보고. 직관적이고 robust.

### IQM (Interquartile Mean)
```
1. seed × task 모든 점수를 모음
2. 상위 25%, 하위 25% 제거
3. 중간 50%의 평균
```

→ Mean보다 이상치에 robust, Median보다 정보 보존.
→ 우리 RL trading에서 한두 seed가 우연히 강세장 episode를 잡아 부풀린 결과를 자동 보정.

## Bayesian Estimation (BEST, Kruschke 2013)

t-test의 Bayesian 대안:

```
표준 t-test:
  H_0: μ_A = μ_B
  결과: p-value (검정 통계량의 극단성)
  → "유의하다 / 안 하다" 이분법

BEST:
  Prior + Likelihood → Posterior of (μ_A - μ_B)
  결과: 95% HDI (Highest Density Interval) of difference
  → "차이가 [a, b] 범위에 95% 확률로 있다"
  → 분포 전체 → effect size + uncertainty 동시 표현
```

장점:
- t-분포 사용으로 outlier에 robust
- Sample이 작아도 적절한 결과
- "차이가 정확히 0일 확률" 같은 직접 질문 가능

## 우리 프로젝트 직결 — exp032 평가 설계 강화

### 현재 plan의 약점
exp032 변경 사항 (project_continuation_plan.md):
- 각 variant 5 seeds × Val 5 episode = **25 점수/variant**
- 비교 방법: Mann-Whitney U test (pairwise)
- **Effect size 측정 없음**
- **CI 추정 없음**
- **IQM 같은 robust aggregate 없음**

→ "asymmetric이 Sharpe 0.1 높다, p < 0.05" 결과만으로는 디펜스 약함.

### 보강 제안 (exp032 평가에 추가)

```python
# 1. Cohen's d 매 pair에 대해
from numpy import std
d = (mean_A - mean_B) / pooled_std

# 2. rliable의 IQM + stratified bootstrap CI
import rliable.metrics as rly
iqm = rly.aggregate_iqm(scores)  # scores: (n_seed, n_episode) shape
ci_lower, ci_upper = rly.aggregate_iqm_ci(scores, num_bootstrap=2000)

# 3. Probability of Improvement
prob_improvement = (scores_A[:, None] > scores_B[None, :]).mean()
# scores_A가 scores_B보다 클 확률

# 4. Performance Profile
import rliable.plot_utils as rlp
rlp.plot_performance_profiles(scores, taus=np.linspace(0, max_score, 100))

# 5. (선택) Bayesian BEST
import pymc as pm
with pm.Model() as best:
    # BEST 모델 정의 ...
    trace = pm.sample(2000)
# HDI of (μ_A - μ_B) 보고
```

### exp032 평가 표 (강화 후)

| Variant pair | Cohen's d | IQM diff | 95% CI | P(A > B) | BEST HDI (95%) |
|---|---|---|---|---|---|
| asym vs sym | 0.85 | +0.42 | [0.31, 0.53] | 0.92 | [0.28, 0.55] |
| dsr vs sym | 0.41 | +0.18 | [0.05, 0.31] | 0.74 | [0.06, 0.30] |
| pt vs sym | 0.90 | +0.45 | [0.34, 0.56] | 0.94 | [0.31, 0.58] |
| asym vs pt | 0.05 | -0.02 | [-0.12, 0.08] | 0.48 | [-0.13, 0.09] |

→ 이렇게 보고하면:
- 통계적 유의 ✓
- 실질적 효과 크기 ✓ (Cohen's d)
- 불확실성 정량화 ✓ (CI, HDI)
- 직관적 확률 ✓ (P(A > B))
- 디펜스에서 "통계 robust한가" 질문 즉답 가능

## 본 논문 활용

| 챕터 | 활용 |
|---|---|
| §3.5 Evaluation Methodology | rliable 도구 + Cohen's d 인용으로 평가 방법론 정당화 |
| §5 Positive finding | 위 강화 표를 메인 결과로 사용 |
| §7.2 Robustness | Performance profile로 분포 전체 비교 |

## 백링크

- [[walk_forward_cv]] — CPCV의 fold 분포와 결합 (각 path마다 effect size 계산)
- [[bayesian_optimization_tpe]] — DSR과 함께 다중검정 보정의 일부
- [[gort_2022_crypto_overfitting]] — PBO + 본 노트의 effect size = 완전한 통계 검증

## 출처

- [Agarwal et al. (2021) — Deep RL at the Edge of the Statistical Precipice (NeurIPS)](https://arxiv.org/abs/2108.13264)
- [rliable GitHub](https://github.com/google-research/rliable)
- [rliable 시각 설명 (Antonin Raffin)](https://araffin.github.io/post/rliable/)
- [Kruschke (2013) — Bayesian Estimation Supersedes the t Test](https://pubmed.ncbi.nlm.nih.gov/22774788/)
- [Effect Size — Wikipedia](https://en.wikipedia.org/wiki/Effect_size)
- [Effect Size: Beyond Statistical Significance (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4975211/)
