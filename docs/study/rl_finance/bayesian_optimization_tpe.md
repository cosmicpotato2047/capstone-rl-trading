# Bayesian Optimization — TPE & Hyperband

> Bergstra, J., Bardenet, R., Bengio, Y., Kégl, B. (2011). _Algorithms for Hyper-Parameter Optimization._ NeurIPS.
> Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., Talwalkar, A. (2017). _Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization._ JMLR.

## 요지

1. **TPE (Tree-structured Parzen Estimator)**: hyperparameter 탐색을 두 분포의 ratio로 모델링하여 효율적으로 탐색.
2. **Hyperband**: 학습 자원(epoch/step)을 successive halving으로 동적 배분. 좋아 보이는 trial만 끝까지 학습.
3. **Optuna**는 두 알고리즘을 결합 (Sampler=TPE, Pruner=Hyperband or MedianPruner). 우리 프로젝트가 이미 사용 중.

## TPE 알고리즘 핵심

```
관찰값을 quantile γ 기준으로 분리:
  L = {θ : f(θ) < y_γ}    (좋은 trial들)
  H = {θ : f(θ) ≥ y_γ}    (나쁜 trial들)

각 그룹에서 kernel density estimation:
  l(θ) = p(θ | θ ∈ L)
  g(θ) = p(θ | θ ∈ H)

다음 후보 θ*:
  θ* = argmax  l(θ) / g(θ)
  (좋은 그룹에서는 자주, 나쁜 그룹에서는 드물게 나타나는 영역)
```

→ Gaussian Process Bayesian Optimization과 달리 **O(n³) 비용 없음** (GP의 본질적 한계).
→ Categorical, conditional, hierarchical search space 자연스럽게 지원.

## Hyperband 알고리즘 핵심

```
Successive Halving:
1. n개 trial 시작
2. 모두 r_0 만큼 학습 → 성능 측정
3. 상위 n/η 만 살아남고 r_0 × η 만큼 추가 학습
4. 반복

Hyperband: 여러 (n, r_0) 조합을 동시 실행해서 best trade-off 자동 발견
```

→ "더 많은 trial을 짧게" vs "적은 trial을 길게" 의 균형을 자동으로.
→ Optuna의 `MedianPruner`는 hyperband 변형 (간소 버전).

## 통계적 정당성

| 알고리즘 | 수렴 보장 | 노이즈 robust | High-dim 효율 |
|---|---|---|---|
| Grid Search | 보장됨 (전수조사) | 약함 | 매우 나쁨 |
| Random Search | 점근적 | 보통 | GP보다 나음 (Bergstra 2012) |
| GP-BO | 점근적, smoothness 가정 | 좋음 | O(n³) 한계 |
| **TPE** | 점근적 | 좋음 | GP보다 우수 |
| **TPE + Hyperband** | TPE 동일 | 좋음 + early stop으로 효율 | 매우 우수 |

## 한계 및 주의

1. **Multiple testing problem**: trial이 많을수록 우연히 좋은 결과가 나옴.
   → Bonferroni correction 또는 Deflated Sharpe Ratio (López de Prado) 필요.

2. **Non-stationary objective**: RL에서 같은 trial을 다시 돌려도 다른 결과 (seed 변동).
   → 각 trial을 multiple seed 평균으로 평가하거나, seed 자체를 trial axis로 포함.

3. **Local optima**: TPE는 mode-seeking 성향 — 한 영역에 빠르게 수렴하면 다른 영역 탐색 부족.
   → n_startup_trials를 충분히 크게 (전체의 20~30%).

## 우리 프로젝트와의 연결점

1. **현재 사용 중인 Optuna 설정 점검 (exp022, exp026, exp027, exp029)**:
   - Sampler: TPE (good)
   - Pruner: MedianPruner (good)
   - n_startup_trials: 10 — **너무 적을 수 있음** (50 trials 중 20%면 적정)
   - 단일 seed 평가 — **multi-seed 권장**

2. **exp029 3단계 Optuna 정당화**:
   - 1단계: PPO 하이퍼파라미터 (30 trials × 200k steps)
   - 2단계: 환경 파라미터 (50 trials × 200k steps)
   - 3단계: full 학습 (1M steps)
   → 학술적으로는 **factored optimization** (조건부 독립 가정).
   - 가정: PPO 파라미터와 환경 파라미터가 약 독립 → 별도 최적화 가능
   - 위반 가능성: 환경 파라미터를 바꾸면 PPO 최적값도 바뀔 수 있음
   - **검증 필요**: 2단계 결과로 1단계 재실행 했을 때 차이 ≤ noise floor인지

3. **Multiple testing 보정 (지금까지 안 한 부분)**:
   - exp022: 50 trials → Sharpe 56.4 best
   - exp026: 150 trials → Sharpe 1.978 best
   - exp029: 30 + 50 trials → Sharpe 2.019 best
   - **Deflated Sharpe Ratio** 계산해야 진짜 알파 vs 우연 구분 가능.
     ```
     DSR = (Sharpe - μ_uniform) / σ_uniform
       μ, σ는 동일 trial 수에서 임의로 얻을 수 있는 분포 추정
     ```

4. **즉시 적용 가능한 개선**:
   - Optuna 결과를 PBO (Probability of Backtest Overfitting) 계산기에 통과
   - n_startup_trials를 30~50으로 증가
   - 각 trial을 3 seeds 평균으로 평가 (compute 3배, 노이즈 1/√3)

## 백링크

- [[gort_2022_crypto_overfitting]] — PBO 적용 frame
- [[walk_forward_cv]] — 동일 Optuna 결과를 fold별로 재현성 검증
- [[Backtest Statistics]] (López de Prado Ch.14) — DSR 정식

## 출처

- [Bergstra et al. (2011) NeurIPS](https://www.researchgate.net/publication/304781977_Algorithms_for_hyper-parameter_optimization)
- [TPE: Understanding Its Components (Watanabe 2023)](https://arxiv.org/abs/2304.11127)
- [Building a TPE from Scratch (Towards Data Science)](https://towardsdatascience.com/building-a-tree-structured-parzen-estimator-from-scratch-kind-of-20ed31770478/)
