# Causal & Counterfactual Analysis in RL — 메커니즘 답변자

> COUNTERPOL: Bhattacharyya et al. (2023). _Counterfactual Explanation Policies in RL._ arXiv: 2307.13192.
> Madumal et al. (2020). _Explainable RL through a Causal Lens._ AAAI.
> Survey: _Redefining Counterfactual Explanations for RL_ (ACM CSUR 2024).

## 요지

1. **RQ-3 답변에 필수**: "어떤 reward가 우위를 만든다면 정책 행동이 어떻게 다른가?" 에 답하려면 단순 통계 비교 너머 **인과적/반사실적 분석**이 필요.
2. **세 가지 도구**: (a) Counterfactual policy comparison, (b) SHAP for RL (feature attribution), (c) Causal influence diagram.
3. 우리 exp032의 메커니즘 분석 (현재 "regime별 행동 분포" 한 줄) 을 이 세 도구로 강화.

## 1. Counterfactual Policy Comparison (COUNTERPOL)

### 질문 형태
> "Reward를 A→B로 바꿨을 때, 정책이 같은 state에서 어떻게 다르게 행동했을까?"

### 방법
```
1. 환경 + 시작 state 고정
2. 정책 π_A (asym reward로 학습) 와 π_B (sym reward로 학습) 비교
3. 동일 trajectory τ에서 각 state s_t에 대해:
   action_A = π_A(s_t)
   action_B = π_B(s_t)
   diff_t = action_A - action_B
4. diff_t가 어떤 state feature에 의해 결정되는지 회귀 분석
```

→ "asym 정책은 bear regime + 큰 손실 위협 시 sym 대비 더 보수적 action 선택"
   같은 인과적 진술 가능.

### 우리 exp032 적용
- exp032 4 variant 모두 같은 random_start seeds로 평가
- 각 step의 action 차이를 state feature (trend, volatility, holdings) 와 회귀
- → "Variant 차이가 어떤 state에서 가장 크게 발현되는가" 정량화

## 2. SHAP for RL — Feature Attribution

### 질문 형태
> "정책 π의 action 결정에 어떤 state 변수가 얼마나 기여하는가?"

### 방법
```python
import shap

# 정책을 함수로 wrapping
def policy_fn(states):  # shape (N, state_dim)
    return model.predict(states, deterministic=True)[0]

# SHAP explainer
explainer = shap.Explainer(policy_fn, background_states)
shap_values = explainer(test_states)

# action[0] (aggressiveness) 에 각 state feature 기여도
shap.summary_plot(shap_values[:, :, 0], test_states,
                  feature_names=['log_price', 'divergence', 'hvr',
                                'cash_ratio', 'volatility',
                                'trend_short', 'trend_long'])
```

### 우리 exp032 적용
- 각 variant의 best 정책에 대해 SHAP 분석
- variant 간 SHAP 분포 비교
- → "asym 정책은 trend_long에 더 민감, sym 정책은 volatility에만 반응"
   같은 메커니즘 진술

### 한계
- PPO의 stochastic policy → SHAP은 기댓값 기반
- High-dim state에서 SHAP 계산 비용 큼 (KernelExplainer O(2^d))
- 우리는 7~9D state라 부담 작음

## 3. Causal Influence Diagram

### 질문 형태
> "Reward의 형태가 state-action 매핑의 어떤 채널을 통해 알파에 영향을 미치는가?"

### 다이어그램 구조
```
Reward shape (sym/asym/dsr/pt)
        ↓
Policy 학습 (loss surface 변화)
        ↓ ↘
Policy entropy ↓     ↘
Action 분포 ↓         ↘
   ↙↓↘                 ↘
거래 빈도   사이클 승률   포지션 보유 기간
   ↘↓↙
   Sharpe
```

→ Reward → Sharpe 의 직접 효과 vs 간접 효과 (mediator 통한)를 정량화.

### 통계 도구: Mediation Analysis
```
직접 효과: Reward → Sharpe (controlling for mediators)
간접 효과: Reward → Mediator → Sharpe
총 효과: 직접 + 간접
```

→ "asym reward의 Sharpe 향상 중 70%는 '거래 빈도 감소' 를 통해 매개됨" 같은 답.

## 4. Behavioral Cloning Distance — variant 간 정책 거리

### 질문
> "두 variant 정책이 얼마나 다른가?"

### 방법
```
# 동일 state에서 두 정책의 action KL divergence
KL(π_A || π_B) = E_{s~D}[ KL(π_A(·|s) || π_B(·|s)) ]

또는 deterministic policy의 경우:
Distance = E_{s~D}[ ||π_A(s) - π_B(s)||² ]
```

### 우리 exp032 적용
- 4 variant 정책 간 pairwise distance matrix
- t-SNE / UMAP으로 정책 임베딩 시각화
- → variant들이 정책 공간에서 어디에 위치하는지 직관적 표현

## 우리 프로젝트 직결 — exp032 메커니즘 분석 강화

### 현재 plan (project_continuation_plan.md)
```
exp032의 행동 차원 분석:
- 거래 빈도
- 사이클 승률
- regime별 action 분포
```

→ 표면적 통계만. RQ-3 ("그 reward가 정책 행동을 어떻게 변화시키는가?") 에 답하기 약함.

### 보강 (exp032에 추가)

```
exp032 메커니즘 분석 (강화 버전):

1. Counterfactual policy comparison:
   - 동일 trajectory에서 4 variant의 action 차이
   - state feature와의 회귀 → 차이의 driver 식별

2. SHAP feature attribution:
   - 각 variant의 SHAP value 매트릭스
   - variant 간 attribution 패턴 비교

3. Mediation analysis:
   - Reward → {거래 빈도, 사이클 승률, 보유 기간} → Sharpe
   - 직접/간접 효과 분해

4. Policy distance:
   - variant pairwise KL divergence
   - t-SNE 시각화

5. 기본 통계 (기존 plan 유지):
   - regime별 action 분포 + 사이클 통계
```

→ 위 5개를 종합하면 RQ-3에 "정량적으로 답할 수 있는 메커니즘 진술" 가능.

## 본 논문 활용

| 챕터 | 활용 |
|---|---|
| §6 Mechanism (RQ-3 메인 챕터) | 위 5개 분석을 사용해 reward → policy → alpha 인과 채널 정량화 |
| §3.5 Evaluation | Counterfactual + SHAP 방법론 인용 |
| §8 Discussion | 한계: 인과 추정의 confounding 가능성 (시뮬레이터 한정) |

## 한계 / 주의

1. **시뮬레이터 환경의 인과 추정은 협소**: 실제 시장은 더 복잡한 confounder 존재
2. **SHAP의 계산 비용**: PPO action distribution의 expectation 추정에 sample 많이 필요
3. **Mediation 가정**: 매개변수 선택 임의성 — 다른 매개 변수 선택 시 결과 다를 수 있음
4. **Counterfactual의 OOD 문제**: π_A로 학습된 정책이 한 번도 보지 못한 state에서의 counterfactual은 신뢰성 낮음

## 백링크

- [[reward_shaping_ng1999]] — Reward 변화가 정책에 미치는 영향의 이론적 출발
- [[ppo_schulman_2017]] — 우리 PPO 정책의 stochastic 성질이 분석 방법에 영향
- [[effect_size_rliable]] — Variant 간 차이의 통계적 검증
- [[distributional_rl]] — Risk-sensitive 정책 인식과 결합 가능

## 출처

- [COUNTERPOL: Counterfactual Explanation Policies in RL](https://arxiv.org/pdf/2307.13192)
- [Redefining Counterfactual Explanations for RL (ACM CSUR 2024)](https://dl.acm.org/doi/10.1145/3648472)
- [Survey of Explainable RL (2025)](https://arxiv.org/html/2507.12599)
- [Behaviour Discovery and Attribution for Explainable RL](https://arxiv.org/html/2503.14973)
- [Counterfactual Explanations for Continuous Action RL](https://arxiv.org/html/2505.12701v1)
