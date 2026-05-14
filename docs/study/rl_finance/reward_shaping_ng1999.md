# Reward Shaping — Ng, Harada, Russell (1999)

> Ng, A. Y., Harada, D., Russell, S. (1999). _Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping._ ICML.

## 요지

1. **Reward shaping** = 학습을 빠르게 하기 위해 원래 reward에 추가 신호를 더하는 기법.
2. **위험**: 잘못 더하면 **최적 정책이 바뀜** → 에이전트가 "shaped reward는 최대화하지만 진짜 목표는 안 함" (= reward hacking 한 종류).
3. **Ng 정리**: shaping이 정책 불변(policy invariance)을 보장하는 **필요충분조건**은 **potential-based** 형태:
   ```
   F(s, a, s') = γ Φ(s') - Φ(s)
   ```

## 정리 (Theorem 1, Ng 1999)

```
원래 MDP: M = ⟨S, A, P, R, γ⟩
변형:     M' = ⟨S, A, P, R + F, γ⟩

F가 임의의 잠재함수 Φ: S → ℝ 에 대해
  F(s, a, s') = γ Φ(s') - Φ(s)
형태이면:
  π는 M에서 최적 ⟺ π는 M'에서 최적

역도 성립:
  F가 위 형태가 아니면, 어떤 보조 MDP에서 최적 정책이 바뀜
```

→ "shaping해도 안전한 유일한 형태는 potential difference."
→ Q-value 초기화로도 해석 가능: $Q'(s,a) = Q(s,a) + \Phi(s)$.

## 왜 단순 shaping이 위험한가

```
나쁜 shaping 예:
  원래 reward: 미로 끝에서 +1
  추가 reward: 한 칸 이동마다 +0.01 ("탐색 격려")

문제: 에이전트가 끝까지 안 가고 영원히 미로를 돌면 reward 누적 → 무한 보상
     실제 목표(끝 도달)와 무관하게 학습 망가짐
```

→ Goodhart's law의 RL 버전.

## Potential function 설계

```
Φ(s) = 어떤 휴리스틱 "이 state가 얼마나 좋은가" 추정

예시:
- 미로: Φ(s) = -manhattan_distance(s, goal)
- 체스: Φ(s) = material_balance(s)
- 트레이딩: Φ(s) = log(equity_t / start_capital)  ???
```

**중요**: Φ 자체는 학습 신호 ≠ Q-value 추정. 그냥 "이 state가 얼마나 좋아 보이는가"의 직관.

## 우리 프로젝트와의 연결점

1. **현재 reward는 sparse하지 않음** (매 step equity 변화) → shaping 직접 필요성은 낮음.

2. **exp029의 r_idle = -idle_rate × cash_ratio는 reward shaping의 한 형태**:
   ```
   r_t = r_step + r_cycle + r_idle
   ```
   - 이게 Ng의 potential-based 인가?
   - r_idle = -k × cash_ratio = -k × (cash / start_capital)
   - 만약 Φ(s) = -k × cash_ratio라면? cash_ratio가 1에 가까우면 페널티
   - 사이클 끝나면 cash_ratio → 1 (전량 청산) → Φ 큰 음수 → 다음 사이클 빨리 시작 유도
   - → **부분적으로 potential-based 해석 가능하지만 엄밀하지 않음**

3. **잠재적 reward hacking 위험**:
   - r_idle이 cash_ratio 기반 → 에이전트가 "cash를 줄이는" 방향으로 학습 가능
   - 비합리적 매수(가격 무관)로 cash 소진 → r_idle 회피 → 손실 사이클
   - exp028/exp029에서 학습 oscillation의 한 원인일 수도

4. **Potential-based 재설계 제안**:
   ```python
   # 위험: 단순 cash penalty
   r_idle = -idle_rate × cash_ratio

   # 안전: potential-based
   def potential(state):
       return -idle_rate × cash_ratio × (idle_steps > grace_period)
   r_idle = γ × potential(next_state) - potential(state)
   ```
   - 이렇게 하면 episode 끝에서 potential이 0으로 수렴 → 누적 효과 0
   - 학습 속도만 가속, 최적 정책 불변 보장

5. **DSR을 potential-based로 해석**:
   - DSR_t = ∂Sharpe/∂R_t (테일러 1차)
   - 누적 시: ΔSharpe = Σ DSR_t
   - 에피소드 끝의 Sharpe와 step DSR 합은 거의 같음 → 자연스럽게 potential-like

## Reward shaping 외 안전한 reward 보강 방법

| 방법 | 설명 | 정책 불변 |
|---|---|---|
| Potential-based shaping | F = γΦ' - Φ | ✓ (Ng 정리) |
| 절대 가산 reward | F = const | ✗ (보장 안 됨) |
| 시간 페널티 | F = -k × step | ✗ |
| Curiosity bonus (RND, ICM) | F = state novelty | △ (탐색 강화는 OK, 최종 정책은 다를 수 있음) |
| Distillation reward | F = -KL(π || π_expert) | ✗ (다른 정책 유도) |

## 우선순위 액션 (다음 실험)

1. **exp029 reward 점검** (D-LV1):
   - r_idle을 potential-based 형식으로 재작성
   - 이론적으로 r_step 최적화 정책과 동일해야 함
   - 학습 안정성 비교

2. **DSR로 전체 reward 교체** (D-LV2):
   - r_step → DSR
   - r_idle은 유지 (idle 정의가 다른 의미)
   - 학습 속도와 안정성 동시 측정

3. **r_cycle 페널티 위험 점검**:
   - exp029 r_cycle = w_cycle × (cycle_return - 1)
   - 큰 loss cycle에서 -10~-50% 가능 → outlier
   - clipping 또는 winsorization 고려

## 백링크

- [[mdp_bellman_pomdp]] — Ng 정리의 수학적 토대
- [[reward_hacking]] — Shaping 잘못하면 hacking
- [[differential_sharpe_moody2001]] — DSR이 자연스러운 dense reward
- [[ppo_schulman_2017]] — On-policy에서 reward 시그널 영향 큼

## 출처

- [Ng, Harada, Russell (1999) — Policy Invariance](https://www.andrewng.org/publications/policy-invariance-under-reward-transformations-theory-and-application-to-reward-shaping/)
- [Reward Shaping — Springer](https://link.springer.com/rwe/10.1007/978-0-387-30164-8_731)
- [Policy Invariance lecture notes (UToronto CSC2542)](https://www.teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_reward_shaping.pdf)
- [Improving the Effectiveness of Potential-Based Shaping (2025)](https://arxiv.org/html/2502.01307v1)
