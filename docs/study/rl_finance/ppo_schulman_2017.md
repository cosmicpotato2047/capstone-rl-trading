# PPO — Proximal Policy Optimization (Schulman et al., 2017)

> Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). _Proximal Policy Optimization Algorithms._ arXiv: 1707.06347.

## 요지

1. **TRPO의 단순화 버전.** TRPO의 trust region 제약을 clipped objective로 대체 → 1차 최적화로 가능.
2. 핵심 아이디어: policy update의 "거리"를 제한해서 한 번에 너무 멀리 가지 않게 → 안정성.
3. 우리 프로젝트가 채택한 알고리즘. SB3가 표준 구현 제공.

## TRPO의 핵심 아이디어 (배경)

```
표준 Policy Gradient (VPG):
  ∇_θ J(θ) = E[∇_θ log π_θ(a|s) · A^π(s,a)]

문제: gradient step이 너무 크면 정책이 망가짐 (catastrophic update)

TRPO 해법:
  maximize  L(θ) = E[π_θ(a|s) / π_old(a|s) · A^π(s,a)]
  subject to  KL(π_old || π_θ) ≤ δ

→ 2차 최적화 + Fisher Information matrix → 계산 비쌈
```

## PPO Clipped Objective (핵심 기여)

```
r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)    (probability ratio)

L^CLIP(θ) = E_t[ min(r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t) ]
```

ε ≈ 0.2 (표준값)

**직관**:
- A_t > 0 (좋은 action): r_t를 1+ε 이상 키우려고 하면 clip → 보상 제한
- A_t < 0 (나쁜 action): r_t를 1-ε 이하로 줄이려고 하면 clip → 보상 제한
- **너무 큰 update를 자동으로 페널티**

## 전체 PPO Loss

```
L^PPO(θ) = L^CLIP(θ)  - c_1 · L^VF(θ)  + c_2 · S[π_θ](s)

L^VF: value function loss (MSE)
S: entropy bonus (탐색 유도)
c_1 ≈ 0.5, c_2 ≈ 0.01
```

## PPO 학습 알고리즘 (의사코드)

```python
for iteration in range(N):
    # 1. Rollout 수집 (n_steps × n_envs)
    trajectory = run_policy(π_θ, env, n_steps)

    # 2. Advantage 계산 (GAE-λ)
    advantages = compute_gae(trajectory, V_θ, γ, λ)

    # 3. Mini-batch SGD (K epochs)
    for epoch in range(K):
        for batch in mini_batches:
            loss = -L^CLIP(θ) + c_1 * L^VF(θ) - c_2 * S[π_θ]
            θ ← Adam(θ - lr * ∇θ loss)
```

## GAE (Generalized Advantage Estimation) — 짝궁

```
A_t^GAE(γ, λ) = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}
where δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

- λ=0: TD(0), 낮은 분산, 큰 bias
- λ=1: Monte Carlo, 높은 분산, 0 bias
- λ=0.95 (표준): 적당한 trade-off

## 왜 PPO가 사실상 표준이 됐는가

| 기준 | DQN | TRPO | PPO | SAC |
|---|---|---|---|---|
| Action 공간 | Discrete | Both | Both | Continuous |
| Sample efficiency | High (off-policy) | Low (on-policy) | Low (on-policy) | High (off-policy) |
| 구현 복잡도 | 보통 | 매우 높음 | 낮음 | 보통 |
| Stability | 보통 | 매우 좋음 | 좋음 | 좋음 |
| 병렬 학습 | 어려움 | 가능 | **매우 적합** | 어려움 |

→ PPO의 위치: "TRPO의 안정성, 더 단순한 구현, 병렬 학습 적합" 셋의 균형.
→ 우리 프로젝트가 PPO 선택한 정당화.

## PPO의 약점

1. **On-policy** → sample efficiency 낮음 (off-policy SAC, TD3 대비)
2. **Catastrophic forgetting** 가능성 — 학습이 진행되며 이전에 잘 하던 영역 까먹기
3. **Hyperparameter sensitivity** — n_steps, batch_size, n_epochs 조합에 민감

## 우리 프로젝트 직결 — PPO 디테일 점검

### 현재 사용 중인 SB3 PPO 디폴트와 우리 설정 비교

| 파라미터 | SB3 default | 우리 (exp029) | 권장 |
|---|---|---|---|
| learning_rate | 3e-4 | 1.06e-4 (Optuna) | OK |
| n_steps | 2048 | 1024 (Optuna) | 너무 작을 수 있음 |
| batch_size | 64 | 64 추정 | n_steps/2~4 권장 |
| n_epochs | 10 | 10 | OK |
| gamma | 0.99 | 0.988 | OK |
| gae_lambda | 0.95 | 0.902 | OK |
| clip_range | 0.2 | 0.367 (Optuna) | 너무 크다 — 정책 너무 빠르게 변할 위험 |
| ent_coef | 0.0 | 0.008 | OK |
| vf_coef | 0.5 | 0.5 | OK |
| target_kl | None | None | **추가 권장** |

→ exp029 학습 불안정의 원인 후보: **clip_range=0.367 너무 큼** (표준 0.2)
→ Optuna가 short-term reward 최대화를 위해 큰 clip을 선호했을 가능성
→ target_kl 추가하면 clip이 커도 KL early stop으로 안전 보장

### "37 Implementation Details of PPO" (ICLR 2022 blog) 핵심
- Advantage normalization (per-batch)
- Value function clipping (PPO2 변형)
- Reward scaling (학습 안정성)
- Orthogonal weight init
- Adam epsilon=1e-5 (default 1e-8 아님)
- → 이 디테일들이 PPO 성능의 50%를 결정

## 백링크

- [[mdp_bellman_pomdp]] — PPO가 푸는 문제 정의
- [[ddpg_continuous_control]] — 연속 action 대안
- [[policy_gradient_stabilization]] — 실용 안정화 기법
- [[Spinning Up — PPO]] (외부)

## 출처

- [Schulman et al. (2017) PPO arXiv 1707.06347](https://arxiv.org/abs/1707.06347)
- [PPO Explained (Towards Data Science)](https://towardsdatascience.com/proximal-policy-optimization-ppo-explained-abed1952457b/)
- [37 Implementation Details of PPO (ICLR 2022 blog)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Wikipedia — PPO](https://en.wikipedia.org/wiki/Proximal_policy_optimization)
