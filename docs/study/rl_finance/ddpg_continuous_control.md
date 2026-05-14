# DDPG — Continuous Control with Deep RL (Lillicrap et al., 2015)

> Lillicrap, T., Hunt, J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., Wierstra, D. (2015). _Continuous Control with Deep Reinforcement Learning._ arXiv: 1509.02971.

## 요지

1. **DQN(discrete action)을 연속 action으로 확장.** Deterministic policy gradient + neural function approximator.
2. Actor-critic 구조: **Actor** π_θ(s) → action (deterministic), **Critic** Q_φ(s, a) → value.
3. **Off-policy** — replay buffer 사용으로 sample efficient.
4. 우리가 PPO 선택한 이유와의 비교 — 두 알고리즘의 trade-off 이해.

## 핵심 알고리즘

```python
# Actor: deterministic policy μ_θ(s) → a
# Critic: action-value Q_φ(s, a)

# Target networks (slow-moving copies)
μ_target, Q_target = clone(μ, Q)

for step in range(N):
    a = μ_θ(s) + ε   # exploration noise (Ornstein-Uhlenbeck)
    s', r, done = env.step(a)
    replay.add(s, a, r, s', done)

    # Sample batch
    batch = replay.sample()

    # Critic update (TD)
    y = r + γ Q_target(s', μ_target(s'))
    L_critic = MSE(Q_φ(s, a), y)
    φ ← φ - α ∇φ L_critic

    # Actor update (deterministic policy gradient)
    L_actor = -mean(Q_φ(s, μ_θ(s)))
    θ ← θ - α ∇θ L_actor

    # Soft update target networks
    θ_target ← τ θ + (1-τ) θ_target  (τ ≈ 0.005)
    φ_target ← τ φ + (1-τ) φ_target
```

## DDPG의 주요 트릭

1. **Replay buffer** — 과거 transition 재사용으로 sample efficient
2. **Target networks** — 학습 안정성 (moving target 문제 해결)
3. **Soft update** (Polyak averaging) — hard update보다 안정
4. **Batch normalization** — feature scale 정규화
5. **Action noise** — deterministic policy의 탐색 부족 보완 (보통 Ornstein-Uhlenbeck process)

## DDPG의 문제점 (TD3가 해결)

- **Q-value 과대추정** — actor가 critic의 과대평가된 영역을 활용
- **Instability** — target Q 변동 큼
- **Hyperparameter sensitivity** — DDPG는 운에 좌우되는 정도가 큼

→ Fujimoto et al. (2018) **TD3** (Twin Delayed DDPG): twin critic + delayed policy update + target smoothing → 훨씬 안정적
→ Haarnoja et al. (2018) **SAC** (Soft Actor-Critic): entropy bonus 내장 → 가장 안정적 연속 control 알고리즘

## PPO vs DDPG/SAC 비교 (우리 선택 이유)

| 측면 | PPO | DDPG/TD3 | SAC |
|---|---|---|---|
| Sample efficiency | 낮음 (on-policy) | 높음 (off-policy) | 높음 (off-policy) |
| 안정성 | 좋음 | 불안정 | 매우 좋음 |
| Hyperparameter robust | 좋음 | 까다로움 | 좋음 |
| 병렬 학습 | 매우 적합 | 어려움 | 보통 |
| 구현 단순 | 단순 | 보통 | 보통 |
| Discrete + Continuous | 둘 다 | Continuous only | Continuous only |
| 학계 표준 | FinRL, Spinning Up | 일부 | 일부 |

→ **우리가 PPO 선택한 정당화**:
- 안정성 + 병렬 학습 적합성 (n_envs=4 사용)
- 구현 단순 → 디버깅 용이
- DRL trading 학계 표준 (Zhang 2020, FinRL, Gort 2022 모두 PPO/A2C 위주)

→ **잠재적 SAC 시도 가치**:
- 우리 PPO가 sample efficiency 낮음 (1M+ 스텝 필요)
- SAC는 같은 데이터로 더 빠르게 학습
- 단점: 우리 환경에서 SAC가 PPO보다 잘 한다는 보장 없음

## 우리 프로젝트와의 연결점

1. **현재 action 공간 [0,1]² 또는 [0,1]^5는 연속**:
   - DDPG/TD3/SAC 모두 직접 적용 가능
   - 단, 우리는 PPO 선택 (위 trade-off 근거)

2. **PPO + Continuous Action의 SB3 구현**:
   - Gaussian policy: π(a|s) = N(μ_θ(s), σ_θ(s))
   - Squashing: tanh 또는 sigmoid로 [0,1] 또는 [-1,1] 클립
   - 우리: tanh(rescale) — exp016에서 raw mean [-9.19, -4.30] → 항상 [0, 0] 출력 → 정책 포화

3. **정책 포화의 DDPG 관점 해석**:
   - Deterministic policy μ가 corner solution에 도달하면 Q gradient도 corner 방향 → 더 corner로
   - Actor-critic 구조에서 흔한 실패 모드
   - 해결: noise scheduling, replay buffer 다양성

4. **만약 SAC로 옮긴다면**:
   ```python
   from stable_baselines3 import SAC

   model = SAC(
       'MlpPolicy', env,
       learning_rate=3e-4,
       buffer_size=1_000_000,
       ent_coef='auto',         # 자동 조정 ← entropy 보장
       target_entropy='auto',    # 자동 조정
   )
   ```
   - ent_coef='auto'가 진짜 강점: SAC가 자동으로 탐색 수준 유지
   - 우리 정책 포화 문제 자연 해결 가능성

## 우선순위 (수정 시 비용)

1. **PPO 유지 + 안정화** (가장 안전): exp028~030 이미 진행 중
2. **SAC 시도** (보통): SB3에서 한 줄 변경. compute 비용 비슷
3. **TD3 시도** (보통): SAC와 유사
4. **DDPG 시도** (낮음): 일반적으로 SAC/TD3가 dominant

## 백링크

- [[ppo_schulman_2017]] — 우리가 채택한 알고리즘
- [[mdp_bellman_pomdp]] — MDP framework
- [[distributional_rl]] — DDPG의 distributional 확장 (D4PG)

## 출처

- [Lillicrap et al. (2015) arXiv 1509.02971](https://arxiv.org/abs/1509.02971)
- [DDPG Systematic Review (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2405844024067288)
- [DDPG Explained (Medium)](https://medium.com/data-science/deep-deterministic-policy-gradients-explained-2d94655a9b7b)
