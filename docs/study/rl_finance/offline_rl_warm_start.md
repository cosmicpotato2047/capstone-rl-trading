# Offline RL + Behavioral Cloning — Warm Start for Trading

> Levine et al. (2020) — Offline RL: Tutorial, Review, Perspectives.
> Kumar & Levine (2022) — When Should We Prefer Offline RL Over Behavioral Cloning? arXiv: 2204.05618.
> WSRL (Warm Start RL): 2025 ICLR. Trading 응용: Offline RL for Stock Trading (2023).

## 요지

1. **Offline RL**: 환경과 상호작용 없이 사전 수집된 데이터로 정책 학습.
2. **Behavioral Cloning (BC)**: 전문가 시연을 지도학습으로 모방. 가장 단순한 offline 방법.
3. **Warm Start**: BC 또는 offline RL로 초기 정책 학습 → online RL로 fine-tune.
4. 우리 프로젝트 직결: ATR 고정 정책(Bayesian 최적화)으로 BC → PPO fine-tune → exp029의 "초반 -6 Sharpe 학습 낭비" 회피.

## Offline RL vs BC 결정 기준 (Kumar 2022)

| 데이터 품질 | 권장 방법 |
|---|---|
| 전문가 수준 (단일 sub-optimal 정책) | **BC 우세** |
| 혼합 (전문가 + 평균 + 나쁜 정책) | **Offline RL** (CQL, IQL 등) |
| 광범위 noise, 다양한 정책 | Offline RL |
| 단순 demonstration | BC |

→ 우리 프로젝트: ATR 고정 정책 = 단일 sub-optimal 전문가 → **BC가 적절**.

## 핵심 알고리즘

### Behavioral Cloning
```python
# 가장 단순. 지도학습.
for (state, action) in expert_dataset:
    loss = -log π_θ(action | state)
```

문제: 분포 shift — 학습한 state 외에서 행동 불안정 (compounding error).

### Conservative Q-Learning (CQL) — 상세는 [[cql_kumar_2020]]
```
표준 Q-learning loss + min/max Q(s, OOD action) 페널티
→ "training set에 없는 action에 대해서는 보수적 Q 추정"

수식:
L_CQL = α · [log Σ_a exp(Q(s,a)) − E_{a~π_β}[Q(s,a)]] + L_Bellman
```

→ Offline 데이터셋 밖에서의 환상 알파(hallucinated value)를 억제.
→ **Mixed dataset (여러 sub-optimal 정책 혼합)** 에서 BC보다 우수.
→ 단일 ATR 전문가 → BC 우세. 여러 베이스라인 + ATR 변형 mix → CQL 우세.

### Implicit Q-Learning (IQL)
```
quantile regression으로 V(s) 추정 (action 직접 안 봄)
정책: argmin KL(π || π_β) where π_β는 데이터의 분포
```

→ OOD action을 명시적으로 피하지 않고, 데이터 분포 안에서 최적화.

### WSRL (Warm Start RL, ICLR 2025)
```
Phase 1: Offline RL or BC로 초기 정책 π_init
Phase 2: π_init으로 짧은 online rollout → replay buffer warmup
Phase 3: 표준 online RL (PPO, SAC 등)으로 fine-tune
```

→ "초반 학습 낭비" 해결. 우리 exp029에 정확히 매칭.

## 우리 프로젝트와의 연결점

1. **현재 학습 곡선의 핵심 비효율**:
   ```
   exp029 초반:
     step=50k:  Sharpe= -6.222 (Return -34%, MDD 35%)
     step=100k: Sharpe= -5.869
     step=150k: Sharpe= -2.803
     step=200k: Sharpe= -4.763
     ...
     step=450k: Sharpe= +1.440 (BEST)
   ```
   - 200k+ 스텝을 "거의 random에서 baseline까지" 끌어올리는데 낭비
   - 그 사이 정책이 큰 손실을 학습 → off-policy bias

2. **ATR 고정 정책 = 사용 가능한 전문가**:
   - exp026 ATR (A_b=1.921, C_b=5.719, A_s=0.688, C_s=9.673, n_splits=3) → Val Sharpe 1.978
   - Sub-optimal이지만 "랜덤보다는 훨씬 나은" 정책
   - BC dataset: ATR 정책의 (state, action) trajectory 수십만 샘플 수집 가능

3. **구현 절차**:
   ```python
   # Phase 1: BC dataset 수집
   atr_agent = ATRGridAgent(coefs_from_exp026)
   dataset = []
   for episode in range(N_episodes):
       obs = env.reset()
       while not done:
           action = atr_agent.act(obs)  # action ∈ [0,1]^2 or 5D
           dataset.append((obs, action))
           obs, _, done, _ = env.step(action)

   # Phase 2: PPO policy network를 dataset에 BC pretrain
   for batch in dataset:
       loss = MSE(ppo.policy(obs), action)
       loss.backward()

   # Phase 3: 평소 PPO 학습. 초기 정책이 이미 ATR 수준이므로
   #          학습이 ATR을 "출발선"으로 시작
   ```

4. **예상 효과**:
   - 초반 200k 스텝 낭비 제거 → 같은 compute로 더 깊이 학습
   - 최악의 행동(거래 0건, 과도한 진입)을 처음부터 회피
   - PPO가 ATR이 못 잡는 미세 신호(trend, regime)만 학습하면 됨
   - **exp020/021/022에서 본 "RL = ATR" 결과의 다른 해석 가능**: 어쩌면 RL이 ATR을 출발선으로 시작하지 못해서 ATR 너머를 못 발견한 것일 수도

5. **위험**:
   - BC 정책이 sub-optimal → fine-tune 단계에서 local optima에 갇힐 가능성
   - 해결: BC pretrain 후 entropy를 잠시 크게 잡아 탐색 유도
   - 또는 IQL/CQL로 더 안정적 offline 단계

## 단순한 변형: Action Bias Initialization

본격적 BC가 부담스러우면 더 단순:

```python
# PPO 정책 네트워크의 마지막 layer bias를 ATR 정책 값에 맞춰 초기화
# 예: ATR이 항상 [0.5, 0.0] 출력하면, 마지막 bias = [0, -∞]
# (tanh squashing 가정: 0.5 → 0, 0 → -∞ before squash)

policy.action_net.bias = init_to_match_atr()
policy.action_net.weight = small_random()  # 거의 0
```

→ 정책이 학습 0 step에서도 ATR과 거의 같은 행동 → 곧바로 양수 reward.
→ 가장 가볍게 시도해볼 수 있는 warm start.

## 백링크

- [[cql_kumar_2020]] — Conservative Q-Learning 상세 (mixed data 시 BC 대안)
- [[policy_gradient_stabilization]] — Online 단계에서 fine-tune 안정화
- [[curriculum_learning]] — 점진적 학습 난이도 조절과 결합
- [[reward_shaping_ng1999]] — 초기 학습 신호 보강

## 출처

- [Should I Use Offline RL or BC? (BAIR Blog)](https://bair.berkeley.edu/blog/2022/04/25/rl-or-bc/)
- [When Should We Prefer Offline RL? (arXiv 2204.05618)](https://arxiv.org/abs/2204.05618)
- [Offline RL for Automated Stock Trading (2023)](https://www.researchgate.net/publication/374722752_Offline_Reinforcement_Learning_for_Automated_Stock_Trading)
- [WSRL (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/504491292cb71e7681eedfe0e602b72f-Paper-Conference.pdf)
