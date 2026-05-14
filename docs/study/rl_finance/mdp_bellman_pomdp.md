# MDP, Bellman Equation, POMDP — RL의 수학적 토대

> Sutton & Barto (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. Ch. 3, 4.
> Howard (1960). _Dynamic Programming and Markov Processes._

## 요지

1. **MDP (Markov Decision Process)** 는 RL의 수학적 framework. State, Action, Transition, Reward, Discount 다섯 요소.
2. **Bellman equation**: value function의 재귀적 정의. 모든 RL 알고리즘이 이를 어떻게 풀지의 변형.
3. **POMDP (Partially Observable MDP)** 는 실제 시장에 더 정확한 모델 — 우리는 시장의 일부만 관측한다.

## MDP 5-tuple

```
M = ⟨S, A, P, R, γ⟩

S: 상태 공간
A: 행동 공간
P(s' | s, a): 전이 확률
R(s, a, s'): 보상 함수
γ ∈ [0, 1]: 할인율
```

**Markov property**: $P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)$

→ 현재 state가 미래에 필요한 모든 정보를 담음.

## Bellman Equation

### Bellman expectation (policy π에 대해)
```
V^π(s) = E_π[R + γ V^π(S') | S=s]
       = Σ_a π(a|s) Σ_s' P(s'|s,a) [R + γ V^π(s')]

Q^π(s,a) = E_π[R + γ Q^π(S', A') | S=s, A=a]
        = Σ_s' P(s'|s,a) [R + γ Σ_a' π(a'|s') Q^π(s', a')]
```

### Bellman optimality (최적 정책 π*에 대해)
```
V*(s) = max_a Σ_s' P(s'|s,a) [R + γ V*(s')]
Q*(s,a) = Σ_s' P(s'|s,a) [R + γ max_a' Q*(s', a')]
```

→ Bellman equation을 **푸는 두 가지 접근**:
- **Dynamic Programming**: P, R 다 알 때 (Policy Iteration, Value Iteration)
- **Sample-based RL**: P, R 모를 때 (Monte Carlo, TD-learning, Q-learning, Policy Gradient)

## Discount γ의 의미

```
G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ...
    = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

| γ 값 | 의미 | 유효 horizon | 우리 프로젝트 적용 |
|---|---|---|---|
| 0.9 | 매우 단기 | ~10 step | 너무 짧음 |
| 0.99 | 표준 | ~100 step | 1h봉 100시간 ≈ 4일 |
| 0.999 | 장기 | ~1000 step | ≈ 42일 (사이클 여러 개 포함) |

→ exp006 이후 우리는 **에피소드 단축** 전략 (2016 step / 12주) + γ=0.99
→ exp020 이후 일부 실험에서 γ=0.999로 장기 사이클 대응
→ exp022 Optuna에서 γ=0.985~0.999 범위 선호 확인

## POMDP — 실제 시장의 정확한 모델

```
POMDP M = ⟨S, A, O, P, R, Ω, γ⟩

O: 관측(observation) 공간 (S와 다를 수 있음)
Ω(o | s', a): 관측 확률
```

핵심 차이: 에이전트가 state s를 **직접 보지 않고 observation o**만 본다.

### 시장이 POMDP인 이유

| 진짜 state (관측 불가) | 우리가 보는 observation |
|---|---|
| 모든 거래자의 포지션, 의도 | 가격, 거래량, 호가창 일부 |
| 미공개 정보 (실적 발표 전, 내부자) | 공개된 가격 시계열 |
| 시장 조성자의 inventory | bid/ask spread |
| 미래 펀더멘털 변화 | 과거 가격 + 매크로 지표 |

### POMDP 대응 방법

1. **History/Frame stacking**: 최근 k개 observation을 state에 포함 → Markov 근사
2. **Recurrent policy (LSTM/Transformer)**: hidden state로 belief 유지
3. **Belief state**: P(s | history)를 명시적으로 추정 (Bayesian RL)

### 우리 프로젝트의 POMDP 처리

- State 7D~9D: rolling z-score, ATR, holdings 등으로 압축한 **observation**
- 근본적으로 POMDP를 MDP로 근사 — 일부 정보 누락 불가피
- 보강 가능 방향:
  - **LSTM policy**: SB3의 RecurrentPPO 사용
  - **Frame stacking**: 최근 k=24봉 (1일치) state를 stack
  - **Order book features** (라이브 트레이딩 단계): bid/ask spread, depth 추가

## 우리 프로젝트와의 연결점

1. **현재 MDP 가정의 한계**:
   - state에 trend_72h, trend_720h는 포함했지만 **호가창 정보 없음**
   - 시장 조성자/대형 매도자의 inventory 미관측
   - → 본질적으로 POMDP를 MDP로 근사 중

2. **RecurrentPPO 시도해볼 가치**:
   - 현재 PPO는 single-step state만 봄
   - 정책 포화 현상(exp016 [0,0] 수렴)이 POMDP를 MDP로 강제 변환한 부작용일 수 있음
   - LSTM이 hidden state로 belief 유지하면 다른 행동 발현 가능성

3. **γ 설계 재검토**:
   - 그리드 사이클이 평균 4시간 → γ=0.99에서 γ^4 ≈ 0.96, γ^24 ≈ 0.78
   - 다음 사이클을 거의 무시 → "지금 이 거래만 잘하기"로 수렴 (exp022 RL=ATR 현상 일부 설명)
   - γ=0.999로 늘리면 ~24시간 사이클 평가 가능
   - exp029의 어려움 원인 중 하나일 수 있음

4. **Reward design with γ**:
   - DSR (Differential Sharpe Ratio)를 step reward로 쓰면 γ에 덜 민감
   - 누적 Sharpe는 에피소드 끝에서만 정의 → sparse → γ 영향 큼
   - DSR은 매 step에서 정의 → dense → γ 영향 적음

## 백링크

- [[ppo_schulman_2017]] — Bellman을 푸는 한 가지 방법 (policy gradient)
- [[differential_sharpe_moody2001]] — Bellman과 무관하게 direct policy 학습 (RRL)
- [[hierarchical_rl_trading]] — Semi-MDP로 자연스럽게 확장
- [[ddpg_continuous_control]] — Bellman을 연속 action에 적용

## 출처

- [Sutton & Barto — RL: An Introduction (2nd ed.)](http://www.cmap.polytechnique.fr/~lepennec/files/RL/Sutton.pdf)
- [Sutton & Barto Ch.3 MDP summary](https://lcalem.github.io/blog/2018/09/23/sutton-chap03-mdp)
- [Berkeley CS287 — Exact MDP methods](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf)
