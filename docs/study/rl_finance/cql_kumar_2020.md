# Conservative Q-Learning (CQL) — Kumar et al. (2020)

> Kumar, A., Zhou, A., Tucker, G., Levine, S. (2020). _Conservative Q-Learning for Offline Reinforcement Learning._ NeurIPS 2020. arXiv: 2006.04779.
> 코드: github.com/aviralkumar2907/CQL (rlkit/torch/sac/cql.py)

## 요지

1. **Offline RL의 가장 큰 문제는 distributional shift** — 학습 데이터에 없는 (out-of-distribution, OOD) action의 Q값을 신경망이 멋대로 추정 → 정책이 그 환상에 끌려가 실패.
2. **CQL의 해법**: Q-function이 **진짜 value의 lower bound**가 되도록 학습. "잘 모르면 보수적으로 낮게 추정."
3. **구현 단순**: 기존 Q-learning/Actor-critic에 **conservative regularizer 한 줄 추가**. SAC, DQN 위에 쉽게 얹음.
4. **결과**: D4RL 벤치마크에서 기존 offline RL 대비 **2~5배 성능**. 특히 multi-modal/복잡 데이터에서 압도적.

## 핵심 수식

### Conservative Regularizer

```
L_CQL(θ) = α · E_{s~D}[ log Σ_a exp(Q_θ(s,a)) − E_{a~π_β(·|s)}[Q_θ(s,a)] ]
         + L_Bellman(θ)

  D:        offline dataset
  π_β:      behavior policy (데이터 수집한 정책)
  α:        conservative 강도 (auto-tuning 가능)
  L_Bellman: 표준 Q-learning loss (TD error)
```

### 두 항의 직관

| 항 | 효과 |
|---|---|
| `log Σ_a exp(Q(s,a))` (softmax-like) | **모든 action**의 Q를 낮추려 함 (OOD 포함) |
| `−E_{a~π_β}[Q(s,a)]` | **dataset에 있는 action**의 Q를 높임 (상쇄) |

→ **순효과**: dataset 안에 없는(OOD) action만 Q 값이 깎임.
→ 정책이 OOD action을 "환상의 고수익"으로 잘못 인식해 따라가는 것을 자동 차단.

### 변형 (CQL-H, CQL-ρ)

| 변형 | 정의 | 사용처 |
|---|---|---|
| **CQL(ℋ)** | μ = uniform action distribution (log Σ exp = soft-max over uniform) | 표준 (D4RL에서 사용) |
| **CQL(ρ)** | μ = 학습된 정책 또는 임의 ρ(a|s) | continuous action에서 더 효율적 |

→ Continuous action에서는 모든 action에 대한 softmax 계산 불가 → ρ-sampling 사용.

## 이론적 보장 (Theorem 3.2, 3.3)

```
Theorem 3.2 (Lower bound):
  α가 충분히 크면 학습된 Q^π_CQL ≤ Q^π_true  (pointwise)

Theorem 3.3 (Policy improvement):
  CQL의 lower-bound Q 위에서 정책 개선하면 진짜 환경에서도 개선 보장
```

→ "보수적 Q에서 잘 하는 정책 = 진짜 환경에서도 적어도 그만큼 잘 함" (단, regularizer α 적절히 선택).

## CQL vs BC vs 표준 Offline RL

| 방법 | 분포 shift 대응 | OOD action | 데이터 품질 요구 |
|---|---|---|---|
| **Behavioral Cloning (BC)** | 분포 안 벗어남 (단순 모방) | 절대 시도 안 함 | 전문가 데이터 필요 |
| **표준 Q-learning offline** | 안 함 | 환상 추정 → 실패 | 어떤 데이터든 안 됨 |
| **CQL** | 정면 대응 (pessimism) | Q 자동 낮춤 | mixed/sub-optimal 데이터 OK |
| **IQL (Implicit Q-Learning, 2022)** | 다른 방식 (quantile regression) | dataset 안에서만 정책 | mixed 데이터 OK |

→ **CQL의 sweet spot**: 다양한 정책이 섞인 sub-optimal 데이터셋. (혼합 데이터)
→ 전문가만 있는 경우: BC가 더 단순하고 비슷한 성능.

## 우리 프로젝트와의 연결점

### 현재 [[offline_rl_warm_start]] 노트에서 단순 BC를 제안한 이유
- ATR 고정 정책 (exp026 best params) = **단일 sub-optimal 전문가**
- Kumar et al. (2022)의 BC vs Offline RL 결정 기준: **단일 정책이면 BC가 적절**
- → 우리는 1차로 BC로 시도

### CQL이 더 유리해지는 시점
1. **여러 정책 mix**:
   - ATR 다양한 계수 (Optuna trial 50개)
   - 베이스라인 (Fixed Grid 1%, 2%, 5%, ATR k=0.5/1.0/2.0)
   - 랜덤 정책 일부
   - → mixed dataset → BC는 평균 흉내, CQL은 best 부분 추출

2. **데이터에 명백한 실패 사례 포함**:
   - exp025 이전 결과 (체결가 버그로 망친 정책)
   - 의도적 노이즈 추가
   - → BC는 실패도 모방, CQL은 자동 회피

3. **Hierarchical RL과 결합**:
   - 상위 정책 (regime classifier) + 하위 정책 (grid params)
   - 하위에서 다양한 정책 적용 → CQL이 적합

### exp031 → exp031b 가능성

```
exp031 (BC warm-start, 현재 계획):
  ATR 정책 → BC pretrain (10k trajectories) → PPO online

exp031b (CQL alternative, mixed data 있을 때):
  Mixed dataset (ATR 변형 + baselines) → CQL pretrain → PPO online
```

→ exp031 결과가 부족하면 exp031b로 확장.
→ Mixed dataset 수집이 추가 작업이지만 어렵지 않음 (베이스라인 코드 이미 존재).

### 실용적 고려사항

| 항목 | CQL | BC |
|---|---|---|
| 구현 복잡도 | 보통 (regularizer 추가, α auto-tune) | 매우 단순 (지도학습) |
| 컴퓨팅 비용 | Q-learning 수준 (offline) | 학습 빠름 (지도) |
| 우리 환경 적합도 | continuous action → CQL(ρ) | OK |
| SB3 지원 | ✗ (직접 구현 또는 d3rlpy) | 직접 가능 |
| Sample efficiency | 좋음 | 보통 |

→ **즉시 사용 가능한 도구**: `d3rlpy` 라이브러리 (CQL, IQL, BCQ 모두 내장)
   ```python
   from d3rlpy.algos import CQL
   cql = CQL(use_gpu=True, n_action_samples=10)
   cql.fit(dataset, n_epochs=100)
   # 그 다음 PPO online fine-tune
   ```

## CQL의 한계 / 후속 연구

1. **α 선택 까다로움**: 너무 크면 정책이 너무 보수적 (BC와 비슷해짐), 너무 작으면 OOD 환상 못 막음. CQL paper에서 Lagrangian dual로 auto-tune 제안.
2. **Sub-optimal 데이터에서도 sub-optimal에 머무름**: lower bound가 너무 tight하면 dataset의 best policy 정도만 학습.
3. **IQL (Kostrikov 2021)이 더 단순 + 자주 더 좋은 성능**: 우리가 시도한다면 CQL과 IQL 둘 다 비교.
4. **MCQ (Mildly Conservative Q-Learning, 2022)**: CQL의 over-pessimism 완화 변형.

## 트레이딩 응용 사례

- **Offline RL for Stock Trading** (2023, ResearchGate 374722752): CQL + behavior cloning regularization → Sharpe ratio 향상 보고
- **D4RL benchmark에 finance 없음**: 우리가 적용한다면 다소 frontier 영역

## 우리 프로젝트의 즉시 활용 시나리오

**시나리오 A — 단순 (먼저 시도)**:
- exp031: ATR 정책 → BC pretrain → PPO online
- 결과 평가 후 다음 단계 결정

**시나리오 B — Mixed data + CQL (B로 확장)**:
- Dataset 구성:
  - ATR 정책 trajectory (5000개)
  - Fixed Grid 1%/2%/5% trajectory (각 1000개)
  - ATR k=0.5/1.0/2.0 trajectory (각 1000개)
  - 약간의 랜덤 정책 (1000개)
  - → 총 ~12000 trajectory, 다양한 행동 분포
- CQL pretrain (d3rlpy, 100 epochs)
- PPO online fine-tune
- 비교: BC pretrain vs CQL pretrain

**시나리오 C — Hierarchical + CQL (장기)**:
- 상위: regime classifier (LSTM, supervised)
- 하위: CQL로 학습한 regime-specific policy
- Online step에서 regime → 정책 선택

## 백링크

- [[offline_rl_warm_start]] — BC 단순 버전. CQL이 mixed data에서 더 유리
- [[ppo_schulman_2017]] — Online fine-tune phase
- [[ddpg_continuous_control]] — Off-policy actor-critic 토대. CQL은 SAC 기반 변형이 표준
- [[distributional_rl]] — 또 다른 Q-function 정교화 방향
- [[reward_hacking]] — OOD action을 환상으로 인식하는 것은 reward hacking의 한 형태

## 출처

- [Kumar et al. (2020) arXiv 2006.04779](https://arxiv.org/abs/2006.04779)
- [NeurIPS 2020 PDF](https://papers.neurips.cc/paper_files/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf)
- [Official Implementation (aviralkumar2907/CQL)](https://github.com/aviralkumar2907/CQL)
- [BAIR Blog — Offline RL with Conservative Algorithms](https://bair.berkeley.edu/blog/2020/12/07/offline/)
- [Project page](https://sites.google.com/view/cql-offline-rl)
- [Mildly Conservative Q-Learning (후속, 2022)](https://openreview.net/forum?id=VYYf6S67pQc)
