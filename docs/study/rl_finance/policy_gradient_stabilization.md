# PPO 학습 안정화 — 실전 기법 모음

> Schulman et al. (2017). _Proximal Policy Optimization Algorithms._ arXiv: 1707.06347.
> 실용 가이드: SB3 docs, Spinning Up, Andrychowicz et al. (2020) _What Matters in On-Policy RL?_

## 요지

1. PPO의 clipped objective는 학습 안정성을 **자동 보장하지 않음**. 추가 안전장치가 필요.
2. 핵심 안정화 도구: (a) **adaptive KL early stop**, (b) **gradient clipping**, (c) **LR schedule**, (d) **entropy annealing**, (e) **value coefficient 조정**, (f) **advantage normalization**.
3. 우리 exp028/exp029의 학습 불안정 (450k peak 후 oscillation)은 이 도구들의 조합 부족이 원인 가능성 높음.

## PPO 표준 하이퍼파라미터 (SB3 default)

| 파라미터 | Default | 합리적 범위 | 영향 |
|---|---|---|---|
| learning_rate | 3e-4 | [1e-5, 1e-3] | 너무 크면 oscillation, 너무 작으면 학습 정체 |
| n_steps | 2048 | [512, 8192] | rollout 길이. 에피소드와의 비율 중요 |
| batch_size | 64 | [32, 512] | n_steps의 약수여야 함 |
| n_epochs | 10 | [3, 20] | 한 rollout 재사용 횟수. 너무 크면 off-policy bias |
| gamma | 0.99 | [0.9, 0.999] | discount. 에피소드 길이에 따라 조정 |
| gae_lambda | 0.95 | [0.9, 1.0] | GAE bias-variance tradeoff |
| clip_range | 0.2 | [0.1, 0.3] | policy clip. 너무 작으면 학습 느림, 크면 불안정 |
| ent_coef | 0.0 | [0.0, 0.1] | 탐색 유도. policy entropy 감시 |
| vf_coef | 0.5 | [0.25, 1.0] | value loss 가중치 |
| max_grad_norm | 0.5 | [0.5, 1.0] | gradient clipping |
| target_kl | None | [0.01, 0.03] | KL early stop. 설정 권장 |

## 안정화 기법 상세

### 1. KL Early Stop
```python
# PPO의 clip은 KL을 직접 제한하지 못함 (clip = policy ratio 제한)
# target_kl 설정 → 평균 KL이 한계 초과 시 epoch 중단
target_kl = 0.015  # 보통 0.01~0.03

# SB3 PPO에서:
PPO(..., target_kl=0.015)
```

→ **on-policy 가정 깨짐 방지**. policy가 한 번에 너무 멀리 가지 않게.

### 2. Learning Rate Schedule
```python
# Linear decay
lr_schedule = lambda progress: initial_lr * progress  # progress: 1 → 0

# Cosine decay
lr_schedule = lambda progress: initial_lr * 0.5 * (1 + cos(π × (1 - progress)))
```

→ 초반: 크게 학습 / 후반: 미세 튜닝. exp003에서 cosine decay 시도했지만 다른 문제(체결가 버그)에 가려짐.

### 3. Entropy Annealing
```python
# 초기: 탐색 충분히 (ent_coef=0.05)
# 후기: 수렴 집중 (ent_coef=0.001)
ent_schedule = lambda progress: max(0.001, initial_ent * progress)
```

→ exp028 baseline 1.5는 학습 후반에도 ent_coef=0.005 유지. annealing이 oscillation 줄였을 수 있음.

### 4. Advantage Normalization

```python
# SB3 PPO 기본 활성화: normalize_advantage=True
# 매 batch 안에서 (A - mean) / (std + 1e-8)
```

→ 다양한 reward 스케일에 robust. 우리는 이미 활성.

### 5. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
# 보통 max_grad_norm=0.5
```

→ exploding gradient 방지. SB3 PPO 기본 활성화.

### 6. Reward Normalization (VecNormalize)
```python
# SB3
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
```

⚠️ **주의**: 우리 exp006/007에서 VecNormalize가 평가 시 분포 mismatch 일으킨 사례 있음.
- 학습 중에는 도움, 평가 시 statistics 고정 필요
- 또는 reward를 환경 내부에서 미리 normalize (VecNormalize 안 쓰는 경우)

### 7. n_steps vs 에피소드 길이

```
이상적: n_steps ≥ 에피소드 평균 길이
→ 한 rollout 안에 적어도 한 에피소드 완결

우리 exp006~007: max_episode_steps=2016, n_steps=8192 → OK
exp029: ?  확인 필요
```

n_steps가 에피소드보다 짧으면 truncation 빈번 → value bootstrapping 불안정.

## 우리 exp028/exp029 학습 불안정 진단

### 관찰된 증상
```
exp029 학습 곡선:
  step= 450k: Sharpe= +1.44  ← BEST
  step= 500k: +1.37  (회복)
  step= 550k: +0.84  (저하)
  step= 600k: +0.41  (저하)
  step= 650k: +1.31  (회복)
  step= 700k: +0.87
  step= 750k: +0.12  ← Early Stopping
```

### 가설별 처방

| 가설 | 처방 | 우선순위 |
|---|---|---|
| LR이 후반에도 큼 → policy 크게 흔들림 | LR을 1e-4 → 1e-5 linear decay | 높음 |
| Entropy 너무 높음 → 후반에도 탐색 | ent_coef 0.008 → 0.001 annealing | 높음 |
| target_kl 없음 → epoch 안에서 큰 update | target_kl=0.02 설정 | 중간 |
| Reward 스케일 변동 | Differential Sharpe Reward로 교체 | 중간 |
| n_steps 너무 작음 | n_steps 1024 → 2048 또는 4096 | 낮음 |
| value loss 너무 큼 | vf_coef 0.5 → 0.25 | 낮음 |

### Andrychowicz et al. (2020) "What Matters in On-Policy RL" 핵심
- **가장 중요**: normalize advantage, large batch, value clipping
- **중간**: GAE λ, learning rate, n_epochs
- **거의 무관**: optimizer choice (Adam이 표준), exact entropy bonus 값

## 즉시 적용 가능한 exp030 설계

```yaml
# 보수적 안정화 패키지
ppo:
  learning_rate:
    type: linear_schedule
    start: 3e-4
    end: 1e-5
  n_steps: 4096
  batch_size: 256
  n_epochs: 10
  target_kl: 0.02         # ★ 새로 추가
  ent_coef:
    type: linear_schedule  # ★ annealing
    start: 0.01
    end: 0.001
  vf_coef: 0.5
  max_grad_norm: 0.5
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2

callbacks:
  early_stopping:
    metric: val_sharpe
    patience: 6
    min_delta: 0.05
```

## 백링크

- [[ppo_schulman_2017]] — Bundle B의 PPO 논문 정리와 교차참조
- [[differential_sharpe_moody2001]] — 더 안정적인 reward로 교체
- [[reward_shaping_ng1999]] — Potential-based shaping으로 학습 신호 개선

## 출처

- [SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [PPO Hyperparameters and Ranges (Medium)](https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe)
- [Reparameterization PPO (2025)](https://arxiv.org/html/2508.06214)
- [Schulman et al. (2017) PPO Paper](https://arxiv.org/abs/1707.06347)
