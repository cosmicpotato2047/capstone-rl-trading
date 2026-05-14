# Curriculum Learning + Domain Randomization

> Bengio et al. (2009) — Curriculum Learning. Narvekar et al. (2020) — Curriculum Learning for RL: Survey (JMLR).
> Tobin et al. (2017) — Domain Randomization for Transferring Deep Neural Networks (Sim-to-Real).
> ADR (Automatic Domain Randomization): OpenAI (2019).

## 요지

1. **Curriculum learning**: 쉬운 환경 → 어려운 환경으로 점진적 학습. 인간 학습 메타포.
2. **Domain randomization**: 학습 중 환경 파라미터를 랜덤화 → 정책 robust + sim2real gap 완화.
3. 둘은 결합 가능: **ADR (Automatic Domain Randomization)** — 정책 성능을 보고 randomization 강도를 자동 조절.
4. 우리 프로젝트 직결: Train(2017-2022) → Val(2023) → Test(2024+) 일반화 실패 사례에 정확한 해결책.

## Curriculum Learning 정식

```
Task 분포 T = {t_1, t_2, ..., t_N}
난이도 측정: d(t_i)

Curriculum c = sequence of distributions:
  D_0 → D_1 → ... → D_K
  s.t. easier(D_0) → harder(D_K)
  
정책 학습:
  for k in 0..K:
    학습 from D_k
    if performance ≥ threshold: move to D_{k+1}
```

→ 핵심: "현재 수준에서 풀 수 있는 가장 어려운 문제"를 자동 선택.

## Domain Randomization 정식

```
표준 RL:    학습 환경 E (고정)  →  테스트 환경 E* (실거래)
DR-RL:     학습 분포 {E(φ) : φ ~ p(φ)}  →  단일 정책 π
           (φ: 환경 파라미터)

정책이 다양한 φ에서 작동 → unseen φ (실거래)에도 robust
```

### 트레이딩 적용 가능 randomization 파라미터

| 파라미터 | 범위 (예시) | 효과 |
|---|---|---|
| Fee rate | [0.04%, 0.08%] | 다양한 거래소/등급 robust |
| Slippage | [0%, 0.1%] | 유동성 변화 robust |
| ATR window | [120, 240] | window 선택 sensitivity 감소 |
| Initial cash | [5k, 50k] | scale invariance |
| Episode start time | random in train | regime 분포 균형 학습 (exp006/007에서 이미 사용) |
| Volatility shock | 매 K step마다 ATR ±50% noise | 변동성 regime 변화 시뮬 |
| Maker fee rebate on/off | random | 거래소 정책 변화 robust |

→ 우리는 **episode start randomization은 이미 적용 중**. 나머지는 미적용.

## ADR (Automatic Domain Randomization)

```
표준 DR: φ ~ Uniform([φ_min, φ_max])  (고정 범위)
ADR:    범위를 [φ_min(t), φ_max(t)]로 동적

정책이 잘 하면 범위 확대 (어려워짐)
정책이 못 하면 범위 축소 (쉬워짐)
```

→ Curriculum learning + Domain randomization의 통합.

## 우리 프로젝트와의 연결점

1. **현재 분포 차이 진단**:
   - Train 2017-2022: BULL/BEAR/SIDEWAYS 혼재, 2022 -64% 폭락 포함
   - Val 2023: 회복+H2 BULL 강세장
   - Test 2024+: ETF 승인, 반감기, 강한 상승 — 이전에 없던 regime
   - **각 분포가 완전히 다름** → 일반화 본질적 어려움

2. **Curriculum 설계 옵션**:

   **Option A: 변동성 기준 curriculum**
   ```
   Stage 1: 저변동성 구간만 (ATR < median)  ← 쉬움
   Stage 2: 중간 변동성 추가
   Stage 3: 고변동성 + 점프 구간 포함
   ```

   **Option B: 시간 순서 curriculum**
   ```
   Stage 1: 2017-2019 (초기 BTC)
   Stage 2: + 2020-2021 (코로나 + 강세장)
   Stage 3: + 2022 (폭락)
   Stage 4: + 2023-2024 (회복)
   ```

   **Option C: 환경 단순화 curriculum**
   ```
   Stage 1: fee=0, slippage=0, n_splits=2 (단순)
   Stage 2: fee=0.05%, slippage=0
   Stage 3: fee=0.05%, slippage=0.02%, partial fill 가능
   ```

3. **Domain randomization 즉시 적용 가능**:
   ```yaml
   domain_randomization:
     enabled: true
     params:
       fee_rate:
         distribution: uniform
         range: [0.0004, 0.0008]
       slippage_rate:
         distribution: uniform
         range: [0.0, 0.0005]
       initial_cash:
         distribution: log_uniform
         range: [5000, 50000]
   ```

   → 매 에피소드마다 환경 파라미터 재샘플링.
   → 정책이 단일 (fee, slippage) 조합에 과적합 안 됨.

4. **exp029의 학습 불안정과의 관계**:
   - exp029에서 학습 oscillation 관찰 (450k peak 후 진동)
   - 가설: 같은 분포 안에서 local optima 사이 진동 → curriculum 없이 어려운 분포 한꺼번에 학습 시 발생
   - **Curriculum + Warm start ([[offline_rl_warm_start]]) 결합**이 가장 강력한 처방 후보

5. **2학기 자산 확장 시 자연스러운 적용**:
   - Stage 1: BTC만 (현재 학습)
   - Stage 2: BTC + ETH (유사 자산)
   - Stage 3: + 주식 (다른 시간 스케일)
   - Stage 4: + 외환 (다른 변동성 특성)
   → Curriculum이 sub-asset에서 multi-asset으로의 자연스러운 다리

## 구현 패키지 (다음 실험 후보)

```python
# pseudo-code
class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, stages):
        self.stages = stages
        self.current_stage = 0
        self.recent_rewards = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            self.recent_rewards.append(info.get('episode_return', reward))
            if len(self.recent_rewards) >= 10:
                avg = mean(self.recent_rewards[-10:])
                if avg > self.stages[self.current_stage]['threshold']:
                    self.advance_stage()
        return obs, reward, done, info

    def reset(self):
        # Stage별 환경 설정 적용
        config = self.stages[self.current_stage]
        self.env.configure(config)
        return self.env.reset()
```

## 한계 및 위험

- Curriculum 단계 설계 자체가 hyperparameter (어떤 순서? 언제 advance?)
- DR이 너무 강하면 단순한 정책으로 수렴 (모든 환경에서 평균적으로 작동)
- 학습 시간 2~3배 증가 가능

## 백링크

- [[sim2real_finance]] — DR이 sim2real gap의 직접 해결책
- [[offline_rl_warm_start]] — Stage 1을 BC로 시작
- [[policy_gradient_stabilization]] — Curriculum이 학습 안정성 개선

## 출처

- [Curriculum for RL (Lil'Log)](https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/)
- [Narvekar et al. (2020) — Curriculum Learning for RL Survey (JMLR)](https://jmlr.org/papers/volume21/20-212/20-212.pdf)
- [ADR: Train Hard, Transfer Smart](https://medium.com/@kdk199604/adr-train-hard-transfer-smart-bad19432c3b9)
- [Balanced Domain Randomization for Safe RL (MDPI)](https://www.mdpi.com/2076-3417/14/21/9710)
- [Automatic Curriculum Learning for Driving (2025)](https://arxiv.org/html/2505.08264)
