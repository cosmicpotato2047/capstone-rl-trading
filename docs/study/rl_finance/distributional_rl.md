# Distributional RL in Trading (C51, QR-DQN, IQN)

> Bellemare, Dabney, Munos (2017). _A Distributional Perspective on Reinforcement Learning._ ICML. (C51)
> Dabney et al. (2018). _Distributional RL with Quantile Regression._ AAAI. (QR-DQN)
> Dabney et al. (2018). _Implicit Quantile Networks for Distributional RL._ ICML. (IQN)

## 요지

1. 일반 RL은 **value의 기댓값 E[Q(s,a)]** 만 학습. Distributional RL은 **value의 전체 분포 Z(s,a)** 를 학습.
2. 분포를 학습하면 **CVaR, VaR, 분산** 같은 위험 지표를 정책에 직접 통합 가능 → "risk-sensitive RL".
3. 트레이딩에 자연스럽게 맞는 이유: 금융 수익률이 두꺼운 꼬리(fat tail), 비대칭(skewness) 분포라서 기댓값만으로는 위험 표현 불가.

## 핵심 알고리즘

| 알고리즘 | 분포 표현 | 핵심 아이디어 |
|---|---|---|
| **C51** (2017) | 51개 고정 atom (value axis 고정, 확률 학습) | KL divergence로 distributional Bellman 갱신 |
| **QR-DQN** (2018) | N개 quantile (확률 고정, value 학습) | Quantile regression, Wasserstein loss |
| **IQN** (2018) | Quantile function 자체를 신경망으로 (continuous) | Implicit quantile sampling |
| **D4PG** | 연속 action용 distributional DDPG | Distributional + actor-critic |
| **Distributional SAC** | 연속 action용 distributional SAC | Maximum entropy + distribution |

## CVaR 기반 risk-sensitive 정책

```
일반 RL:    π* = argmax E[Z(s,a)]
CVaR-RL:    π* = argmax CVaR_α(Z(s,a))
            = argmax E[Z | Z ≤ VaR_α]

α=0.05: 하위 5% 시나리오의 기댓값 최대화 → 매우 보수적
α=0.50: 중간값 → 약간 보수적
α=1.00: 전체 평균 → 위험중립 (일반 RL)
```

→ **α를 sliding하면서 학습하면 risk-aversion 정도가 조절 가능한 정책**을 얻을 수 있다.

## 트레이딩 적용 사례

### Risk-averse policies for natural gas futures (2025, arXiv 2501.04421)
- C51과 IQN을 CVaR 최대화로 학습
- 다양한 α 값에서 명확한 risk-aversion spectrum 형성
- C51 > 일반 DQN: 성능 +32% (천연가스 선물)
- QR-DQN은 trading 환경에서 less predictable (분포 표현이 거친 quantile이라 derivative가 까다로움)

### Risk preference adaptive (2025)
- 투자자의 risk preference에 맞춰 α를 동적으로 결정
- 단일 모델로 위험 보수~위험 추구 정책 모두 표현

## 우리 프로젝트와의 연결점

1. **현재 우리는 expected return만 보고 학습 중**:
   ```
   reward = (equity_t - equity_{t-1}) / start_capital
   PPO가 학습하는 것 = E[Σ γ^t · reward]
   ```
   → 꼬리 위험(2022 BTC 폭락 같은 시나리오)에 대한 정책 인식 없음.

2. **exp027_rl의 asymmetric reward는 distributional RL의 "가난한 사람 버전"**:
   - asymmetric beta = 손실을 더 페널티 → CVaR 정책과 비슷한 효과를 reward shaping으로 흉내
   - 더 깔끔한 정식: **PPO를 distributional PPO로 교체** 또는 **CVaR objective**로 변경

3. **즉시 적용 가능한 변형**:
   - SB3의 PPO를 IQN-based actor-critic (또는 D-SAC)으로 교체
   - α=0.10~0.50 다양화하여 risk profile별 정책 학습
   - 동일 환경에서 PPO vs Distributional PPO 비교 → 논문 ablation

4. **그리드 봇과의 특별한 적합성**:
   - 그리드 봇 분포 = "꾸준한 작은 이익 + 가끔 큰 손실" (short-vol)
   - CVaR로 학습하면 "추세장 급락 시 포지션 축소" 같은 행동을 자발적으로 학습할 가능성
   - 현재 exp020/021에서 보이는 "항상 100% 진입" 정책의 대안

5. **구현 비용**:
   - SB3에는 distributional PPO 없음 → tianshou 또는 직접 구현 필요
   - 학습 시간 2~3배 증가 가능
   - 1차 시도는 IQN-based off-policy로 시작 (구현 부담 적음)

## 한계

- 트레이딩 환경의 reward 분포가 학습 가능할 만큼 안정적인가? (비정상성 문제)
- CVaR 최대화가 정말 robust한 정책을 만드는가, 아니면 단지 보수적인 정책을 만드는가? (논쟁 중)

## 백링크

- [[differential_sharpe_moody2001]] — 단일 reward로 risk-sensitivity 달성하는 대안
- [[risk-adjusted return]] — CVaR/VaR 정의
- [[prospect_theory_kahneman]] — 행동경제학적 risk aversion 근거

## 출처

- [Risk-averse policies for natural gas futures (arXiv 2501.04421)](https://arxiv.org/abs/2501.04421)
- [Distributional RL Part 1 (Medium)](https://medium.com/analytics-vidhya/distributional-reinforcement-learning-part-1-c51-and-qr-dqn-a04c96a258dc)
- [IQN paper (Dabney 2018)](https://proceedings.mlr.press/v80/dabney18a/dabney18a.pdf)
- [Risk preference adaptive distributional RL (2025)](https://www.sciencedirect.com/science/article/abs/pii/S1568494625015820)
