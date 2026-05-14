# Differential Sharpe Ratio — Moody & Saffell (1998/2001)

> Moody, J., & Saffell, M. (2001). _Learning to trade via direct reinforcement._ IEEE Transactions on Neural Networks, 12(4), 875–889.
> 원형: Moody, J., Wu, L., Liao, Y., & Saffell, M. (1998). _Performance functions and reinforcement learning for trading systems and portfolios._ Journal of Forecasting, 17, 441–470.

## 요지

1. 강화학습 에이전트가 매 스텝 **온라인으로 Sharpe ratio를 미분 가능한 형태로 보상**받게 만든 최초의 시도.
2. 누적 Sharpe = (mean return) / std(return) 은 에피소드 끝에서만 계산 가능 → 학습 신호로 직접 사용 불가. **DSR(Differential Sharpe Ratio)** 은 이를 EMA(지수이동평균) 두 개의 미분으로 분해해 매 스텝 reward로 활용한다.
3. Direct Reinforcement (RRL, Recurrent Reinforcement Learning) 프레임 — value function을 추정하지 않고 정책을 직접 최적화. Bellman의 curse of dimensionality를 우회.

## 핵심 수식

EMA 두 개로 1차/2차 모멘트를 추적:

```
A_t = A_{t-1} + η (R_t - A_{t-1})           # mean of returns
B_t = B_{t-1} + η (R_t² - B_{t-1})          # 2nd moment
S_t = A_t / sqrt(B_t - A_t²)                # Sharpe estimate
```

DSR은 η에 대한 S_t의 1차 테일러 전개에서 1차 항만 취한 것:

```
D_t = (B_{t-1} ΔA_t  -  ½ A_{t-1} ΔB_t) / (B_{t-1} - A_{t-1}²)^{3/2}

여기서 ΔA = R_t - A_{t-1}, ΔB = R_t² - B_{t-1}
```

이 D_t를 매 스텝 reward로 쓰면 **에이전트가 단순 수익이 아니라 위험조정수익을 직접 최대화**하게 된다.

## 변형: Differential Downside Deviation Ratio (D3R)

상방 변동성은 위험으로 치지 않는 Sortino 버전. 그리드 봇처럼 "꾸준한 작은 이익 + 가끔 큰 손실" 비대칭 분포에 더 적합.

```
하방 편차만 EMA로 추적:
DD_t = DD_{t-1} + η (min(R_t, 0)² - DD_{t-1})
D3R_t = ... (Sortino 형태)
```

## 우리 프로젝트와의 연결점

1. **exp027_rl의 asymmetric reward (`reward_loss_beta=2.0`)의 학술적 뿌리.**
   beta로 손실에만 가중 = D3R의 비대칭 발상과 동일 계열.
   현재 임시방편으로 beta를 잡아 둔 것을 **DSR/D3R 표준 정식으로 재정의**하면 디펜스 강화 가능.

2. **현재 reward = `(equity_t - equity_{t-1}) / start_capital`** (symmetric).
   이대로면 에이전트가 변동성을 자발적으로 통제하지 않음. DSR로 바꾸면 Sharpe를 직접 최적화하므로 exp028/exp029에서 본 "0거래 수렴" 또는 "학습 불안정" 문제를 reward 차원에서 완화 가능.

3. **구현 비용 극소**: EMA 두 개 + 3줄 수식. `_step_reward` 함수만 교체.
   η는 시계 단위(1h봉)에 맞춰 1/N (N = 1~3일치 봉 수)로 두면 합리적.

4. **단점 인식**: DSR도 Sharpe의 정규분포 가정 한계를 그대로 물려받음. Sortino 변형(D3R) 또는 [[risk-adjusted return]]에서 본 Calmar/MDD 패널티와 결합하는 것이 더 안전.

## 후속 연구

- Almahdi & Yang (2017) — RRL을 LSTM으로 확장, Calmar ratio reward 사용
- Deng et al. (2017) — Deep direct reinforcement learning
- 현대 PPO 기반 구현에서도 DSR을 step reward로 직접 사용 가능 (Continuous action에 무관)

## 백링크

- [[risk-adjusted return]] — Sharpe/Sortino/Calmar 정의
- [[reward_shaping_ng1999]] — DSR이 potential-based shaping과 동치인지 검토
- [[prospect_theory_kahneman]] — asymmetric reward의 인지심리학적 근거

## 출처

- [Moody & Saffell (2001) IEEE TNN](https://www.semanticscholar.org/paper/Learning-to-trade-via-direct-reinforcement-Moody-Saffell/1a4999c918c6206cd9804c48f7dce1bac6ec5b4a)
- [Moody & Saffell (1998) NIPS — Reinforcement Learning for Trading](https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading)
