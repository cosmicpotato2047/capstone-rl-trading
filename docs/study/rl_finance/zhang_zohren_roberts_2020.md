# Deep Reinforcement Learning for Trading — Zhang, Zohren, Roberts (2020)

> Zhang, Z., Zohren, S., & Roberts, S. (2020). _Deep Reinforcement Learning for Trading._ The Journal of Financial Data Science, 2(2), 25–40. arXiv: 1911.10107.

## 요지

1. **다자산(50개 가장 유동성 높은 선물, 2011~2019) 전반에 DRL을 적용**한 대표 실증 연구.
2. 핵심 아이디어 두 가지: (a) **volatility scaling reward** — 변동성에 따라 포지션을 자동 스케일링, (b) 이산/연속 action space 모두 비교.
3. **고전적 time-series momentum + transaction cost가 큰 환경에서도 DRL이 outperform**. 횡보장에서 hold/scale-down 가능.

## 핵심 설계

### State
- 가격 시계열 + 거시지표
- normalized return: $r_t / \sigma_t$ — 변동성 정규화로 자산 간 비교 가능하게

### Action space (두 변형 모두 실험)
- **Discrete**: {long, short, hold} (3)
- **Continuous**: position size ∈ [-1, +1]
- 결론: 연속이 일반적으로 우세하지만 자산에 따라 갈림

### Volatility scaling reward (이 논문의 진짜 기여)

```
R_t = A_t · r_t · σ_target / σ_t

A_t: 현재 action (포지션)
r_t: 자산 수익률
σ_t: 최근 변동성
σ_target: 목표 변동성 (고정)
```

→ **변동성이 크면 포지션이 자동으로 축소**되는 효과. 학습 결과가 한 자산에 과적합되는 것을 막고, 위험 조정 보상을 자연스럽게 부여.

### 알고리즘
- DQN (discrete), Policy Gradient (REINFORCE-like), Actor-Critic
- 비교 결과: A2C와 PPO가 가장 안정적

## 실험 결과 (요약)

- 4대 자산군(commodities, equity indices, fixed income, FX) 모두에서 DRL이 time-series momentum 대비 outperform
- **거래비용 차감 후에도 양의 PnL** — 단순 신호 대비 RL의 가치 증명
- 추세장: 포지션을 크게 유지, 횡보장: 포지션 축소 → 명시적으로 학습된 regime 적응

## 우리 프로젝트와의 연결점

1. **volatility scaling reward = 우리의 ATR/price 비례 공식의 학술적 친척.**
   우리는 action을 ATR로 스케일링했지만, Zhang은 **reward**를 변동성으로 스케일링했다. 이론적으로는 reward 쪽이 더 깨끗 (action은 정책이 결정하는 변수, reward는 환경이 주는 시그널).
   → 우리 시스템의 "ATR이 RL을 흡수한다" 발견은 사실 Zhang의 reward scaling 효과가 action 단으로 옮겨간 결과라고 해석 가능.

2. **다자산 일반화 근거**: 이 논문이 50개 선물에서 작동을 보여줬으므로, Semester 2 자산 확장(SOXL, FX, 원자재) 시 "이미 검증된 프레임"을 따라간다는 정당성 확보.

3. **action space 설계 비교**: 이 논문이 discrete vs continuous를 직접 비교 — 우리가 연속 [0,1]²을 선택한 것의 baseline 비교 근거.

4. **차이점 명확화 (논문 디펜스용)**:
   - Zhang: directional trading (long/short/hold)
   - 우리: grid (방향 무관, 양방향 동시 호가)
   - → "Zhang의 reward scaling 아이디어를 directional이 아닌 grid에 적용한 것이 우리 기여"

## 백링크

- [[differential_sharpe_moody2001]] — 또 다른 reward 설계 계보
- [[gort_2022]] — 같은 DRL trading 라인의 후속 작업
- [[avellaneda_stoikov_2008]] — Zhang의 directional 접근과 대비되는 market making 접근

## 출처

- [arXiv 1911.10107](https://arxiv.org/abs/1911.10107)
- [Journal of Financial Data Science](https://jfds.pm-research.com/content/2/2/25/tab-article-info)
- [Semantic Scholar](https://www.semanticscholar.org/paper/Deep-Reinforcement-Learning-for-Trading-Zhang-Zohren/4b27b8b28b0959989e144ac7273aacfe05267cf8)
