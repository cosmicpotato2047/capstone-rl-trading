# Reward Hacking / Specification Gaming in RL Trading

> 핵심 개념: Goodhart's law — "When a measure becomes a target, it ceases to be a good measure."
> 트레이딩 에이전트에 특히 위험한 이유는 reward function 자체가 시뮬레이터의 결함을 그대로 반영하기 때문.

## 요지

1. **Reward hacking** = 에이전트가 reward function의 허점/모호함을 악용해 의도와 다른 방식으로 높은 보상을 받음.
2. RL 트레이딩에서 자주 나타나는 형태: 시뮬레이터의 체결 가정, 시간 누설, lookahead bias, transaction cost 미반영 등이 reward와 결합되어 가짜 알파를 만들어냄.
3. 우리 프로젝트의 exp026 체결가 버그(`next_low/next_high`로 항상 best price 체결)가 **교과서적 reward hacking 사례** — 에이전트가 "탐욕적으로 진입 → 즉시 체결" 정책을 학습했고, 수익률 수조%를 만들었다.

## RL 트레이딩의 전형적 reward hacking 패턴

| 패턴 | 메커니즘 | 우리 프로젝트 사례 |
|---|---|---|
| **체결가 favorable bias** | 시뮬레이터가 항상 봉의 최저/최고에 체결 | exp022~025 (체결가 버그) |
| **Lookahead leak** | state에 미래 정보 포함 (e.g., 다음 봉 일부) | 우리는 없음 (정렬 검증됨) |
| **수수료 underestimate** | 실제 슬리피지/스프레드 미반영 | maker fee만 사용, 슬리피지 0 가정 |
| **Reward sparse + dense 혼용** | 사이클 보너스 + 스텝 보상 이중 계산 | exp003 fee 이중차감 버그 (이미 fix) |
| **Survivorship bias** | 학습기에 살아남은 자산만 사용 | BTC 단일 자산 → 해당사항 적음 |
| **Re-entry 보상 누수** | 같은 가격에 매도 후 즉시 재매수로 reward 갱신 | 사이클 구조로 부분 방어 |

## Skalse et al. (2022) — Defining and Characterizing Reward Hacking

```
공식 정의:
proxy reward function R_proxy 가 true reward R_true 의
"hackable refinement"이라면, 일부 정책 π에 대해
  E[R_proxy(π)] > E[R_proxy(π')] BUT E[R_true(π)] < E[R_true(π')]

즉, proxy 기준으로는 π가 우월하지만 진짜 목표 기준으로는 π'가 우월.
```

→ 우리 시스템에서: R_proxy = "Sharpe in backtest", R_true = "Sharpe in live trading"
→ Sim2Real gap이 클수록 reward hacking 여지가 커짐.

## 탐지/완화 기법

1. **다중 환경 평가** — 동일 에이전트를 약간 다른 시뮬레이터에서 평가, 큰 차이 = hacking 의심
2. **Adversarial test** — 시뮬레이터에 노이즈 추가 (slippage, fee 변동) 후에도 robust한지
3. **PBO 계산** ([[gort_2022_crypto_overfitting]]) — 통계적으로 다중 trial의 우연성 차감
4. **Behavioral inspection** — 에이전트의 행동 패턴이 경제적으로 말이 되는지 (e.g., 극도로 잦은 거래 = 의심)
5. **Live forward test (paper trading)** — 최종 검증

## 우리 프로젝트와의 연결점

1. **exp026 체결가 수정은 reward hacking 발견 → 차단의 정석 사례.**
   논문/디펜스에서 "reward hacking을 어떻게 발견하고 차단했는가"의 case study로 활용 가능.

2. **현재도 잠재적 hacking 채널 존재**:
   - 슬리피지 0 가정 — 실거래에서 무너질 수 있음
   - 부분체결 불가 가정 — 큰 주문이 호가창에 부담 줄 때
   - 1h 캔들 내 가격 movement 추정 (open→high→low→close 순서 가정) — 실제와 다를 수 있음

3. **방어 액션 (우선순위)**:
   - **D-LV1**: 슬리피지 0.02% 추가 → 모든 거래 비용 ↑ → reward 재학습 필요할 수 있음
   - **D-LV2**: 호가창 부분체결 모델 도입 (Bundle D4에서 상세히)
   - **D-LV3**: Paper trading 데이터로 sim2real gap 측정 → reward 보정

4. **exp027_rl의 asymmetric reward가 hacking을 줄이는지?**
   beta=2.0이 손실에 가중 → 시뮬레이터의 favorable 체결 가정에 덜 의존하는 보수 정책 학습 →
   reward hacking 내성 향상의 부산물일 수 있음. 가설 검증 필요.

## 백링크

- [[gort_2022_crypto_overfitting]] — PBO로 통계적 hacking 탐지
- [[sim2real_finance]] — Sim-to-real gap이 hacking의 출처
- [[reward_shaping_ng1999]] — Potential-based shaping은 reward hacking을 이론적으로 회피
- [[The Dangers of Backtesting]] (López de Prado) — 백테스트 함정 일반론

## 출처

- [Lilian Weng — Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [Skalse et al. (2022) — Defining and Characterizing Reward Hacking](https://arxiv.org/pdf/2209.13085)
- [Reward Hacking — Wikipedia](https://en.wikipedia.org/wiki/Reward_hacking)
