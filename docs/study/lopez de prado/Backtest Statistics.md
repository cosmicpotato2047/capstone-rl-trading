# Chapter 14: Backtest Statistics — 세부 구성

López de Prado가 직접 정리한 9개 절 구조 (14.1 Motivation, 14.2 Types of Backtest Statistics, 14.3 General Characteristics, 14.4 Performance, 14.5 Runs, 14.6 Implementation Shortfall, 14.7 Efficiency, 14.8 Classification Scores, 14.9 Attribution).

이 챕터의 핵심 메시지는 단순함: 어떤 백테스트 방식(walk-forward, CV/CPCV, 합성 데이터)을 선택하든 결과는 일련의 통계로 보고해야 하며, 투자자(여기서는 심사위원)는 그 통계로 전략을 비교·판단한다. 즉, 보상함수에 무엇을 넣을지 정하기 전에 **무엇을 측정할지** 먼저 정의해야 한다는 관점.

## 절별 내용과 보상함수 설계에의 응용

### 14.3 General Characteristics — 메타 통계

시간 범위, 빈도, AUM, 레버리지, 롱/숏 비율, 베팅 빈도, 평균 보유 기간, 회전율, 상관성 등. **보상함수 직접 입력은 아님.** 하지만 그리드 봇 특성상 "베팅 빈도"와 "평균 보유 기간"은 논문 표에 반드시 들어가야 함. 보상함수가 무의식적으로 과도한 거래를 유도하는지 진단할 지표.

### 14.4 Performance — **보상함수 1순위 후보군**

이 절이 졸업 프로젝트에 가장 직접적임. 다루는 지표:

- **PnL (절대/상대)**: 가장 단순한 reward. RL에서 흔히 쓰지만 분산이 커서 PPO 학습 불안정의 주범.
- **수익률 시계열의 분포 특성**: 평균, 표준편차, 왜도(skewness), 첨도(kurtosis). BTC는 fat-tail이 강해서 평균 수익률만으로는 위험이 가려짐.
- **Sharpe Ratio**: 가장 표준적. 다만 **단일 에피소드 내에서 매 스텝마다 계산하면 노이즈가 큼.** Differential Sharpe Ratio (Moody & Saffell, 1998)로 변형해서 step-wise reward로 쓰는 게 RL 관례 — 이 책은 안 다루지만 응용 발판은 여기서 나옴.
- **Probabilistic Sharpe Ratio (PSR)**: Sharpe가 0보다 클 확률. 짧은 백테스트의 신뢰성 보정. **졸업 논문 평가 섹션에 PSR을 같이 보고하면 심사 방어력 크게 상승.**
- **Deflated Sharpe Ratio (DSR)**: 여러 번 시도(하이퍼파라미터 탐색, 시드 변경)로 인한 selection bias 보정. **PPO처럼 시드/하이퍼파라미터 민감도가 큰 모델은 DSR이 필수에 가까움.**

**보상함수 설계 권장 흐름**:

1. 1차 시도 — log return 기반 단순 reward (베이스라인)
2. 2차 — Differential Sharpe Ratio (DSR이 아니라 differential Sharpe; 매 스텝 Sharpe의 증분)
3. 평가는 PSR/DSR로 보고 → 보상이 학습에 미친 영향과 보고 지표를 분리

### 14.5 Runs — **드로다운/연속손실 보상화**

연속된 양수/음수 수익 구간 분석, **drawdown / time under water**, HHI 형태의 수익 집중도. RL 보상에서 가장 자주 무시되지만 가장 중요한 부분:

- 그리드 봇은 횡보장에서는 잘 동작하다가 추세장에서 큰 drawdown을 만드는 게 전형적 실패 패턴. 보상에 drawdown penalty를 명시적으로 넣지 않으면 PPO는 "잘 동작하는 구간"의 누적 보상에 끌려 위험을 과소평가함.
- 구체적 응용: `reward = log_return - λ * current_drawdown` 또는 `reward = log_return - λ * max(0, drawdown - threshold)`.
- **HHI 기반 수익 집중도 페널티**도 가능 — "수익이 몇 개 거래에 몰려 있나"를 직접 보상에 반영.

### 14.6 Implementation Shortfall — **그리드 봇에 치명적으로 중요**

거래 비용, 슬리피지, 시장 충격, brokerage fee, 결제 지연. **그리드 봇은 거래 빈도가 높아 이 항목이 전략의 생존을 결정.** 보상함수에 비용 모델이 빠지면 시뮬레이션에서는 흑자, 실거래에서는 적자가 거의 확정.

구체적 권장:

- 보상에서 매 거래마다 fee + slippage를 차감 (Binance 기준 maker 0.02% / taker 0.04% + 슬리피지 모델).
- `aggressiveness` 액션을 시장가/지정가 비율로 해석한다면, taker 비율에 비례한 비용을 명시적으로 보상에 반영.
- 이걸 안 하면 PPO는 `aggressiveness`를 최대로 밀어붙이는 정책으로 수렴하기 쉬움 — 보상 누수의 전형.

### 14.7 Efficiency — **보상 정규화/연환산**

연환산 Sharpe, 정보 비율(IR), 수익/회전율 비율 등. 직접적인 reward 항은 아니지만 **평가 시 다른 baseline(buy-and-hold, naive grid)과 비교할 때 시간축 보정 기준**을 제공. 논문 결과표에 필수.

### 14.8 Classification Scores

정확도, AUC, F1 등. **지도학습 평가용. RL에는 직접 무관.** 단, 행동을 사후적으로 "맞았다/틀렸다"로 라벨링해 분석할 때만 참고. 후순위.

### 14.9 Attribution

수익을 요인(factor)별로 분해 — 시장 베타, 모멘텀, 변동성 등. **단일자산 BTC 봇에서는 적용 범위가 제한적이지만**, "이 봇의 수익이 buy-and-hold 대비 어디서 왔는가"를 분해하는 분석은 졸업 논문 토론 섹션에 강력함. `divergence`와 `volatility` 상태변수가 실제로 수익에 기여했는지 사후 분석할 때 8장 Feature Importance와 함께 사용.

## 졸업 프로젝트 관점 우선순위 정리

|절|보상함수 직접성|평가 지표로서의 가치|우선순위|
|---|---|---|---|
|14.3 General|낮음|높음 (논문 표)|읽기만|
|**14.4 Performance**|**매우 높음**|**매우 높음**|**1순위**|
|**14.5 Runs**|**매우 높음**|**매우 높음**|**1순위**|
|**14.6 Implementation Shortfall**|**매우 높음**|높음|**1순위**|
|14.7 Efficiency|낮음|높음|2순위|
|14.8 Classification|거의 없음|낮음|스킵 가능|
|14.9 Attribution|낮음|중간|2순위|

## 보상함수 설계에 바로 적용할 체크리스트

이 챕터를 읽고 나면 다음 5가지가 명확해져야 함:

1. **베이스 reward signal은 무엇인가?** (log return? differential Sharpe? PnL?)
2. **거래 비용은 어떤 모델로 매 스텝 차감하는가?** (14.6) — 안 넣으면 결과 무의미
3. **drawdown penalty는 어떻게 들어가는가?** (14.5) — 안 넣으면 그리드 봇 특성상 폭망
4. **최종 평가 지표로 무엇을 보고하는가?** Sharpe 외에 PSR, DSR 포함 (14.4)
5. **baseline과 어떻게 비교하는가?** (14.7) — buy-and-hold, naive grid 대비 efficiency 분석

## 한 가지 주의점

이 챕터는 **지도학습 전략 가정**으로 쓰여서 "백테스트 후 한 번 계산하는 통계"로 제시됨. RL 보상함수는 **매 스텝 또는 매 에피소드마다 미분 가능한 형태**로 변환해야 함. 예를 들어 Sharpe는 그대로 못 쓰고 Differential Sharpe Ratio로 바꿔야 step reward가 됨. López de Prado는 "무엇을 측정할지"를 주고, RL 측에서 "어떻게 미분 가능하게 만들지"를 별도로 풀어야 하는 분업 구조.

이 부분 보강용으로 Moody & Saffell의 "Learning to Trade via Direct Reinforcement" (2001) 한 편을 같이 보면 14.4의 Sharpe → differential Sharpe로 어떻게 RL 보상이 되는지의 다리가 채워짐. 졸업 논문에서 보상함수 절을 쓸 때 이 두 출처(López de Prado 14장 + Moody & Saffell)를 함께 인용하는 게 표준 구도.