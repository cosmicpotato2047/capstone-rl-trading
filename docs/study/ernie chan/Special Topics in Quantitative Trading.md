Chapter 7은 **약 50페이지(p.133-182, 1판 / p.133-192, 2판)** 분량으로, 책에서 가장 기술적인 챕터입니다. 2판에서 §7.2가 "Regime Change and **Conditional Parameter Optimization**"으로 확장된 것이 핵심 변화 — 이게 RL 프로젝트와 가장 직접 연결됩니다.

## 챕터 구조 요약

|섹션|페이지 (2판)|분류|핵심|
|---|---|---|---|
|§7.1 Mean-Reverting vs Momentum|134-137|**A**|그리드 = MR 베팅의 통계적 정당화|
|§7.2 Regime Change & Conditional Param Opt|137-147|**A**|volatility 상태변수, BTC regime|
|§7.3 Stationarity and Cointegration|147-160|**B** (선별 A)|log_price 정당화, half-life|
|§7.4 Factor Models|160-169|C|멀티에셋|
|§7.5 What Is Your Exit Strategy?|169-174|**A**|profit_target 설계|
|§7.6 Seasonal Trading|174-186|C|농산물·에너지|
|§7.7 High-Frequency Trading|186-188|C|tick 데이터 필요|
|§7.8 High-Leverage vs High-Beta|188-190|C|포트폴리오|

A 분류 4개 섹션부터 깊게 들어갑니다.

---

## §7.1 Mean-Reverting vs Momentum (p.134-137) — 그리드의 통계적 디펜스

### 책의 핵심 주장

**같은 가격 시계열도 시간 스케일에 따라 mean-reverting일 수도, momentum일 수도 있다.** 이게 챕터 7 전체를 관통하는 통찰입니다.

Chan이 제시하는 **3가지 통계적 진단 도구**:

#### ① Augmented Dickey-Fuller (ADF) Test

- 귀무가설: "이 시계열은 unit root를 갖는다 (=non-stationary, random walk)"
- p-value < 0.05 → MR 성향 인정
- **BTC log_price 1시간봉**에 ADF 돌리면 보통 non-stationary로 나옴 (강한 추세) → **그래서 단순 가격이 아닌 변형(divergence)이 필요**

#### ② Hurst Exponent (H)

- H = 0.5 → random walk
- H < 0.5 → mean-reverting
- H > 0.5 → trending
- BTC는 시간 스케일에 따라 H가 달라짐: 분/시 단위는 0.45~0.5 (약한 MR), 일/주 단위는 0.55~0.6 (모멘텀)
- **이게 1시간봉 그리드 봇이 작동하는 통계적 근거**: 단기 MR 성향 활용

#### ③ Variance Ratio Test

- N기간 분산 / (N × 1기간 분산) ≈ 1이면 random walk
- < 1이면 MR, > 1이면 momentum

### `divergence` 상태변수의 정당화

`divergence = (current_price - moving_avg) / moving_avg` 형태일 텐데, 이건 **본질적으로 mean-reversion signal**입니다:

- divergence > 0 → 평균 위, MR 가정 시 매도 압력 예상
- divergence < 0 → 평균 아래, MR 가정 시 매수 압력 예상

**디펜스 라인**: "왜 divergence를 상태변수로 썼는가?" → "Chan §7.1에서 제시하는 3가지 MR 진단(ADF, Hurst, Variance Ratio)이 BTC 1시간봉에서 약한 MR 성향을 보이며, 그리드 트레이딩이 MR 베팅의 한 형태라는 점에서 divergence는 MR signal의 직접적 표현."

### 한 가지 위험 신호 — 디펜스 대비

Chan이 §7.1 말미에 강조: **"strategy 효과는 시간이 지나며 사라질 수 있다"**. BTC 시장의 효율성이 높아지면서 MR 성향이 약해지는 추세. → **이게 RL의 정당화**: 정적 MR 전략은 작동 안 할 때가 늘지만, RL은 **regime별 적응**으로 alpha 잔존 가능성. §7.2와 자연스럽게 연결.

---

## §7.2 Regime Change and Conditional Parameter Optimization (p.137-147) — **2판 핵심 신규 내용**

### 1판 → 2판 변화

**1판 §7.2 "Regime Switching"**: 단순히 Markov regime switching 모델 소개로 그침.

**2판 §7.2**: 제목이 "**Conditional Parameter Optimization**"으로 확장. **머신러닝 기반 동적 파라미터 조정**이 추가됨. → **이게 졸업 프로젝트 motivation의 직접적 근거**.

### 책의 핵심 통찰 (2판)

전통적 백테스트: 전체 기간에 대해 **하나의 최적 파라미터**를 찾음. 하지만 시장은 regime이 바뀜.

Chan의 제안: **regime-conditional parameter optimization**. 즉, regime을 먼저 감지하고, 각 regime별로 다른 파라미터를 사용. ML/RL 기반 접근의 직접적 근거.

### Regime 정의 — Chan이 제시하는 방법들

#### ① Markov Regime Switching (Hamilton 1989)

- 숨은 상태 (bull/bear/sideways)가 Markov chain
- 각 regime마다 다른 mean·variance
- 관측: returns; 추론: regime probability
- **단점**: 추론 지연. 실시간 사용 어려움.

#### ② Volatility 기반 Regime

- 단순하지만 강력: **현재 변동성 σ_t를 기준으로 regime 분류**
- σ_t < 30th percentile → low-vol regime
- σ_t > 70th percentile → high-vol regime
- 그 사이 → mid-vol
- **BTC에 매우 적합**: BTC는 변동성 자기상관이 높아서 (volatility clustering, GARCH 효과) σ_t로 regime 식별이 잘 됨

#### ③ Trend 기반 Regime

- price_now / price_N_periods_ago 비율
- > 1.05 → bull, < 0.95 → bear, 그 사이 → sideways
    

### `volatility` 상태변수의 의미 부여

현재 상태변수에 `volatility`가 들어가 있는 것의 정당화:

**디펜스 라인**: "왜 volatility를 상태변수로 포함했는가?" → "Chan §7.2의 conditional parameter optimization 관점에서 volatility는 regime의 가장 강력한 단일 지표. PPO agent가 volatility를 입력받으면 regime별로 다른 정책(aggressiveness, profit_target)을 학습할 수 있음. 이게 정적 그리드 대비 RL 그리드가 alpha를 보존하는 핵심 메커니즘."

이 한 문단이 **졸업 프로젝트의 motivation 핵심 진술**이 될 수 있습니다. "왜 RL을 썼는가?"의 학술적 답변.

### BTC Regime Change의 실제 사례

|시기|Regime|그리드 봇 적정 행동|
|---|---|---|
|2017 말|Bull mania|aggressiveness↓, profit_target↑ (추세 따라가기)|
|2018 약세장|Long bear|aggressiveness↓, 또는 거래 중단|
|2019-2020 횡보|Range-bound|aggressiveness↑, 좁은 그리드 (MR 최적)|
|2020-2021 폭등|Strong bull|그리드 손실 (놓치는 상승)|
|2022 약세장|Bear|aggressiveness↑, 넓은 그리드|
|2023-현재|회복기|다양|

**Chan §7.2의 메시지**: 이 6가지 regime을 모두 같은 파라미터로 접근하면 평균은 손실. **regime-conditional 정책 = RL의 자연스러운 표현 영역**.

### Conditional Parameter Optimization과 RL의 동치성

Chan 2판이 제안하는 framework:

1. regime 식별 → 이산 변수 r ∈ {1, ..., K}
2. 각 regime r에 대한 최적 파라미터 θ*(r) 학습
3. 실시간으로 r 추론, θ*(r) 적용

**RL framework로 변환하면**:

1. state s_t (volatility 등 포함)
2. policy π(a | s) — agent가 s로부터 직접 행동 학습
3. 이산 regime을 명시할 필요 없이, **연속 state로부터 연속 action을 직접 학습**

→ **PPO + 5-state는 Chan의 conditional parameter optimization을 더 일반화한 형태**. 이게 학술적 contribution 진술.

---

## §7.3 Stationarity and Cointegration (p.147-160) — log_price 정당화 + Half-life

### Stationarity 부분 (p.147-153) — A 분류

**책의 정의**: 시계열 X_t가 stationary면:

- 평균이 시간 불변
- 분산이 시간 불변
- 자기공분산이 시차에만 의존

**가격 vs log_price**:

- Price: trending non-stationary (BTC 100 → 70000)
- log(Price): 여전히 trending이지만 **수익률 = Δlog(Price)는 stationary에 가까움**
- **수익률 변환이 더 유용**한 경우가 많음

### log_price 상태변수의 정당화

상태변수에 `log_price`가 들어가는 이유:

**Chan 관점**: 가격 자체보다 log 변환이 통계 분석에 적합 (Box-Cox류 변환).

**RL 관점**: 추가 이유들:

1. **Numerical stability**: BTC 가격 range 100~70000 → log 변환하면 4.6~11.2로 압축
2. **Scale invariance**: log return이 가격 수준에 무관 (multiplicative process의 자연 표현)
3. **Reward와 정합성**: §6에서 다룬 log return reward와 동일 시간 스케일

**유의점 — 디펜스에서 받을 수 있는 질문**: "log_price는 여전히 non-stationary인데 PPO가 이걸 입력받아도 되는가?"

**답변**: 가능한 두 가지 접근:

- (a) **Rolling normalization**: `log_price_normalized = (log_price - rolling_mean) / rolling_std`
- (b) **Differencing**: log_price 대신 log_return 사용
- 현재 설계가 (a)라면 §3.4.1 look-ahead bias 점검 필수 (rolling이 미래 보지 않게)

만약 **divergence가 이미 normalized log_price와 같다면**, raw log_price와 divergence는 redundant 가능성. 점검 필요.

### Half-life of Mean Reversion (Ornstein-Uhlenbeck)

Chan §7.3에서 **가장 실용적인 도구**: 평균 회귀 속도를 정량화.

**OU 프로세스**: $$dx_t = -\theta(x_t - \mu)dt + \sigma dW_t$$

**Half-life 추정**: AR(1) 회귀 $\Delta x_t = a + b \cdot x_{t-1} + \epsilon$로 b 추정 후 $$\text{half-life} = -\frac{\ln(2)}{b}$$

**프로젝트 적용 — 그리드 폭 설계의 직접적 근거**:

- BTC 1시간봉 log_price의 half-life를 측정
- 만약 half-life가 12시간이면, 그리드는 **12시간 내에 가격이 그리드 라인을 통과해야 수익**
- `profit_target` 행동이 너무 작으면(예: 1시간 내 도달 불가) 미체결, 너무 크면(half-life의 5배 이상) 수익 기회 상실
- → **profit_target 범위 설계의 통계적 근거**

**디펜스 라인**: "profit_target의 행동 범위는 BTC log_price의 추정 half-life(12h)를 기준으로 0.5~3 half-life에 해당하는 가격 변화율로 설정. 이는 OU 프로세스 가정 하에 평균 회귀가 통계적으로 의미 있는 시간창."

### Cointegration 부분 (p.153-160) — C 분류

페어 트레이딩 전용. 단일 BTC 자산에 무관. **건너뛰기**.

(2판은 이 부분에 Engle-Granger, Johansen test의 Python 코드 추가됨. 시간 여유 있다면 cointegration 자체의 개념 학습 가치는 있지만 졸업 프로젝트와는 무관.)

---

## §7.5 What Is Your Exit Strategy? (p.169-174) — profit_target 설계의 핵심

### 책의 핵심 주장

**"엔트리는 쉽지만 엑시트가 진짜 어렵다."** Chan은 strategy type별로 다른 exit 원칙을 제시.

### Mean-Reversion 전략의 Exit (그리드 봇이 여기 해당)

**Chan의 명시적 권고 3가지**:

#### ① Target Price (가장 일반적)

- 진입 시점에 목표 가격을 명시
- MR 가정 하에 평균까지 회귀하면 청산
- **그리드 봇의 자연스러운 메커니즘**: 매수 후 X% 위에 매도 주문

#### ② Holding Period (시간 기반)

- Half-life의 1~2배 시간 후 강제 청산
- MR이 일어나지 않으면 가정이 틀린 것 → 손절
- **그리드 봇 보강**: 그리드 위에 N시간 이상 머문 포지션은 청산

#### ③ Stop-Loss는 권장하지 않음 (놀라운 입장)

Chan은 자신의 블로그에서 명시적으로 "mean reversion 전략에는 stop loss를 권장하지 않는다. 다만 절대 발동되지 않을 거라고 확신할 때를 제외하고는"이라고 답변합니다.

**이유**: MR 전략은 가격이 평균에서 멀어질수록 진입 신호가 강해짐. Stop-loss는 가격이 멀어질 때 청산 → MR 신호와 반대 방향. 가장 좋은 진입 가격에 청산하게 됨.

### `profit_target` 행동의 설계 정당화

현재 `profit_target ∈ [a, b]` 연속 행동이 그리드 폭(또는 매도 목표가)에 매핑되어 있을 텐데:

**Chan §7.5의 framework로 재정당화**:

|profit_target 값|Chan §7.5 해석|트레이드오프|
|---|---|---|
|작음 (좁은 그리드)|짧은 holding period 기대|거래빈도↑, 비용↑, 체결률↑|
|큼 (넓은 그리드)|긴 holding period 기대|거래빈도↓, 비용↓, 미체결 위험↑|

**디펜스 라인**: "왜 profit_target이 연속 행동인가?" → "Chan §7.5에서 강조하듯 MR 전략의 exit는 target price 또는 holding period 기반이고, 두 변수는 monotonic 관계. 시장 regime(volatility)에 따라 최적 target은 달라짐. 정적 target 대신 RL이 regime-conditional optimal target을 학습하게 함."

### Stop-Loss 부재의 RL 함의

Chan의 "MR 전략에 stop-loss 부적절" 입장은 **RL 보상함수에도 영향**:

- 단순 drawdown 페널티 reward는 agent에게 "손실 시 청산" 행동을 유도 → MR 가정과 충돌
- 대안: drawdown 페널티는 **포지션 크기 축소**(aggressiveness 감소)는 학습시키되, 강제 청산은 학습시키지 않도록 설계
- 보상함수에 "stop-loss 행동에 대한 직접 페널티"는 두지 않는 것이 안전

### Momentum 전략의 Exit (참고용 — BTC bull market 시 그리드 봇의 약점)

Chan은 momentum 전략에는 정반대 권고:

- **Target은 두지 말고 trailing stop 사용**
- 추세를 끝까지 타라

**그리드 봇의 약점이 여기서 드러남**: BTC 강세장(예: 2020년 말~2021년 초)에서는 momentum 전략이 우월. 그리드는 일찍 청산하고 추가 상승 놓침.

→ **졸업 프로젝트의 한계 명시**: "본 RL 그리드 봇은 §7.1의 MR regime에서 alpha를 추구하며, 강한 momentum regime에서는 buy-and-hold 대비 underperform할 수 있다. 이는 §7.2의 regime detection으로 부분 보완하나, 본질적 한계." → 디펜스에서 솔직한 한계 인정 = 신뢰도 상승.

---

## C 분류 섹션 — 짧게만

### §7.4 Factor Models (p.160-169)

Fama-French 3-factor 등. 멀티 자산 cross-sectional 전략용. 단일 BTC 봇과 무관. **건너뛰기**.

### §7.6 Seasonal Trading (p.174-186)

천연가스 겨울 수요, 농산물 수확기 패턴 등. BTC는 명확한 seasonality 없음 (요일/시간대 효과는 미미). **건너뛰기**. (단, "BTC에 seasonality가 있는가?"라는 졸업 디펜스 질문 받을 수 있으니 답변 준비: "통계적으로 유의한 패턴 없음, factor로 포함하지 않음.")

### §7.7 High-Frequency Trading (p.186-188)

틱 데이터, 마이크로구조, 시장조성. 1시간봉 사용 중이라 무관. **건너뛰기**.

### §7.8 High-Leverage vs High-Beta (p.188-190)

포트폴리오 구성 문제. 단일 자산 봇과 무관. **건너뛰기**.

---

## 졸업 프로젝트용 Chapter 7 매핑표

|Chan §7 개념|프로젝트 요소|디펜스 라인|
|---|---|---|
|§7.1 ADF/Hurst/VR test|divergence 상태변수|"BTC 1시간봉의 약한 MR 성향이 그리드 작동 근거"|
|§7.2 Regime detection|volatility 상태변수|"regime-conditional 정책 학습이 RL의 핵심 가치"|
|§7.2 Conditional Param Opt|RL framework 자체|"Chan의 ML 기반 conditional optimization을 PPO로 일반화"|
|§7.3 Stationarity|log_price 변환|"non-stationary 가격의 통계 분석 적합 변환"|
|§7.3 Half-life (OU)|profit_target 범위|"0.5~3 half-life에 해당하는 가격 변화율"|
|§7.5 MR exit (target+time)|profit_target 행동|"target price와 holding period의 dual role"|
|§7.5 Stop-loss 부적절|reward 설계|"drawdown 페널티는 사이즈 축소만 유도, 강제 청산 X"|
|§7.5 Momentum 약점|한계 인정|"강한 momentum regime에서 B&H 대비 약점, regime detection으로 부분 보완"|

---

## 학습 우선순위 — 50p 중 25p가 핵심

시간이 부족하면 이 순서로:

1. **§7.5 MR exit (5p)** — profit_target 설계 직접 근거. 짧고 명확.
2. **§7.1 (4p)** — divergence 정당화. ADF/Hurst 개념만 잡으면 충분.
3. **§7.2 (10p, 핵심)** — volatility 상태변수와 RL motivation. **2판 필수**, 1판은 약함.
4. **§7.3 stationarity 부분 (6p)** — log_price와 half-life. cointegration은 skip.
5. (시간 여유 시) §7.5의 momentum exit 부분 — 한계 인정용

**나머지 25페이지(§7.3 cointegration + §7.4 + §7.6 + §7.7 + §7.8)는 건너뛰어도 됩니다.**

---

## Chapter 7 한 줄 요약

**"Chan §7.1+§7.5는 그리드의 통계적·전술적 정당화이고, §7.2+§7.3은 RL이 정적 그리드를 능가하는 이론적 근거다."**

이전 챕터들과 연결하면:

- **§Ch.2** = 함정 회피 (방어)
- **§Ch.3** = env 정확성 (인프라)
- **§Ch.6** = aggressiveness 의미 (Kelly)
- **§Ch.7** = divergence/volatility/profit_target 의미 (시장 가설)

이 4개 챕터(약 130페이지)가 졸업 논문의 "Background & Theoretical Foundation" 섹션의 거의 전부를 차지할 수 있습니다. RL 알고리즘(PPO) 부분은 Sutton & Barto, Schulman 2017 PPO 논문에서 끌어오면 되니, **Chan은 정확히 "왜 BTC 그리드 트레이딩이 RL의 좋은 응용 영역인가"의 학술적 디펜스를 제공**하는 역할입니다.