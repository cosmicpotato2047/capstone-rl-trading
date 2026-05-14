Chapter 6 "Money and Risk Management"은 **약 24페이지(p.109-132, 2판 기준)** 분량으로, 졸업 프로젝트의 보상함수 설계와 가장 직결되는 챕터입니다. 2판에서 Kelly formula 부분이 대폭 보강되고 Python 코드가 추가됐습니다.

## 챕터 구조 (2판 기준)

| 섹션                                           | 페이지       | 내용                    | 중요도          |
| -------------------------------------------- | --------- | --------------------- | ------------ |
| §6.1 Optimal Capital Allocation and Leverage | p.109-119 | Kelly formula의 수학과 응용 | ★★★          |
| §6.2 Risk Management                         | p.120-125 | 포지션 사이징, drawdown 통제  | ★★★          |
| ├ Model Risk                                 | p.124     | 모델 가정 위반              | ★            |
| ├ Software Risk                              | p.125     | 코드 버그                 | ★            |
| ├ Natural Disaster Risk                      | p.125     | (졸업 프로젝트 무관)          | —            |
| §6.3 Psychological Preparedness              | p.125-129 | 손실 회피, 규율             | ★★ (2판 새 내용) |
| §6.4 Summary                                 | p.130     |                       | —            |
| §6.5 Appendix: Kelly Derivation (Gaussian)   | p.131-132 | Kelly 수식 유도           | ★★           |

---

## §6.1 Optimal Capital Allocation and Leverage — Kelly Formula의 모든 것

### Kelly Formula 핵심 수식

**다전략 일반형** (Chan이 강조하는 형태): $$F^* = C^{-1} M$$

- $F^*$: 각 전략에 할당할 레버리지 벡터
- $C$: 전략 수익률의 공분산 행렬
- $M$: 평균 수익률 벡터

**독립 전략 단순화** (단일 BTC 자산에 적용 가능): $$f^* = \frac{m}{s^2}$$

- $m$: 한 기간 평균 수익률 (excess return)
- $s^2$: 한 기간 수익률 분산
- 또는 동등하게: $f^* = \frac{S}{s}$ ($S$ = Sharpe ratio)

**복리 성장률 (Chan의 핵심 통찰)**: $$g = r + \frac{S^2}{2}$$

- $g$: 장기 복리 성장률
- $r$: 무위험 수익률
- $S$: Sharpe ratio
- → **Sharpe ratio 최대화 = 장기 부 최대화 (Kelly 레버리지 사용 시)**. 이게 RL 보상함수로 Sharpe 기반을 정당화하는 가장 강력한 논거입니다.

### Half-Kelly 권장 — Chan의 명시적 입장

Chan은 본인 블로그에서 "log-normal 모델은 실제 위험을 과소평가하므로 대부분의 실무자는 안전 마진을 위해 half-Kelly 공식을 사용한다"고 답변합니다.

**왜 Half-Kelly인가**:

1. Gaussian 가정은 BTC 같은 fat-tail 자산에서 큰 위험 과소평가
2. Full-Kelly에서 50% 드로우다운 확률이 매우 높음
3. Half-Kelly의 복리 성장률은 $r + \frac{3S^2}{8}$ (full의 75% 수준이지만 drawdown 위험 급감)

**프로젝트 적용 — `aggressiveness` 행동 설계의 이론적 디펜스**:

현재 `aggressiveness ∈ [0, 1]` 연속 행동을 그리드 폭에 매핑하고 있는데, **Kelly framework로 재해석하면**:

|aggressiveness 값|해석 (Kelly framework)|
|---|---|
|0.0|0 × Kelly = 포지션 없음|
|0.5|half-Kelly (Chan 권장 안전 상한)|
|1.0|full-Kelly (이론적 성장 최대, 실전 위험)|

**디펜스에서의 정당화**: "왜 action을 `[0, 1]`로 bound했는가?"라는 질문에 → "0이 무포지션, 1이 full-Kelly에 대응. RL agent가 이 범위 내에서 최적 fraction을 학습하도록 설계함. Chan(2021) §6.1의 half-Kelly 권장에 따르면 학습된 정책이 평균 0.5 부근에 수렴해야 건강한 신호."

이건 단순히 `[0, 1]`로 잘랐다는 것보다 **훨씬 강력한 디펜스 라인**입니다.

### BTC 단일 자산에 Kelly 적용 — 구체 수치

가상의 예시로 직관 잡기:

- BTC 1시간 수익률 평균 $m \approx 0.0001$ (연 약 ±100%, 1시간당)
- BTC 1시간 수익률 표준편차 $s \approx 0.01$ (연 변동성 ~85%)
- $f^* = 0.0001 / 0.01^2 = 1.0$ → 100% 자기자본 (레버리지 없음)
- Half-Kelly = 0.5 → **자본의 50%를 BTC에 holding하는 것이 안전 상한**

이 수치가 `holdings_ratio` 상태변수의 자연스러운 reference point가 됩니다.

---

## holdings_ratio / cash_ratio 상태변수의 의미 부여

이 두 상태변수는 **사실상 Kelly fraction의 직접 관측값**입니다:

$$\text{holdings_ratio} = \frac{\text{position value}}{\text{total equity}} = f_{\text{current}}$$ $$\text{cash_ratio} = 1 - \text{holdings_ratio}$$

**왜 이게 강력한 상태변수인가**:

1. **Kelly 이론의 직접 표현**: agent가 학습할 정책의 핵심 변수가 곧 $f_{\text{current}}$. agent는 "지금 $f$가 너무 높은가/낮은가"를 인지 가능.
2. **자기참조성**: 단순 가격 변수만 있으면 agent는 "내가 지금 얼마 들고 있는지" 모름. holdings_ratio가 있어야 거래 행동이 일관됨.
3. **Drawdown과 연결**: equity가 줄면 holdings_ratio도 변함 → agent가 drawdown을 간접 인지.

**유의점**: 두 변수는 서로 완전 종속(합 = 1). PPO에서 둘 다 입력하면 redundant. 하나만 사용하거나 각각 다른 정규화(예: holdings_ratio raw + cash_ratio log scale)를 쓰는 게 효율적입니다.

---

## §6.2 Risk Management — Drawdown 통제와 보상함수 설계

### Chan의 핵심 메시지

Risk management의 본질은 **"파산 회피 (avoid ruin)"**. Kelly는 평균 성장 최대화지만, 그 과정에서 drawdown이 너무 깊으면 심리·자금 조달 면에서 실전 사용 불가.

### Position Sizing by Maximum Drawdown Tolerance

Chan이 제시하는 실용 공식: $$f_{\text{used}} = f_{\text{Kelly}} \times \frac{\text{drawdown tolerance}}{\text{historical max drawdown at full Kelly}}$$

예: full Kelly에서 historical max DD가 80%인데, 본인이 30%까지만 견딜 수 있다면 → f_used = 0.375 × Kelly (약 1/3 Kelly).

**RL 프로젝트 적용 — 보상함수 후보 3종**:

#### 후보 A: Drawdown 페널티 (가장 단순)

$$r_t = \Delta\text{equity}_t - \lambda \cdot \max(0, \text{current_drawdown}_t - \text{threshold})$$

- threshold = 5~10% (작게 시작)
- λ는 hyperparameter — ablation 필요 (§Ch.2에서 다룬 data-snooping 주의)

#### 후보 B: Differential Sharpe Ratio (Moody & Saffell 1998, Chan 정신과 정렬)

$$r_t = \frac{B_{t-1}\Delta A_t - \frac{1}{2}A_{t-1}\Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}$$

- $A_t, B_t$: 수익률의 1차/2차 EMA
- **장점**: Chan의 "Sharpe = 장기 성장 최적" 통찰을 매 step 보상으로 분해
- **단점**: 구현 복잡, 학습 초반 noisy

#### 후보 C: Risk-Adjusted Return (실용적 절충)

$$r_t = \Delta\text{equity}_t - \lambda_1 \cdot \text{drawdown}_t^2 - \lambda_2 \cdot |\text{trade_size}_t| \cdot \text{cost}$$

- drawdown 제곱 → 깊은 DD에 더 큰 페널티 (convex)
- 거래비용도 직접 차감 (§Ch.3 §3.5와 정렬)

**권장 접근**: A로 시작 → 학습 안정 확인 → C로 발전. B는 논문 차원에서 시도. 세 가지 모두 비교 ablation을 디펜스 자료로.

### Model Risk (p.124)

**Chan**: "당신이 Gaussian을 가정하고 Kelly를 썼는데 실제 분포가 fat-tail이면 Kelly는 거짓말을 한다."

- BTC는 명백히 fat-tail (jump, regime change) → Kelly 추정치는 항상 보수적으로 사용
- RL 프로젝트 디펜스에서 "왜 action을 [0, 1]로 한정?"의 추가 답변: "model risk 회피"

### Software Risk (p.125)

**Chan의 강한 경고**: 거래 시스템 버그가 Kelly 가정을 모두 무력화한다. 특히 RL 프로젝트에서:

- 정규화 코드 버그 → state distribution shift
- 보상 계산 부호 오류 → agent가 손실 추구
- env step 시점 정렬 오류 → look-ahead로 인공 알파

---

## §6.3 Psychological Preparedness — 2판의 새 통찰

2판에서 Chan이 입장을 바꾼 부분: **"loss aversion은 행동편향이 아니라 수학적 필연이다"**.

### 핵심 통찰

복리 환경에서:

- 50% 손실 → 회복하려면 100% 수익 필요 (비대칭)
- 따라서 **합리적 agent는 손실을 이익보다 더 무겁게 평가해야** 한다 (Kelly 도출의 자연스러운 귀결)

### RL 프로젝트 적용

이 통찰을 보상함수에 반영하는 두 가지 방법:

1. **로그 수익률 사용**: $$r_t = \log\left(\frac{\text{equity}_t}{\text{equity}_{t-1}}\right)$$

- 손실의 "복구 어려움"을 자연스럽게 인코딩 (-50% = log(0.5) ≈ -0.69, +50% = log(1.5) ≈ +0.41 → 비대칭이 수식에 내재)
- **PPO BTC grid bot에 강력 권장**. 단순 pnl보다 이론적으로 깔끔.

2. **CVaR 페널티** (선택적 고급): $$r_t = \Delta\text{equity}_t - \lambda \cdot \text{CVaR}_{\alpha}(\text{recent returns})$$

- 꼬리 위험 직접 페널티
- 구현 부담 큼, ablation 항목으로만 고려

---

## §6.5 Appendix: Kelly의 Gaussian 유도 (p.131-132)

수학적 디펜스가 필요한 학생에게 **귀중한 자원**입니다. 4페이지 분량으로:

- 기대 log return을 leverage f의 함수로 전개
- 미분해서 최적점 찾는 표준 유도
- Gaussian 가정 명시

졸업 논문에 Kelly 도출 한 페이지 넣을 거면 이 부록을 그대로 참고하면 됩니다 (재인용은 적절히).

---

## 졸업 프로젝트용 Chapter 6 요약 매핑표

|Chan §6 개념|졸업 프로젝트 요소|디펜스 라인|
|---|---|---|
|Kelly $f^* = m/s^2$|`aggressiveness` 행동 ∈ [0, 1]|"0=무포지션, 1=full-Kelly에 대응"|
|Half-Kelly 권장|aggressiveness 평균 ≈ 0.5 기대|"학습된 정책이 안전영역에 수렴 확인"|
|$g = r + S^2/2$|Sharpe-based reward 사용|"Sharpe 최대화 = 장기 성장 최대"|
|holdings_ratio|상태변수|"현재 Kelly fraction의 직접 관측"|
|Drawdown 통제|reward에 DD 페널티|"model risk 대응한 보수적 사이징"|
|Loss aversion (수학적)|log return reward|"복리 비대칭의 자연스러운 인코딩"|
|Model Risk|action 범위 [0, 1] 한정|"Gaussian 가정 위반 대비"|
|Software Risk|단위 테스트, 시드 다중|"Ch.3 체크리스트와 연계"|

---

## 학습 우선순위 제안

이 챕터 24페이지 중 **15-16페이지가 프로젝트 핵심**입니다. 시간이 부족하면:

1. **§6.1 (필독, 11페이지)** — Kelly 수식과 half-Kelly 권장
2. **§6.5 부록 (필독, 2페이지)** — Kelly 유도 (논문에 인용 가능)
3. **§6.2 처음 5페이지** — drawdown 기반 position sizing
4. **§6.3 loss aversion 부분** — 2판 신규 내용, log return reward 정당화
5. (시간 여유) §6.2 Model/Software Risk

**Chapter 6의 한 줄 요약**: "Kelly는 'aggressiveness'의 수학적 정의를 주고, drawdown 통제는 보상함수의 패널티 구조를 주며, log-return-as-reward는 손실 비대칭의 자연스러운 표현이다." 이 세 줄이 졸업 프로젝트 보상함수 설계 챕터의 이론적 토대 전부가 될 수 있습니다.