# Prospect Theory — Kahneman & Tversky (1979)

> Kahneman, D., & Tversky, A. (1979). _Prospect Theory: An Analysis of Decision under Risk._ Econometrica, 47(2), 263–291.
> 후속: Tversky & Kahneman (1992). _Advances in Prospect Theory: Cumulative Representation of Uncertainty._
> RL 응용: KTO (Kahneman-Tversky Optimization, 2024).

## 요지

1. **인간은 기댓값 극대화로 의사결정하지 않는다.** 손실은 같은 크기의 이득보다 약 2배 강하게 느껴진다 (Loss Aversion).
2. Value function v(x)의 특징: **이득은 concave (위험 회피)**, **손실은 convex (위험 추구)**, **손실 측이 더 가파름**.
3. exp027_rl의 asymmetric reward (beta=2.0)의 행동경제학적 근거.

## Prospect Theory의 핵심 발견

### Asymmetric value function

```
v(x) = {  x^α        if x ≥ 0     (concave: 위험 회피)
       { -λ(-x)^β    if x < 0     (convex: 위험 추구)

추정값:
  α = β ≈ 0.88
  λ ≈ 2.25       (loss aversion coefficient)
```

→ "1만원 잃는 고통 ≈ 2만 2천원 얻는 기쁨"
→ 거의 모든 인간 의사결정에 일관되게 나타남 (Ruggeri 2020, 19개국 4,000+명 검증)

### Probability weighting

```
w(p): 객관적 확률 p → 주관적 확률 가중치

특징:
- 작은 확률 과대평가 (복권 효과)
- 큰 확률 과소평가 (보험 효과)
- w(0) = 0, w(1) = 1
- 0 < p < 1에서 w(p) < p 또는 w(p) > p
```

### Reference point dependence

가치 평가는 절대량이 아닌 **기준점(reference point) 대비** 변화로 이루어진다.
→ 같은 결과도 "내가 100을 갖고 있다가 50이 됐다" vs "내가 0에서 50이 됐다"에서 평가 달라짐.

## 트레이딩에서의 prospect theory

### 처분효과 (Disposition Effect)
- 이익 종목은 빨리 팔고, 손실 종목은 오래 들고 가는 경향
- Prospect theory가 정확히 예측: 이익 영역(concave) → 위험 회피 → 빨리 실현
- 손실 영역(convex) → 위험 추구 → 더 들고 가서 만회 시도

### Equity Premium Puzzle
- 주식이 채권보다 너무 높은 초과 수익률 (~7%)
- 합리적 기대효용 이론으로는 설명 불가
- Benartzi & Thaler (1995): "Myopic Loss Aversion"으로 설명 — 사람들은 자주 평가하면 손실에 과민

### Momentum 효과
- 행동재무학파 설명 ([[두 학파 상세]] 참조)
- Underreaction → 추세 형성 → 뒤늦은 추격
- Prospect theory의 reference point dependence와 결합

## RL에서의 활용 — Asymmetric Reward 정식화

### 우리 exp027_rl의 reward
```python
# 단순 asymmetric reward
r_t = equity_change_t / start_capital
if r_t < 0:
    r_t *= beta  # beta = 2.0
```

→ **이게 prospect theory의 RL 버전.**
→ λ ≈ 2.0 ≈ 2.25 (실증치) 거의 일치.

### 더 정식화된 형태 (Prospect-Theoretic Reward)
```python
def prospect_reward(equity_change, ref_point=0, α=0.88, λ=2.25):
    delta = (equity_change - ref_point) / start_capital
    if delta >= 0:
        return delta ** α
    else:
        return -λ * (-delta) ** α
```

→ **Concave gain + convex loss + asymmetric scaling 모두 반영**.
→ exp027_rl 단순 asymmetric보다 더 풍부.

### KTO (Kahneman-Tversky Optimization, 2024)
- LLM alignment에 prospect theory 직접 적용
- "Human-Aware Loss Functions (HALOs)" 프레임워크
- RLHF reward를 prospect-theoretic value function으로 대체
- → LLM이 인간 선호에 더 가깝게 정렬 (DPO 같은 다른 방법보다 우수)

→ **트레이딩 분야에 동일 발상 적용 가능**: reward를 prospect-theoretic으로 만들면 RL 정책이 더 "trader-like"하게 행동.

## 우리 프로젝트와의 연결점

1. **exp027_rl의 학술적 디펜스**:
   - 단순 "beta=2.0 hyperparameter"가 아니라 "prospect theory의 loss aversion coefficient λ를 도입"
   - λ = 2.25가 Kahneman-Tversky 실증치 → 우리 beta=2.0과 거의 일치 → 자연스러운 선택 정당화
   - 그냥 임의로 정한 게 아님

2. **Reference point 선택**:
   - 현재 our reward: equity change (reference point = 직전 equity)
   - 대안 1: cycle entry 가격을 reference (사이클 단위)
   - 대안 2: rolling EMA equity를 reference (장기 추세 대비)
   - → 다른 reference로 정책이 다르게 학습될 수 있음

3. **Probability weighting의 RL 응용**:
   - 우리 시스템은 확률 명시적 사용 안 함
   - 그러나 distributional RL ([[distributional_rl]])과 결합하면:
     - 손실 분포의 꼬리에 더 가중 (CVaR 효과)
     - "복권 효과" 회피 — 작은 확률 대박 추구하는 정책 억제

4. **Prospect-theoretic reward 시도 (exp030 후보)**:
   ```yaml
   reward_design:
     type: prospect_theoretic
     alpha: 0.88          # power exponent
     lambda: 2.25         # loss aversion
     reference_point: previous_equity
   ```

5. **위험**:
   - 이론적 매력 ≠ 실증적 우수성. 실제로 더 나은지는 실험 필요.
   - Power exponent (0.88)이 reward를 sub-linear로 만듦 → 큰 reward의 gradient signal 감쇠
   - → DSR + asymmetric 의 단순 결합이 더 안전할 수도

## 백링크

- [[differential_sharpe_moody2001]] — Reward design의 학술적 출발점
- [[reward_shaping_ng1999]] — Potential-based shaping과의 정합성 검토 필요
- [[distributional_rl]] — Probability weighting과 자연 결합
- [[risk-adjusted return]] — Sortino가 prospect theory의 정량 버전

## 출처

- [Kahneman & Tversky (1979) Original Paper (MIT)](https://web.mit.edu/curhan/www/docs/Articles/15341_Readings/Behavioral_Decision_Theory/Kahneman_Tversky_1979_Prospect_theory.pdf)
- [Prospect Theory — Wikipedia](https://en.wikipedia.org/wiki/Prospect_theory)
- [Loss Aversion Bias (Simply Psychology)](https://www.simplypsychology.org/prospect-theory.html)
- [KTO RLHF Alternative (Argilla)](https://argilla.io/blog/mantisnlp-rlhf-part-7/)
- [Global Study on Loss Aversion (Columbia)](https://www.publichealth.columbia.edu/news/global-study-confirms-influential-theory-behind-loss-aversion)
