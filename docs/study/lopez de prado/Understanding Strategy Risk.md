# Chapter 15: Understanding Strategy Risk — 세부 구성

먼저 페이지 정보 정정: 앞서 ~14페이지로 추정했는데, 실제로는 p.211~220 (약 9페이지)로 더 짧음. 본문은 사실상 6~7페이지 정도. 짧지만 **밀도와 졸업 프로젝트 직접성은 14장보다 더 높음.**

## 구성

15.1 Motivation (p.211), 15.2 Symmetric Payouts (p.211), 15.3 Asymmetric Payouts (p.213), 15.4 The Probability of Strategy Failure (p.216) — 15.4.1 Algorithm, 15.4.2 Implementation.

## 챕터의 핵심 프레임

14장이 "결과를 어떻게 측정할 것인가"라면, **15장은 "전략이 실패할 확률을 모델로 계산하는 법"**. 베팅 기반 전략을 4개 파라미터로 추상화함:

- **p**: 승률(precision)
- **π+**: 이익실현 크기 (profit target)
- **π−**: 손실실현 크기 (stop loss)
- **n**: 단위 시간당 베팅 횟수 (frequency)

이 4개 파라미터로 Sharpe ratio가 **closed-form**으로 유도됨. 그리고 이 식을 뒤집어서 "주어진 (π+, π−, n)에서 목표 Sharpe를 달성하려면 p가 얼마여야 하는가?"를 계산. 이게 챕터의 골격.

**졸업 프로젝트에 왜 결정적인가**: 사용자의 행동 공간 `(aggressiveness, profit_target)` 중 `profit_target`이 **이 챕터의 π+에 그대로 매핑됨.** López de Prado가 제공하는 수식이 PPO의 행동을 평가하는 분석적 기준이 된다는 뜻.

## 절별 내용과 보상함수 응용

### 15.2 Symmetric Payouts — 베이스라인 분석

π+ = −π− 인 경우 (이익/손실 크기 동일). 이항분포 기반 베팅에서:

$$\text{SR} = \frac{(2p-1)\sqrt{n}}{2\sqrt{p(1-p)}}$$

이걸 뒤집으면: 목표 Sharpe를 달성하기 위해 필요한 최소 승률 p가 나옴.

**그리드 봇 응용**: 그리드 트레이딩은 본질적으로 대칭이 아니지만, 베이스라인으로 "각 그리드 셀에서 승/패가 같은 크기일 때 필요한 승률"을 계산해 두면 **현재 봇이 그 임계값을 넘는지 여부**가 진단 지표가 됨. 보상함수에 직접 안 들어가도, 학습 모니터링 대시보드에 띄울 핵심 숫자.

### 15.3 Asymmetric Payouts — **그리드 봇의 본질**

π+ ≠ −π− 인 경우. **그리드 트레이딩이 정확히 이 구조.** 작은 익절(π+)을 자주 가져가지만, 추세장에서는 인벤토리가 쌓여 큰 손실(π−)이 발생. López de Prado는 이걸 **2-mixture Gaussian의 moment matching**으로 풀어서 Sharpe 공식 유도:

$$\text{SR} = \frac{(\pi_+ - \pi_-)p + \pi_-}{\sqrt{(\pi_+ - \pi_-)^2 p(1-p)}} \sqrt{n}$$

이 공식이 보상함수 설계에 주는 직접적 활용:

**1) Implied Sharpe를 step-wise reward로 사용** 매 에피소드(또는 윈도우) 끝에서 (p, π+, π−, n)을 empirical하게 추정 → Implied Sharpe 계산 → 보상.

```python
# 의사코드
window_trades = collect_trades(env, window=100)
p = win_count / len(window_trades)
pi_plus = mean_profit
pi_minus = mean_loss  # negative
n = len(window_trades) / window_length
implied_sharpe = ((pi_plus - pi_minus) * p + pi_minus) / \
                 (sqrt((pi_plus - pi_minus)**2 * p * (1-p))) * sqrt(n)
reward = implied_sharpe - implied_sharpe_prev  # differential form
```

이 방식의 장점: PnL reward의 노이즈를 4개 모멘트로 응축. PPO 학습 안정성에 큰 영향.

**2) Action constraint로 사용** PPO가 `profit_target` (π+)을 결정한 직후, 현재 추정된 p로 위 공식을 돌려 "이 π+에서 양의 Sharpe가 가능한가?"를 즉시 평가 가능. 불가능한 영역으로 가는 행동에 페널티.

**3) Reward shaping의 이론적 근거** 졸업 논문에서 "왜 이 보상함수를 썼는가"를 정당화할 때 López de Prado의 비대칭 페이아웃 공식을 인용하면 **수식적 근거**가 확보됨. RL 논문은 보상 설계의 정당화가 약점이 되기 쉬운데, 이게 메꿔줌.

### 15.4 The Probability of Strategy Failure (PSF) — **리스크 페널티의 정량화**

목표 Sharpe(예: 무위험 수익률, 또는 buy-and-hold)를 정해 두고, **실제 전략이 그 목표를 못 넘길 확률**을 부트스트랩으로 계산.

알고리즘 (15.4.1) 개요:

1. 거래 시퀀스에서 (π+, π−, p)를 추정
2. 두 가우시안 분포(승/패 분포)를 추정
3. 이 분포에서 Monte Carlo 시뮬레이션으로 Sharpe 분포 생성
4. 목표 Sharpe 미달 확률 = PSF

**RL 보상함수에 직접 응용**:

- **Constraint-based reward**: `reward = return - λ * PSF_estimate`. PSF가 높을수록(실패 확률 높을수록) 페널티.
- **Curriculum learning**: 학습 초기엔 PSF 페널티 가중치 λ를 낮게, 후반엔 높게 — Sharpe 추구에서 안정성 추구로 자연스러운 전이.
- **평가 지표**: 14장의 PSR/DSR과는 별도로, PSF를 보고하면 "이 그리드 봇이 1년 동안 buy-and-hold를 못 이길 확률이 X%"라고 직접적 진술이 가능 → 심사위원 친화적.

### 15.4.2 Implementation

저자가 Python 구현 코드를 제공. **그대로 가져다 쓸 수 있음.** 졸업 프로젝트 백테스트 모듈에 함수로 박아두면 됨. 14장 PSR 코드와 묶어서 evaluation utility로 정리하는 게 깔끔.

## 14장 vs 15장 — 보상함수 설계에서의 역할 분담

|측면|14장|15장|
|---|---|---|
|관점|사후 측정 (descriptive)|모델 기반 예측 (prescriptive)|
|보상 입력|log return, Sharpe, drawdown 등 raw signal|(p, π+, π−, n)을 합성한 Implied Sharpe / PSF|
|행동공간 매핑|간접적|**직접적** (profit_target ↔ π+)|
|RL 적합도|중간 (대부분 사후 통계)|**높음** (수식이 미분 가능, 부트스트랩으로 step-wise 가능)|
|코드 분량|보통|짧고 자기완결적|
|페이지|~16p|~9p|

**투자 대비 효율**: 9페이지로 졸업 프로젝트 보상함수 핵심 절반을 짤 수 있는 챕터.

## 권장 학습/적용 순서

1. **15.2 먼저 읽기 (1시간)**: Symmetric 케이스로 직관 확보. 이항분포-Sharpe 관계 손으로 유도해보기.
2. **15.3 깊게 읽기 (2~3시간)**: Asymmetric 공식 유도 따라가기. **여기서 π+를 PPO 행동 `profit_target`으로 치환하는 다이어그램을 노트로 그려둘 것.** 졸업 논문에 그대로 들어갈 그림.
3. **15.4 구현 (반나절)**: 저자 코드를 그리드 봇 환경에 이식. PSF 계산 함수를 evaluation 파이프라인에 추가.
4. **보상함수 v2 시도**: 기존 log return reward에 Implied Sharpe 또는 −PSF 항을 추가해서 비교 실험.

## 한 가지 주의점

15장의 프레임은 **이산 베팅(discrete bets)** 가정. 각 거래가 명확한 진입/청산을 가져야 (p, π+, π−)가 정의됨. 그리드 봇은 이 가정에 잘 맞음 — 그리드 셀 매수→매도가 자연스러운 거래 단위. 단, **보유 중인 미실현 손익(unrealized PnL)을 어떻게 처리할지**가 구현 디테일. 보통은 "거래 종료된 것만 카운트"하지만, 그리드 봇은 long horizon에서 인벤토리가 누적되므로 윈도우 끝에서 강제 청산해 (π+, π−)를 측정하는 방식이 현실적. 이 부분은 책에 명시 안 되어 있어서 **사용자가 졸업 논문에 명시적으로 정의해야 할 methodological choice**.

요약: 15장은 짧지만 사용자의 PPO 행동 공간과 가장 직접적으로 연결되는 챕터. 14장이 "측정 도구함"이라면 15장은 **"행동을 평가하는 수식적 렌즈"**. 보상함수 v1을 만들 때는 14장만으로 충분하지만, v2/v3로 가면서 reward shaping을 정교화할 때 15장이 결정적으로 들어옴.