Chapter 2 "Fishing for Ideas"는 **약 22페이지(p.9-31)** 분량이고, 두 부분으로 나뉩니다.

## 챕터 구조 개요

**§2.1 How to Identify a Strategy That Suits You** (p.9-19) "어떤 전략이 나에게 맞는가" 자기 진단 — 근무시간(전업/부업), 프로그래밍 실력, 자본금, 목표(수익 vs 절대수익) 기준. **이 부분은 졸업 프로젝트에는 거의 무관합니다.** 실거래 사업자 시점의 가이드.

**§2.2 A Taste for Plausible Strategies and Their Pitfalls** (p.20-30) "전략을 평가할 때 반드시 던져야 할 7가지 질문." — **여기가 본론입니다.** 7개 모두 RL 프로젝트와 직결되니 하나씩 짚어드릴게요.

---

## §2.2 7가지 Pitfall — BTC 그리드 봇 관점에서

### ① Benchmark과의 비교 (p.20)

**책의 주장**: 절대수익만 보지 말고 적절한 benchmark 대비 초과수익을 봐라. 주식 long-only면 SPY, long-short면 T-bill, 일정 변동성 가지면 Sharpe ratio.

**프로젝트 적용**:

- BTC 그리드 봇의 자연스러운 benchmark는 **3개**:
    1. **Buy & Hold BTC** (가장 강력한 비교 대상)
    2. **고정 그리드(non-RL)** — RL이 정말 가치를 추가하는지 증명
    3. **무위험 수익률** (Sharpe 계산용)
- **보상함수 함의**: raw return을 보상으로 쓰면 BTC 상승장에서 그냥 buy & hold가 이김. ==**excess return over buy-and-hold**를 보상으로 설계하는 것이 RL의 알파를 분리해서 학습시키는 방법==. 이건 졸업 디펜스에서 "왜 이 보상함수인가"의 강력한 답변이 됩니다.

### ② Drawdown 깊이와 지속기간 (p.23)

**책의 주장**: Sharpe가 좋아도 max drawdown이 깊거나 회복기간(MDD duration)이 길면 실전 사용 불가. Chan은 **drawdown duration**이 깊이만큼 중요하다고 강조.

**프로젝트 적용**:

- **==보상함수에 drawdown 페널티는 거의 필수==**. 안 넣으면 PPO 에이전트가 학습 도중 극단적 leverage/aggressiveness를 시도하다 손실 후 복구하는 경로를 학습하면서 큰 변동성을 정상화시킴.
- 후보 보상 항: `r_t = pnl_t - λ · max(0, current_drawdown)` 또는 `differential Sharpe ratio`
- Chan이 강조하는 **MDD duration**까지 보상에 넣는 건 RL 학습에 sparse signal이라 어렵지만, 평가지표로는 반드시 리포트해야 함.

### ③ Transaction Costs (p.24)

**책의 주장**: 거래비용 = 수수료 + 슬리피지 + 시장충격 + bid-ask spread. Chan은 **"거래빈도 높은 전략의 가장 흔한 실패 원인"**으로 지목.

**프로젝트 적용** (가장 중요한 항목 중 하나):

- 그리드 트레이딩은 **본질적으로 거래빈도가 매우 높음**. 거래비용 없이 백테스트하면 RL이 "그리드를 촘촘히 배치하라"를 학습 → 실전에서 비용에 잠식됨.
- BTC 거래소 기준 maker/taker 0.04~0.1%, **그리드 봇 기준 왕복 0.1~0.2%는 무시 못함**.
- **env에 반드시 포함**: ==매 거래마다 cost 차감==. 보상함수가 net pnl을 반영하도록.
- ==슬리피지 모델==: 1시간봉 사용 중이라 high/low 사이 random 또는 mid-price assumption + spread cost가 현실적.

### ④ Survivorship Bias (p.26)

**책의 주장**: 상장폐지된 주식 빼고 백테스트하면 수익률 과대평가.

**프로젝트 적용**:

- BTC 단일 자산이라 직접 해당 없음.
- **하지만 동일 메커니즘의 함정 존재**: "강세장 데이터에서만 학습"하면 똑같은 편향. **2018, 2022 약세장과 2020-2021 폭등장을 모두 포함**해야 함. yfinance 1시간봉이라면 2017~현재 전 구간 사용 권장.
- 추가로 ==**regime별 성과 분리 리포트**== (강세장/약세장/횡보장) — 이게 Chan의 survivorship bias 경고를 BTC 맥락에 옮긴 것.

### ⑤ 시간에 따른 성과 변화 (p.27)

**책의 주장**: 전략은 시간이 지날수록 alpha를 잃는다. 최근 성과가 더 중요하고, **walk-forward analysis** 필요.

**프로젝트 적용**:

- **이게 RL 트레이딩 프로젝트의 핵심 약점**입니다. 단순 train/test 분할은 부족.
- ==**Walk-forward validation 권장**==: 6개월 학습 → 1개월 테스트 → 윈도우 이동 → 반복. 이러면 BTC regime 변화에 강건한지 검증 가능.
- 졸업 디펜스에서 "왜 walk-forward를 썼는가" → Chan §2.2.5와 §2.2.6 인용 가능.

### ⑥ Data-Snooping Bias (p.28) — **이게 가장 위험합니다**

**책의 주장**: 같은 데이터에 여러 변형 전략을 시도하면 우연히 좋은 결과가 나올 확률이 누적. p-hacking과 동일.

**프로젝트 적용** (RL 프로젝트의 가장 큰 함정):

- PPO 하이퍼파라미터(clip ratio, entropy coef, lr, gamma, batch size) + 보상함수 가중치(λ들) + 상태변수 정규화 방법까지 **건드릴 수 있는 손잡이가 수십 개**. 테스트셋 보고 "이게 좋네"하며 조정하면 곧바로 data snooping.
- ==**방어책 3가지**==:
    1. **Hold-out test set 봉인** — 최종 1번만 평가
    2. **Validation set 별도** — 하이퍼파라미터 튜닝 전용
    3. **여러 random seed로 학습** (보통 5~10개) → seed별 분산 보고. 이건 RL 트레이딩 논문의 표준 reporting이 됐습니다.
- 발표 자료에 "**우리는 X번의 실험 중 best를 보고하지 않고, N개 seed의 평균±std를 보고함**" 명시하면 디펜스 방어력 높음.

### ⑦ "Fly Under the Radar" (p.30)

**책의 주장**: 대형 펀드가 알아챈 전략은 곧 alpha가 사라진다. 작은 시장, 마이너 자산을 노려라.

**프로젝트 적용**:

- BTC 그리드는 이미 전 세계가 하고 있음 → 이 논리대로면 "alpha 없어야" 함.
- 그래서 **반대로 디펜스 무기**: "전통적 그리드는 정적 파라미터로 한계가 있고, **RL이 동적 파라미터 조정으로 잔존 alpha를 노린다**"가 프로젝트의 motivation 진술이 됩니다. Chan §2.2.7을 인용하며 "왜 단순 그리드가 아닌 RL 그리드인가"를 정당화.

---

## §2.2 → 보상함수/env 설계 체크리스트로 압축

졸업 프로젝트에 직접 옮기면 이렇게 됩니다:

|Pitfall|env 설계 반영|보상함수 반영|평가 단계 반영|
|---|---|---|---|
|① Benchmark|—|excess return over B&H|B&H, static grid 비교 표|
|② Drawdown|잔고 추적|drawdown 페널티 항|MDD, MDD duration 보고|
|③ Transaction cost|maker/taker fee + slippage|net pnl 사용|거래빈도 통계 보고|
|④ Survivorship 유사|다양한 regime 데이터|—|regime별 성과 분리|
|⑤ 시간 변화|walk-forward 윈도우|—|walk-forward 결과|
|⑥ Data-snooping|train/val/test 3분할|—|multi-seed mean±std|
|⑦ Radar|—|—|"왜 RL인가" motivation|

이 7개를 디펜스 슬라이드 1장에 넣고 각각 어떻게 대응했는지 표로 보여주면, 심사위원의 단골 질문 80%는 선제적으로 막을 수 있습니다. **§2.2 자체는 10페이지밖에 안 되고 수식도 거의 없어서 1-2시간이면 정독 가능**하니, 분류 B에 뒀지만 시간 투자 대비 효율은 매우 높은 섹션입니다.