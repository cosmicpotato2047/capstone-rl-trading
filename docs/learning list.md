[[how to]] 

[[Ernie Chan - Quantitative Trading_How to Build Your Own Algorithmic Trading Business]] 

[[López de Prado - Advances in Financial Machine Learning]] 

## 1. 퀀트 투자 — 큰 그림부터

**필수 개념**

- `Factor investing` (Fama-French 3/5 factor model) — 모든 퀀트의 출발점
- `Mean reversion` vs `Momentum` — 두 축이 거의 모든 전략의 기반
- `Statistical arbitrage`, `Pairs trading` (Avellaneda-Lee 모델)
- `Market microstructure` — 호가창, 스프레드, 시장가/지정가 메커니즘
- `Market making` — 그리드 봇의 직계 조상

**리스크/성과 측정**

- `Sharpe ratio`, `Sortino`, `Calmar`, `Information Ratio`
- `Maximum Drawdown`, `Ulcer Index`, `Pain Index`
- `Kelly Criterion`, `Risk parity`, `Position sizing`
- `Backtest overfitting` — Marcos López de Prado, _Advances in Financial Machine Learning_ (네 프로젝트 핵심 위험)
- `Walk-forward analysis`, `Combinatorial Purged Cross-Validation`

---

## 2. 그리드 봇 — 너의 본진

**기본 변형**

- `Constant grid` vs `Geometric grid` (등차 vs 등비 간격)
- `Martingale grid` (DCA-style averaging down)
- `Infinity grid` / unlimited grid
- `Bollinger Band grid`, `ATR-based dynamic grid` ← 네가 하는 거
- `Avellaneda-Stoikov market making` — 학술적 그리드의 원형

**상용 시스템 (역공학용)**

- `3Commas DCA bot`, `Pionex grid bot`, `Bitsgap`
- `Hummingbot` (오픈소스) — 코드 직접 읽으면서 공부 가능

**찾아볼 키워드**

- "Optimal grid spacing volatility"
- "Inventory risk market making"
- "Adverse selection grid trading"

---

## 3. 시장별 주류 전략

|시장|핵심 키워드|
|---|---|
|**주식**|Long/short equity, Factor tilting, Pair trading, Statistical arbitrage, VWAP/TWAP execution|
|**비트코인/크립토**|Funding rate arbitrage, Basis trade (현물-선물 스프레드), CEX-DEX arbitrage, Grid bot, MEV|
|**외환 (FX)**|Carry trade, Trend following (CTA), Triangular arbitrage, Interest rate parity|
|**원자재**|Contango/Backwardation, Calendar spread, Roll yield, Trend following (CTA가 주력)|
|**부동산**|REITs factor model, Cap rate arbitrage — _RL 적용은 거의 없음. 유동성·체결 빈도가 RL과 안 맞는다_|

> 부동산은 학습 대상에서 제외해도 된다. Step 단위가 분/시간이 아니라 월/년이라 MDP 설계 자체가 성립 안 한다.

---

## 4. 강화학습 — 알고리즘 트리

**기본기 (이 순서대로)**

1. `MDP`, `Bellman equation`, `Value function`, `Policy`
2. `Q-learning`, `SARSA` — tabular
3. `DQN` (Deep Q-Network), `Double DQN`, `Dueling DQN`
4. `Policy gradient`, `REINFORCE`
5. `Actor-Critic`, `A2C/A3C`
6. **`PPO`** (Proximal Policy Optimization) ← 네가 쓰는 거. 논문 직접 읽기
7. `TRPO`, `SAC`, `TD3` — 비교용

**고급 주제**

- `On-policy` vs `Off-policy` — PPO가 왜 on-policy인지
- `Reward shaping` — 함정 많음, Ng의 _Policy invariance under reward transformations_ 필독
- `Sparse vs Dense reward`
- `Exploration`: ε-greedy, entropy bonus, RND
- `Generalization in RL`, `Distributional shift`
- `POMDP` — 시장은 본질적으로 부분관측

**교재 추천**

- Sutton & Barto, _Reinforcement Learning: An Introduction_ — 무료 PDF, 1~6장만 정독
- Spinning Up in Deep RL (OpenAI) — PPO 구현 디테일
- _Deep RL Hands-On_ (Lapan) — 실전 코드

---

## 5. RL × 금융 — 가장 중요

너의 프로젝트 직결 영역. 이게 핵심이다.

- `FinRL` framework (논문 + GitHub 코드)
- **Zhang, Zohren, Roberts** — _Deep Reinforcement Learning for Trading_ (2020)
- **Gort et al. (2022)** — _Deep Reinforcement Learning for Cryptocurrency Trading_ (네가 백테스트 과적합 방지로 인용 중)
- `Reward hacking` in trading agents
- `Regime shift` / `Non-stationarity` — 학습기와 테스트기 분포 차이
- `Sim2Real` 문제의 금융 버전: slippage, latency, partial fill
- `Distributional RL` (C51, QR-DQN) — 금융 적용 사례 많음
- `Hierarchical RL` — 그리드 간격 결정(상위) + 주문 집행(하위) 구조와 잘 맞음

---

## 6. 너의 프로젝트 약점 보강용

지금 RESEARCH_LOG 흐름 보면 이쪽이 부족할 가능성:

- **`Reward design`** — exp027의 asymmetric reward는 좋은 시도지만, 이론 배경(Ng의 reward shaping, prospect theory)을 깔고 가야 방어가 된다
- **`Hyperparameter tuning under non-stationarity`** — Optuna 3단계 돌리는 게 의미 있는지 통계적으로 검증하는 법
- **`Action space design`** — 연속 vs 이산, Box vs Tuple, 네가 [0,1]² 쓰는 근거를 _Continuous control with deep RL_ (Lillicrap 2015)에 연결

---

## 우선순위 추천

시간이 한정적이라면 **이 순서**:

1. Sutton & Barto 1~6장 (RL 기초)
2. PPO 논문 + Spinning Up PPO 페이지 (네 알고리즘)
3. López de Prado, _Advances in Financial ML_ 1~7장 (백테스트 과적합)
4. Zhang/Zohren/Roberts 2020 + Gort 2022 (RL 트레이딩 핵심 논문)
5. Avellaneda-Stoikov 마켓 메이킹 논문 (그리드 이론적 뿌리)

나머지는 위 5개 읽고 나서 필요할 때 골라 보는 식으로 충분하다.