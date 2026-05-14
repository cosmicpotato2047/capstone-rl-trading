# FinRL Framework — Liu et al. (2020+)

> Liu, X-Y., Yang, H., Chen, Q., Zhang, R., Yang, L., Xiao, B., Wang, C.D. (2020). _FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance._ NeurIPS Deep RL Workshop. arXiv: 2011.09607.
> 코드: github.com/AI4Finance-Foundation/FinRL

## 요지

1. **DRL 트레이딩의 사실상 표준 오픈소스 프레임워크.**
   gym + stable-baselines3 위에 trading 환경/지표/평가/거래소 연동을 얹은 3-layer 구조.
2. 3-layer 아키텍처: **(1) Market Environment** (data, indicators, simulators) → **(2) DRL Agents** (DQN, DDPG, PPO, SAC, A2C, TD3) → **(3) Application** (stock/portfolio/crypto/multi-asset).
3. 시장: NASDAQ-100, DJIA, S&P 500, HSI, SSE 50, CSI 300, crypto (FinRL-Crypto fork).

## 주요 구성

### 표준 환경
- `StockTradingEnv`: 단일/다중 종목 long-only or long-short
- `PortfolioOptimizationEnv`: 비중 결정 (continuous action ∈ simplex)
- 모든 환경이 `gym.Env` 인터페이스 준수

### 보상함수 옵션 (라이브러리에 내장)
- Direct PnL: $r_t = \Delta \text{equity}$
- Sharpe-based: 누적/EMA Sharpe
- Drawdown-penalized
- Turbulence-aware (시장 위기 감지 후 포지션 축소)

### 표준 평가
- Cumulative return, Sharpe, MDD, Calmar, Sortino
- Benchmark: equal-weight portfolio, market index, min-variance

## 우리 프로젝트와의 연결점

1. **우리 시스템은 FinRL과 독립적으로 구현됐다.**
   장점: 그리드 트레이딩이라는 특수 케이스에 최적화. 단점: 학계 표준 베이스라인과 비교가 어렵다.
   **디펜스 전략**: "FinRL의 stock/portfolio 환경은 directional이라 grid에 부적합 → 우리만의 환경이 필요했다"고 명시.

2. **재사용 가능한 부분**:
   - **Turbulence index** — Bull/Bear/Sideways regime 분류기로 우리 행동 분석에 활용 가능
   - **표준 metric 함수** — Sharpe/Sortino/Calmar 계산 검증용 (우리 [src/evaluation/metrics.py](src/evaluation/metrics.py)와 cross-check)

3. **FinRL-Crypto (Gort 2022) 직접 차용 가능**:
   - PBO 계산 코드 (over-fitting 검증)
   - Walk-forward fold 생성 유틸리티
   - 다중 seed ensemble pattern

4. **2학기 자산 확장 시**: SOXL/AAPL을 다룰 때 FinRL의 `StockTradingEnv`를 base로 직접 fork하면 데이터 파이프라인 재발명 안 해도 됨.

## 한계 (왜 우리가 직접 짰는지 설명용)

- FinRL의 `StockTradingEnv`는 action이 "각 종목 매수/매도/홀드" — directional만 지원
- 그리드 트레이딩(양방향 동시 호가, 사이클 개념)을 표현할 수 없음
- 우리 환경의 핵심 가치: **사이클 + n_splits 균등 분할 + ATR 비례 공식** — FinRL에 없는 구조

## 백링크

- [[gort_2022_crypto_overfitting]] — FinRL을 fork한 PBO 검증 프레임
- [[ppo_schulman_2017]] — FinRL이 PPO를 표준 알고리즘으로 채택한 근거
- [[walk_forward_cv]] — FinRL-Crypto의 fold 생성 방식

## 출처

- [arXiv 2011.09607](https://arxiv.org/abs/2011.09607)
- [GitHub AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- [ACM ICAIF 2021](https://dl.acm.org/doi/10.1145/3490354.3494366)
