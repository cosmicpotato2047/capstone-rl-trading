# PPO 기반 BTC 동적 그리드 트레이딩

2026 컴퓨터공학과 캡스톤 디자인

## 연구 개요

고정 간격 그리드 봇과 달리, 시장 상태와 포지션 상태에 반응하여
**그리드 간격과 익절 목표를 동적으로 결정**하는 PPO 에이전트를 학습시킨다.
가격 방향을 예측하지 않고 변동성 자체에서 수익을 추구한다.

**주 연구 질문:**
> 시장 상태와 포지션 상태에 반응하여 그리드 간격과 익절 목표를 동적으로 결정하는
> PPO 에이전트가 비트코인 시장에서 고정 그리드 전략 대비 Sharpe Ratio 기준 우위를 보이는가?

**부 연구 질문:**
> 학습된 에이전트는 어떤 시장 상태(변동성, 가격 수준)에서 어떤 간격을 선택하는가?

---

## MDP 정의

### State (5차원, rolling z-score 정규화)

| # | 변수 | 수식 | 역할 |
|---|------|------|------|
| 0 | log_price | `log(price / price.rolling(168).mean())` | 시장: 가격 수준 |
| 1 | divergence | `(avg_price - price) / avg_price` (미보유 시 0) | 포지션: 손익 |
| 2 | holdings_value_ratio | `(holdings × price) / start_capital` | 포지션: 보유 규모 |
| 3 | cash_ratio | `cash / start_capital` | 포지션: 여력 |
| 4 | volatility | `ATR(168) / price` | 시장: 변동성 레짐 |

### Action (2차원 연속, [0, 1]²)

| 변수 | 범위 | 결정하는 것 |
|------|------|------------|
| aggressiveness | [0, 1] | buy_hi_gap → [0.01%, 5%], buy_lo_gap → [0.1%, 10%] |
| profit_target | [0, 1] | sell_lo_gap → [0.01%, 5%], sell_hi_gap → [0.1%, 15%] |

주문: buy_hi / buy_lo / sell_lo / sell_hi 4개 고정. 체결은 다음 봉 고/저가 기준.

### Reward

```
매 스텝:      (equity_t - equity_{t-1}) / start_capital - fee × n_trades
사이클 종료:  위 + cycle_pnl_pct + alpha / cycle_hours
```

사이클: holdings == 0 → 첫 체결 시 시작 / holdings → 0 복귀 시 종료

---

## 베이스라인 비교

| 전략 | 설명 | 비교 목적 |
|------|------|----------|
| Buy-and-Hold | BTC 단순 보유 | 절대 기준선 |
| 고정 그리드 | 1% / 2% / 5% 고정 간격 | "적응이 의미 있는가?" |
| ATR 비례 그리드 | `gap = k × (ATR_168 / price)` (규칙 기반) | "학습이 규칙보다 나은가?" |
| PPO 에이전트 | 동적 간격 학습 | 피실험자 |

---

## 환경 설정

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements-dev.txt
```

> ⚠️ yfinance 1h 데이터는 최근 730일까지만 제공한다.
> 학습 기간(2020~2022) 데이터는 Binance API(ccxt)로 다운로드한다.

## 데이터 다운로드 및 전처리

```bash
python scripts/download_data.py
python scripts/preprocess_data.py
```

## 베이스라인 실행

```bash
python scripts/run_baselines.py
```

## 학습 실행

```bash
python scripts/train_ppo.py
```

## 실험 결과 확인 (MLflow)

```bash
mlflow ui   # http://localhost:5000
```

---

## 데이터 분할

| 구분 | 기간 | 용도 |
|------|------|------|
| Train | 2020.01 ~ 2022.12 | 에이전트 학습 |
| Validation | 2023.01 ~ 2023.12 | 하이퍼파라미터 튜닝 |
| Test | 2024.01 ~ | 최종 평가 — **학습 완료 전 열람 금지** |

---

## 프로젝트 구조

```
capstone-rl-trading/
├── config/                  # 실험 설정 (하이퍼파라미터)
├── data/raw/                # 다운로드 원본 (불변)
├── data/processed/          # 전처리 완료 parquet (.gitignore)
├── src/
│   ├── data/                # 다운로더, 전처리기
│   ├── env/                 # Gymnasium 트레이딩 환경
│   ├── agents/              # PPO 래퍼, 베이스라인
│   ├── evaluation/          # 지표, 행동 분석
│   └── utils/               # 설정 로더, 시각화
├── scripts/                 # 실행 진입점
├── notebooks/               # 탐색/시각화 전용
├── experiments/             # 실험별 결과 (config + 로그)
├── reports/                 # 학기별 보고서
├── tests/                   # 단위 테스트
└── papers/                  # 참고 논문
```

---

## 1학기 목표 (2026년 3~6월)

| 기간 | 목표 | 산출물 |
|------|------|--------|
| 3~4월 | 환경 구현 + 데이터 파이프라인 | trading_env.py, 전처리 완료 데이터 |
| 4~5월 | PPO 학습 + 베이스라인 비교 | 학습 곡선, Sharpe 비교표 |
| 6월 | 행동 패턴 분석 + 중간 보고서 | aggressiveness/profit_target vs 시장 레짐 분석 |

## 2학기 확장 계획

- Ablation: state 변수별 기여도 측정
- SAC 전환: 연속 행동 공간에서 더 안정적인 off-policy 알고리즘
- 주문 개수 가변화: 2개 고정 → n개 가변

---

## 관련 문서

| 문서 | 위치 | 내용 |
|------|------|------|
| 주제 제안서 v3.0 | `D:\PARA\Assets\Semesters\26-1\캡디\캡스톤_주제제안서_v3.0.docx` | 지도교수 제출용 공식 문서 |
| MDP 설계 | [`docs/MDP.md`](docs/MDP.md) | State / Action / Reward 설계 근거 |
| 선행 연구 | [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | 참고 논문 7편 역할 및 차별점 |
| 실험 기록 | [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | 날짜별 의사결정 및 실험 결과 |
