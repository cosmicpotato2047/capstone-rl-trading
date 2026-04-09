# CLAUDE.md — 프로젝트 브리핑

Claude Code가 이 프로젝트에서 작업할 때 반드시 숙지해야 할 규칙과 맥락이다.
상세 설계는 `docs/MDP.md`, `docs/RELATED_WORK.md`, `config/experiment_config.yaml`을 참고한다.

---

## 절대 금지 규칙

1. **`data/processed/btc_test.parquet` 열람 금지**
   테스트셋은 최종 평가 전까지 완전 봉인된다.
   어떤 코드, 노트북, 스크립트에서도 test 파티션을 로드하거나 출력해서는 안 된다.
   Gort et al.(2022) 백테스트 과적합 방지 설계의 핵심이다.

2. **`data/raw/`의 원본 파일 수정 금지**
   다운로드한 OHLCV 원본은 불변이다. 전처리는 반드시 `data/processed/`에 저장한다.

---

## 프로젝트 핵심 — 반드시 숙지

### 이 프로젝트는 방향성 트레이딩이 아니다

BTC 가격이 오를지 내릴지 예측하는 시스템이 아니다.
**변동성 자체에서 수익을 추구하는 동적 그리드 트레이딩**이다.

- 잘못된 접근: "RSI가 30 이하면 매수" 같은 방향성 신호
- 올바른 접근: "지금 변동성 수준에서 그리드 간격을 얼마로 설정할 것인가"

### Action은 연속 2차원이다. 이산이 아니다

```python
# 틀린 구현
action_space = Discrete(3)  # Buy / Sell / Hold  ← 절대 안 됨

# 올바른 구현
action_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
# [0] aggressiveness  → 매수 간격 결정
# [1] profit_target   → 매도 간격 결정
```

### Sell 우선 원칙

같은 봉에서 buy와 sell이 동시 체결 가능할 때 **sell을 먼저 처리**한다.

---

## MDP 빠른 참조

### State (5차원, rolling z-score 정규화)

```
[0] log_price            = log(price / price.rolling(168).mean())
[1] divergence           = (avg_price - price) / avg_price  # 미보유 시 0
[2] holdings_value_ratio = (holdings × price) / start_capital  # 미보유 시 0
[3] cash_ratio           = cash / start_capital
[4] volatility           = ATR(168) / price
```

### Action → 주문 변환 공식

```python
buy_hi_gap  = 0.0001 + aggressiveness * 0.05   # [0.01%,  5%]
buy_lo_gap  = 0.001  + aggressiveness * 0.10   # [0.10%, 10%]
sell_lo_gap = 0.0001 + profit_target  * 0.05   # [0.01%,  5%]
sell_hi_gap = 0.001  + profit_target  * 0.15   # [0.10%, 15%]

buy_hi  = price * (1 - buy_hi_gap)
buy_lo  = price * (1 - buy_lo_gap)
sell_lo = price * (1 + sell_lo_gap)
sell_hi = avg_price * (1 + sell_hi_gap)  # avg_price=0이면 price fallback
```

### Reward

```python
step_reward = (equity_t - equity_{t-1}) / start_capital - fee_rate * n_trades
# 사이클 종료 시 추가:
step_reward += cycle_pnl_pct + cycle_alpha / cycle_hours
```

- `fee_rate`: 0.05% (Binance maker fee)
- `cycle_alpha`: 0.5 초기값 (Val 셋 튜닝 예정)
- 사이클: holdings == 0 → 첫 체결 시 시작 / holdings → 0 복귀 시 종료

---

## 데이터

- **소스**: ccxt Binance API (`BTC/USDT`, 1시간봉)
- **yfinance는 보조 용도만**: 최근 730일 제한으로 학습 기간 전체 커버 불가

| 구분 | 기간 | 파일 |
|------|------|------|
| Train | 2020.01 ~ 2022.12 | `data/processed/btc_train.parquet` |
| Validation | 2023.01 ~ 2023.12 | `data/processed/btc_val.parquet` |
| Test | 2024.01 ~ | `data/processed/btc_test.parquet` ← **봉인** |

---

## 현재 구현 상태 (2026-04-09 기준)

### 완료
- [x] 프로젝트 폴더 구조 및 스캐폴딩
- [x] `config/experiment_config.yaml` 전체 파라미터 확정
- [x] `src/utils/config.py` YAML 로더
- [x] `scripts/download_data.py` — ccxt Binance 1h 다운로드 (54,933캔들)
- [x] `scripts/preprocess_data.py` — ATR(직접구현), log_price, rolling z-score, parquet 저장
- [x] `src/env/trading_env.py` — 전체 구현 완료, gymnasium env_checker 통과
- [x] 주간 보고서 템플릿 (`reports/WEEKLY_TEMPLATE.md`)

### 미구현 (구현 순서대로)
- [ ] `tests/test_trading_env.py` — 포트폴리오 수학 검증 (평단가, 수수료, 사이클 보너스)
- [ ] `src/agents/baselines.py` — Buy-and-Hold, 고정 그리드(1%/2%/5%), ATR 비례 그리드
- [ ] `notebooks/01_data_exploration.ipynb` — 데이터 분포, ATR 시계열, zscore 시각화
- [ ] `src/agents/ppo_agent.py` + `scripts/train_ppo.py`
- [ ] `src/evaluation/metrics.py` — Sharpe, MDD, 누적수익률, 사이클 통계
- [ ] `src/evaluation/behavior.py` — (state, action) 수집 및 분석
- [ ] 노트북 02~05

---

## 코드 작성 원칙

### 구조
- **로직은 반드시 `src/`에** — 노트북은 `src/`를 import해서 사용, 로직 직접 작성 금지
- **스크립트는 `scripts/`에** — 커맨드라인 실행 진입점. `src/`를 호출하는 얇은 레이어
- **실험 파라미터는 `config/experiment_config.yaml`에** — 코드 내 하드코딩 금지

### 의존성
- 기술 지표: ATR은 `src/data/preprocessor.py`에서 pandas로 직접 계산. 외부 라이브러리 불필요
- 데이터 다운로드: `ccxt` (Binance). `yfinance`는 보조 용도만
- RL: `stable-baselines3` + `gymnasium`. `gym` (구버전) 사용 금지
- 실험 추적: `mlflow` (로컬). Weights & Biases 사용 금지

### 환경 검증
`trading_env.py` 수정 후 반드시 실행:
```python
from gymnasium.utils.env_checker import check_env
check_env(env)
```

---

## 주요 문서 위치

| 문서 | 경로 | 내용 |
|------|------|------|
| MDP 설계 | `docs/MDP.md` | State/Action/Reward 설계 근거, 행동 분석 계획 |
| 선행 연구 | `docs/RELATED_WORK.md` | 논문 7편 역할 및 차별점 |
| 실험 설정 | `config/experiment_config.yaml` | 모든 수치 파라미터 |
| 실험 기록 | `RESEARCH_LOG.md` | 날짜별 의사결정 |
| 제안서 v3.0 | `D:\PARA\Assets\Semesters\26-1\캡디\캡스톤_주제제안서_v3.0.docx` | 지도교수 제출용 공식 문서 |
