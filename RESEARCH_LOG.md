# Research Log

날짜별 의사결정, 실험 결과, 실패 원인, 발견 사항을 기록한다.
최종 보고서 작성 시 이 로그가 근거 자료가 된다.

---

## 2026-04-05 — 프로젝트 초기화 + MDP 확정

### 프로젝트 초기화
- 폴더 구조 스캐폴딩 완료 (data/, src/, experiments/, notebooks/, scripts/, config/, tests/)
- git init, .gitignore, requirements.txt, README.md 생성
- 알고리즘: stable-baselines3 PPO (연속 행동 공간)

### MDP 설계 방향 결정 — 그리드 트레이딩으로 전환
제안서(v2.1)의 이산 3행동(Buy/Sell/Hold) 방향성 트레이딩에서
**연속 2차원 동적 그리드 트레이딩**으로 설계 변경.
- 변경 이유: 방향 예측 의존성 완전 제거 → 변동성 자체에서 수익 추구

### 확정된 MDP

**State (5차원):**
| 변수 | 수식 |
|------|------|
| log_price | `log(price / price.rolling(168).mean())` |
| divergence | `(avg_price - price) / avg_price` (미보유 시 0) |
| holdings_value_ratio | `(holdings × price) / start_capital` |
| cash_ratio | `cash / start_capital` |
| volatility | `ATR(168) / price` |
→ 전처리: rolling z-score(window=168) 정규화

**Action (2차원 연속 [0,1]²):**
- aggressiveness → buy_hi_gap [0.01%,5%], buy_lo_gap [0.1%,10%]
- profit_target → sell_lo_gap [0.01%,5%], sell_hi_gap [0.1%,15%]
- 주문 4개 고정: buy_hi, buy_lo, sell_lo, sell_hi
- 체결: 다음 봉 high/low 기준

**Reward:**
- 매 스텝: `(equity_t - equity_{t-1}) / start_capital - fee × n_trades`
- 사이클 종료 시 추가: `cycle_pnl_pct + alpha / cycle_hours`
- 사이클: holdings==0 → 첫 체결 시작 / holdings→0 복귀 종료
- fee_rate: 0.05% (Binance maker fee)
- cycle_alpha: 0.5 (튜닝 예정)

**데이터:**
- 1시간봉, BTC/USDT, Binance API (ccxt) — yfinance 1h는 730일 제한
- Train: 2020.01~2022.12 / Val: 2023 / Test: 2024~ (봉인)

### 결정된 베이스라인
1. Buy-and-Hold
2. 고정 그리드 (1%, 2%, 5%)
3. ATR 비례 그리드 (k=0.5, 1.0, 2.0)

### 연구 질문 확정
- 주: PPO 동적 그리드가 고정 그리드 대비 Sharpe Ratio 기준 우위를 보이는가?
- 부: 어떤 시장 레짐에서 어떤 간격을 선택하는가?

### 남은 미결 사항
- [x] 제안서 v3.0 작성 완료 → `D:\PARA\Assets\Semesters\26-1\캡디\캡스톤_주제제안서_v3.0.docx`
- [x] _process_fills() 구현 (sell 우선 원칙 적용)
- [ ] order_size_fraction 확정 (현재 0.5 임시 — Val 셋 실험 후 조정)
- [x] ccxt 다운로드 스크립트 작성 (scripts/download_data.py)

---

## 2026-04-09 — 데이터 파이프라인 완성 + 환경 핵심 로직 구현

### 데이터 파이프라인
- `scripts/download_data.py` 완성 및 실행
  - ccxt Binance API, BTC/USDT 1h봉
  - 결과: 54,933개 캔들, 2020-01-01 ~ 2026-04-09
  - 저장: `data/raw/btc_usdt_1h.csv`

- `scripts/preprocess_data.py` 완성 및 실행
  - ATR(168): pandas-ta 미사용, True Range rolling mean 직접 구현
    - 이유: pandas-ta가 Python 3.11과 호환되지 않음 (3.12+ 전용 빌드)
  - log_price = log(close / close.rolling(168).mean())
  - rolling z-score(window=168) 정규화
  - 결과: Train 25,916행 / Val 8,736행 / Test 19,901행
  - zscore 범위: log_price [-6.44, +5.65], volatility [-5.23, +7.21]
  - observation_space [-5, 5] 설계 적합 확인 (극단값 clip 처리)

### 환경 구현
- `src/env/trading_env.py` 5개 핵심 메서드 구현 완료
  - `_get_observation()`: 5D state, clip [-5, 5] 후 float32
  - `_execute_buy()`: 가중평균 avg_price, 사이클 시작 감지 (holdings 0→>0)
  - `_execute_sell()`: 전량 청산 감지 → `_close_cycle()` 자동 호출
  - `_close_cycle()`: cycle_pnl_pct + alpha/cycle_hours 보너스, completed_cycles 기록
  - `_process_fills()`: Sell 우선 원칙 — sell_lo → sell_hi → buy_hi → buy_lo 순

### 검증
- `gymnasium.utils.env_checker` 통과
- 랜덤 에이전트 1 에피소드 (Val set, seed=42)
  - 8,567스텝, 수익률 -0.72%, 거래 1,540회, 완료 사이클 1회
  - 랜덤 정책은 포지션 청산이 드물다는 것 확인 → PPO 학습 필요성 실증
  - 그래프 저장: `reports/semester1/figures/random_agent_portfolio.png`

### 인프라 이슈 및 해결
- `ModuleNotFoundError: No module named 'src'` → pyproject.toml 생성 + editable 설치
- mlflow + pyarrow 버전 충돌 → 엄격한 핀 제거, `>=` 하한선 방식으로 해결
- Python 3.13 기본 설치 상태 → Python 3.11 별도 설치 + uv로 가상환경 격리

### 주간 보고서
- 5주차 보고서 작성: `reports/semester1/week05_2026-04-10.md`
- 주간 보고서 템플릿 생성: `reports/WEEKLY_TEMPLATE.md`

---
