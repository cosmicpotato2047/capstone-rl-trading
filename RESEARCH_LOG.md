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

## 2026-04-12 — 주문 사이징 재설계 + Reward 단순화 + State 개선

### 주문 사이징 재설계: order_size_fraction → n_splits 균등 분할

**문제**: 기존 `order_size_fraction=0.5` 방식은 매수마다 가용 현금의 50%를 소비하는
기하급수적 감소 구조였다. 현실적인 그리드 봇 운용 방식(정해진 예산을 균등 소비)과 불일치.

**결정**: 사이클 시작 시 현금을 n_splits 슬롯으로 등분, 슬롯당 균등 금액 고정 소비.

```python
cycle_slot_size = cycle_start_cash / n_splits       # 1슬롯 예산
per_order_size  = cycle_slot_size / n_buy_orders    # 주문당 고정 금액
# cycle_budget_remaining < per_order_size → 추가 매수 완전 차단
```

- `n_splits=4` 기본값 (Val 셋 튜닝 예정)
- `config/experiment_config.yaml`에서 `order_size_fraction` 제거, `n_splits: 4` 추가

**매도 주문 크기**: threshold_btc 기준 이분법
```python
threshold_btc = cycle_slot_size / avg(sell_lo, sell_hi)  # 1슬롯에 해당하는 BTC 수량
sell_qty = holdings                      # 전량 청산  (holdings ≤ threshold_btc)
sell_qty = holdings / n_sell_orders      # 균등 분할 (holdings > threshold_btc)
```
보유량이 1슬롯 이하로 줄면 전량 청산하여 사이클 종료를 보장한다.

---

### Reward 단순화: 사이클 보너스 완전 제거

**문제**: 기존 보너스 `cycle_pnl_pct + alpha / cycle_hours`의 구조적 결함
- 손실 사이클도 `alpha / cycle_hours`가 크면 양수 보너스 발생 (잘못된 강화)
- n_splits + threshold_btc 설계로 사이클 구조는 이미 기계적으로 보장됨
- `cycle_alpha` 하이퍼파라미터 하나를 줄일 수 있음

**결정**: 사이클 보너스 완전 제거. step_reward만 사용.
```python
step_reward = (equity_t - equity_{t-1}) / start_capital - fee_rate * n_trades
```
사이클 종료 시 `completed_cycles` 리스트에 통계만 기록 (보너스 없음).

- `config/experiment_config.yaml`에서 `cycle_alpha` 파라미터 제거
- `_close_cycle()`: 보너스 계산 로직 제거, `return 0.0`
- `completed_cycles` dict에서 `"bonus"` 키 제거

---

### State 개선: divergence에 last_avg_price 도입

**문제**: 미보유 구간(holdings==0)에서 divergence=0으로 고정되면
"방금 70k에 전량 매도, 현재 60k" 상태와 "한 번도 거래 안 함" 상태가
동일하게 인코딩된다. 재진입 타이밍 신호가 누락됨.

**결정**: 전량 청산 시 평단가를 `last_avg_price`에 보존. 미보유 구간에 활용.
```python
if holdings > 0:
    divergence = (avg_price - price) / avg_price             # 보유 중
elif last_avg_price > 0:
    divergence = (last_avg_price - price) / last_avg_price   # 미보유, 직전 사이클 있음
else:
    divergence = 0.0                                         # 에피소드 초반
```

---

### 검증
- `gymnasium.utils.env_checker` 통과
- 랜덤 에이전트 Val set 실행:
  - 174 사이클 완료 (n_splits 도입 전: 1 사이클)
  - `last_avg_price` 정상 저장 확인 (17311.59)
  - 미보유 구간 divergence = -0.0096 (0이 아님, last_avg_price 활용 중)
  - `completed_cycles` dict에 `bonus` 키 없음 확인

### 아이디어 기록 (2학기/논문 확장 시 검토)
- **레짐 스위칭 그리드 베이스라인**: 변동성 수준(상/중/하)별로 최적 그리드 파라미터를
  오프라인 그리드서치로 확정 후 런타임에 레짐 감지하여 적용하는 방식.
  현재 ATR 비례 그리드보다 강력한 베이스라인이 되어 PPO와의 비교가 더 설득력 있어진다.
  1학기 범위 초과 → 2학기 혹은 논문 확장 시 추가 예정.

### 커밋
- `feat: n_splits equal-slot order sizing` (feature/n-splits-order-sizing → main)
- `feat: remove cycle bonus, add last_avg_price for divergence` (feature/remove-cycle-bonus-add-last-avg-price → main)

---
