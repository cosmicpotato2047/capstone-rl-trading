# MDP 정의 — BTC 동적 그리드 트레이딩 환경

이 문서는 `src/env/trading_env.py`의 설계 의도와 근거를 기록한다.
구현 파라미터는 `config/experiment_config.yaml`, 코드 인터페이스는 `trading_env.py` docstring을 참고한다.

> **본 문서의 위치 (2026-05-14 기준)**:
> - 본 문서는 Phase 1 (환경 설계) 시점의 상세 설계 근거를 담고 있다.
> - 현 시점(Phase 3)의 RQ와 실제 구현 상태는 [`PROJECT_GOAL.md`](PROJECT_GOAL.md), [`../ROADMAP.md`](../ROADMAP.md), [`../CLAUDE.md`](../CLAUDE.md) 참조.
> - Phase 1 이후 변경 사항은 본 문서 끝의 "Phase 2~3 주요 변경" 섹션 참조.

---

## 개요

**목표**: 가격 방향을 예측하지 않고, 현재 시장 상태와 포지션 상태에만 반응하여
그리드 간격과 익절 목표를 동적으로 결정하는 에이전트 학습.

본 환경 설계는 **본 졸업 논문 §3 Method (환경 정의)** 의 근거가 된다.

**데이터**: yfinance → Binance API (ccxt) BTC/USDT **1시간봉**
(yfinance 1h는 최근 730일 제한 → 학습 기간 전체 커버를 위해 ccxt 사용)

---

## 1. State (5차원)

### 변수 정의

| 인덱스 | 변수명 | 수식 | 미보유 시 |
|--------|--------|------|----------|
| 0 | `log_price` | `log(price / price.rolling(168).mean())` | 정상 계산 |
| 1 | `divergence` | `(avg_price - price) / avg_price` | 0 |
| 2 | `holdings_value_ratio` | `(holdings × price) / start_capital` | 0 |
| 3 | `cash_ratio` | `cash / start_capital` | 1.0 |
| 4 | `volatility` | `ATR(168) / price` | 정상 계산 |

rolling window = 168 = 7일 × 24h (1시간봉 기준 1주일)

### 설계 근거

```
시장 상태 (에이전트가 모르는 것):
  log_price   → 가격이 최근 평균 대비 얼마나 높은가/낮은가 (레짐 판단)
  volatility  → 지금 시장이 얼마나 출렁이는가 (그리드 간격 결정의 핵심)

포지션 상태 (에이전트가 아는 것):
  divergence          → 현재 평단 대비 얼마나 수익/손실인가
  holdings_value_ratio → 현재 BTC 보유 포지션이 자본 대비 얼마나 큰가
  cash_ratio          → 추가 매수 여력이 얼마나 남았는가
```

**`holdings_value_ratio` 선택 이유**:
단순 수량 비율(`holdings / max_holdings`)이 아닌 현재가 반영 금액 비율을 쓴다.
BTC 가격이 2배 오르면 같은 수량이더라도 포지션 리스크가 2배 — 에이전트가 이를 인식해야 한다.

**`divergence`와 `holdings_value_ratio`의 비중복성**:
- `divergence`: 평단 대비 현재가 손익률 (방향)
- `holdings_value_ratio`: 현재 포지션의 절대 규모 (크기)
예) 같은 divergence=0.05(5% 수익)라도 holdings_value_ratio=0.1(소량)과 0.9(대량)은 다른 행동이 필요하다.

### 전처리: Rolling Z-score 정규화

```python
def rolling_zscore(series, window=168):
    mean = series.rolling(window).mean()
    std  = series.rolling(window).std()
    return (series - mean) / (std + 1e-8)
```

`log_price`와 `volatility`는 전처리 단계에서 계산 후 parquet에 저장한다.
`divergence`, `holdings_value_ratio`, `cash_ratio`는 매 스텝마다 환경 내부에서 실시간 계산한다 (포트폴리오 상태 의존).

학습 전 변수 간 상관관계(±0.6 초과) 확인 후 중복 제거 여부를 재검토한다.

---

## 2. Action (2차원 연속)

### 변수 정의

| 변수 | 범위 | 결정하는 주문 |
|------|------|--------------|
| `aggressiveness` | [0, 1] | 매수 2개 (buy_hi, buy_lo) |
| `profit_target` | [0, 1] | 매도 2개 (sell_lo, sell_hi) |

### 갭 변환 공식

```python
# 매수 (aggressiveness가 클수록 현재가에 가깝게 공격적으로 매수)
buy_hi_gap  = 0.0001 + aggressiveness * 0.05   # [0.01%,  5%]
buy_lo_gap  = 0.001  + aggressiveness * 0.10   # [0.10%, 10%]

# 매도 (profit_target이 클수록 높은 수익 목표)
sell_lo_gap = 0.0001 + profit_target  * 0.05   # [0.01%,  5%]
sell_hi_gap = 0.001  + profit_target  * 0.15   # [0.10%, 15%]
```

### 주문 가격 계산

```python
buy_hi  = price * (1 - buy_hi_gap)     # 공격적 매수 (현재가에 가까움)
buy_lo  = price * (1 - buy_lo_gap)     # 보수적 매수 (현재가에서 멀리)
sell_lo = price * (1 + sell_lo_gap)    # 빠른 익절 (현재가에 가까움)
sell_hi = avg_price * (1 + sell_hi_gap) # 느린 익절 (평단가 기준)
          # avg_price == 0이면 price 기준 fallback
```

`buy_hi`와 `buy_lo`는 **서로 다른 역할의 독립적인 두 주문**이다 (범위가 아님).
aggressiveness 하나가 두 갭을 같은 방향으로 움직이지만, 각 주문은 별도 체결된다.

### 체결 로직

```python
# 다음 봉 고/저가 기준 체결 판단
if next_high >= sell_lo:  → sell_lo 체결 (빠른 익절)
if next_high >= sell_hi:  → sell_hi 체결 (느린 익절, avg_price 기준)
if next_low  <= buy_hi:   → buy_hi  체결 (공격적 매수)
if next_low  <= buy_lo:   → buy_lo  체결 (보수적 매수)
```

같은 봉에서 buy와 sell이 동시 체결 가능하다. **sell 먼저 처리**하는 것을 원칙으로 한다 (보유 포지션 정리 우선).

### 극단적 행동 억제

범위를 넓게 열었으므로 에이전트가 극단값을 선택해 과매매할 수 있다.

1. **Reward에 거래비용 반영** — 매 거래마다 0.05% 차감
2. **PPO 엔트로피 정규화** — `ent_coef=0.01`로 탐색 유지

### Phase별 Action 변천

```
Phase 1 (exp001~016):
  buy 2개 + sell 2개 고정
  고정 절대 갭 (buy_hi_gap = 0.0001 + agg * 0.05)
  Action: [aggressiveness, profit_target], Box[0,1]^2

Phase 2 (exp017~027):
  ATR 비례 스케일링 도입
  buy_hi_gap = atr_ratio * (A_b + B_b * aggressiveness)
  Sell market vs cost 명칭 분리

Phase 3 (exp028+):
  Action 의미 자체를 변경 (exp020 budget_fraction, exp021 entry_gate 등 ablation 후)
  exp029: 사이클 시작 시 1회 결정 (5D action)
  본 논문은 표준 2D action을 메인 비교 단위로 사용
```

---

## 3. Reward

### 구조

```python
# ── 매 스텝 ──────────────────────────────────────
step_reward = (equity_t - equity_{t-1}) / start_capital   # 포트폴리오 가치 변화율
            - fee_rate * n_trades_t                        # 거래비용 패널티

# ── 사이클 종료 시 추가 ────────────────────────────
if cycle_ended:
    step_reward += cycle_pnl_pct          # 사이클 수익률
    step_reward += alpha / cycle_hours    # 빠른 회전 보너스
```

### 각 항목 설명

| 항목 | 역할 | 비고 |
|------|------|------|
| `(equity_t - equity_{t-1}) / start_capital` | 매 스텝 수익 신호 | 연속적이지만 체결 없으면 희박 |
| `fee_rate × n_trades_t` | 과도한 거래 억제 | 0.05% per trade (Binance maker fee) |
| `cycle_pnl_pct` | 수익 사이클 강화 | `(cycle_end_cash - cycle_start_cash) / cycle_start_cash` |
| `alpha / cycle_hours` | 빠른 회전 강화 | alpha 초기값 0.5, Val 셋 튜닝 예정 |

### 사이클 정의

```
사이클 시작: holdings == 0 → 첫 번째 buy 체결 시
사이클 종료: holdings > 0 → 전량 매도로 holdings == 0 복귀 시

cycle_pnl_pct = (cycle_end_cash - cycle_start_cash) / cycle_start_cash
cycle_hours   = 사이클 종료 스텝 - 사이클 시작 스텝  (1스텝 = 1시간)
```

부분 체결 엣지 케이스: buy_hi만 체결 후 sell → holdings == 0이면 사이클 종료.
cycle_pnl_pct가 실제 발생한 거래 수익만 자동 반영하므로 별도 처리 불필요.

### alpha 설정 가이드

```
초기값: Val 셋의 예상 평균 cycle_pnl_pct 수준 (약 0.5에서 시작)
alpha 너무 크면: 수익보다 빠른 회전에만 집중 → 손절 사이클 양산
alpha 너무 작으면: 회전 속도 신호 무시 → 장기 교착 허용
```

### 설계에서 제외한 항목

| 항목 | 제외 이유 |
|------|----------|
| MDD 패널티 | `cycle_pnl_pct` 음수가 자연스럽게 패널티 역할 |
| 교착 상태 패널티 | 거래비용 패널티가 과도한 거래 억제. hold는 자연스러운 대기 상태 |
| 에피소드 종료 보상 | 종료 시점에만 최적화하는 근시안적 행동 유발 가능 |
| 변동성 패널티 | Bandarupalli(2025)에서 수익 기회 억제로 Buy-and-Hold 하회 확인 |

---

## 4. 행동 패턴 분석 계획 (부 연구 질문 검증)

부 연구 질문: *"학습된 에이전트는 어떤 시장 상태에서 어떤 간격을 선택하는가?"*

학습 완료 후 Val 셋에서 에이전트를 구동하며 매 스텝의
(state, action) 쌍을 수집하여 아래 분석을 수행한다.

### 분석 항목 및 방법

| 분석 항목 | 방법 | 구현 위치 |
|----------|------|----------|
| 변동성 수준별 `aggressiveness` 분포 | 산점도 (x=volatility zscore, y=aggressiveness) | `notebooks/05_behavior_analysis.ipynb` |
| 가격 수준별 `profit_target` 분포 | 산점도 (x=log_price zscore, y=profit_target) | `notebooks/05_behavior_analysis.ipynb` |
| 사이클 수익률 vs 진입 시점 변동성 | 박스플롯 (변동성 분위별 cycle_pnl_pct) | `notebooks/05_behavior_analysis.ipynb` |
| 레짐별 행동 전략 요약 | 변동성 상/중/하 3구간으로 나눠 aggressiveness·profit_target 통계 비교 | `src/evaluation/behavior.py` |

### 데이터 수집 방법

```python
# scripts/analyze_behavior.py 에서 수행
records = []
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    records.append({
        "step":        env.current_step,
        "log_price":   obs[0],
        "divergence":  obs[1],
        "hvr":         obs[2],   # holdings_value_ratio
        "cash_ratio":  obs[3],
        "volatility":  obs[4],
        "aggressiveness": action[0],
        "profit_target":  action[1],
    })
    obs, reward, done, _, info = env.step(action)
    if done:
        break

df_behavior = pd.DataFrame(records)
```

### 기대 결과 패턴 (가설)

- 변동성 高 → aggressiveness 高 (자주 거래, 좁은 간격)
- 변동성 低 → aggressiveness 低 (대기 또는 넓은 간격)
- divergence 음수(수익) → profit_target 高 (높은 익절 목표)
- cash_ratio 低 → aggressiveness 低 (추가 매수 여력 없음)

이 패턴이 실제로 학습되는지가 부 연구 질문의 핵심 검증 포인트다.

---

## 5. 미결 사항 및 구현 주의점

- [ ] `_process_fills()`: 같은 봉에서 buy + sell 동시 체결 시 처리 순서 확정 (sell 우선 원칙)
- [x] `order_size_fraction` → `n_splits` 방식으로 대체 (2026-04-10):
      매수: `per_order_size = (cycle_start_cash / n_splits) / n_buy_orders` (고정 금액)
      매도: `holdings ≤ threshold_btc` → 전량 청산, 아니면 `holdings / n_sell_orders`
      `n_splits` 기본값 4, Val 셋 튜닝 예정
- [ ] 변수 간 상관관계 분석 후 state 구성 재검토 (학습 전 notebooks/02에서 수행)
- [ ] alpha 초기값 검증: Val 셋 랜덤 에이전트 실행 후 평균 cycle_pnl_pct 측정

---

## 6. 관련 파일

| 파일 | 역할 |
|------|------|
| `config/experiment_config.yaml` | 모든 수치 파라미터 |
| `src/env/trading_env.py` | MDP 구현체 |
| `src/data/preprocessor.py` | log_price, volatility rolling z-score 계산 |
| `notebooks/01_data_exploration.ipynb` | 원본 데이터 탐색 |
| `notebooks/02_indicator_analysis.ipynb` | state 변수 분포 및 상관관계 확인 |
| `RESEARCH_LOG.md` | 설계 의사결정 날짜별 기록 |

---

## 7. Phase 2~3 주요 변경 (본 문서 작성 후)

본 문서가 작성된 Phase 1 시점 이후의 누적 변경 사항. 상세는 `RESEARCH_LOG.md`.

### State

- **5D → 7D (exp017)**: `trend_short` (72h pct_change), `trend_long` (720h pct_change) 추가
- **7D → 9D (exp029)**: 사이클 시작 시 1회 결정 방식 도입 시 `idle_norm`, `n_splits_norm` 추가

### Action

- **고정 절대 갭 → ATR 비례 스케일링 (exp003 이후)**:
  ```python
  buy_hi_gap = atr_ratio * (A_b + B_b * aggressiveness)   # 등
  ```
  계수 A_b, B_b, C_b, D_b, A_s, B_s, C_s, D_s는 config에서 조정 (Optuna 최적화 대상)

- **sell_lo/sell_hi → sell_market/sell_cost (exp003 이후)**: 기준가 명시
  - `sell_market` = price 기준 (시장 모멘텀)
  - `sell_cost` = avg_price 기준 (원가 수익 보장)

### Reward

- **사이클 보너스 제거 (exp003 이후)**: `cycle_pnl_pct + alpha/cycle_hours` 항 삭제. step_reward만 사용.
- **fee 이중차감 버그 수정 (exp003 이후)**: `_execute_buy/_sell`에서 이미 fee가 cash에 반영 → 별도 차감 없음.
- **Phase 3 reward 변형 (exp032 예정)**: Symmetric (현재) / Asymmetric / DSR / Prospect-theoretic 4종 비교.

### 체결

- **next_low/next_high → 지정가 체결 (exp026 이후)**: favorable bias 제거. 시뮬레이션 정합성 확보.

### Order sizing

- **`order_size_fraction` → `n_splits` 균등 분할 (exp003 이후)**: 사이클 시작 시 예산 확정.
- **`threshold_btc` 기준**: `cycle_slot_size / price` (exp003 이후, 이전엔 sell_market/sell_cost 평균)
- **균등 분할 매도**: `holdings / n_splits` (exp003 이후, 이전엔 n_sell_orders)

자세한 history는 `RESEARCH_LOG.md` 참조.
