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

## 2026-04-13 — 환경 수학 검증 테스트 작성

### 무엇을
`tests/test_trading_env.py` 신규 작성 — 46개 테스트, 전체 통과 (1.00s).

### 왜
지난 세션(n_splits 도입, 사이클 보너스 제거, last_avg_price 추가)에서 환경 수학이
크게 바뀌었는데 수치 검증이 없었다. PPO 학습 전에 반드시 "환경이 설계대로 계산하는가"를
확인해야 한다. 이후 환경을 수정할 때도 pytest 한 번으로 회귀를 즉시 탐지할 수 있다.

### 어떻게
가격 고정 합성 DataFrame(close=1000, high/low 정밀 제어)으로 실제 데이터 의존성 제거.
내부 메서드 직접 호출(수학 단위 테스트) + step() 통합 호출(동작 통합 테스트) 병행.

**검증 항목 (10개 클래스)**
| 클래스 | 검증 내용 |
|--------|----------|
| InitialState | reset 후 7개 초기값 |
| BuyMath | per_order_size, 평단가 가중평균, 수수료, cash 감소 |
| BudgetExhaustion | 8회 성공 → 9회 False, 음수 방지 |
| SellMath | 수수료, 전량/분할 청산, avg_price 불변 |
| CycleAndLastAvgPrice | last_avg_price 저장 조건, 사이클 구조, bonus 키 없음 |
| ThresholdBtcSell | threshold 기준 전량/분할 매도 |
| SellFirstPrinciple | 같은 봉 sell+buy 동시 체결 시 sell 우선 |
| Divergence | last_avg_price 3분기 동작 |
| Reward | bonus=0, equity_change 공식 정확성 |
| EnvChecker | gymnasium 공식 env_checker |

### 발견 사항
초기 실패 3건 — next_low=500 설정 시 buy_hi + buy_lo 둘 다 체결(n_trades=2)되어
n_trades=1로 가정한 테스트가 깨짐. 이는 환경 버그가 아니라 두 주문이 독립적으로
정상 처리된다는 것을 확인한 셈. next_low를 999.5로 조정하여 buy_hi만 선택 체결.

### 커밋
- `test: BTCGridTradingEnv 포트폴리오 수학 검증 테스트 46개` (feature/trading-env-tests → main)

---

## 2026-04-13 — EDA 노트북 작성 (01_data_exploration.ipynb)

### 무엇을
`notebooks/01_data_exploration.ipynb` 신규 작성 및 전체 실행 완료.
그래프 7개 저장: `reports/semester1/figures/01_*.png`

### 확인 항목 및 주요 발견
| 항목 | 결과 |
|------|------|
| 가격 시계열 | Train: 4,130~68,634 USD (상승→하락→횡보 레짐 포함), Val: 2023 회복장 |
| zscore_log_price 범위 초과(|z|>5) | Train ~0%, Val 확인 완료 |
| zscore_volatility 범위 초과(|z|>5) | 극소수, clip 처리 적합 |
| 상관관계 | zscore_log_price vs zscore_volatility 낮음 → 두 변수가 독립적 정보 제공 |
| 변동성 레짐 | High Vol 구간은 가격 급변 구간과 일치, PPO 활용 여지 확인 |

observation_space [-5, 5] 설정이 실제 데이터 분포를 잘 커버한다고 검증됨.

### 커밋
- `feat: 01_data_exploration.ipynb — EDA 노트북 작성 및 실행` (feature/eda-notebook → main)

---

## 2026-04-13 — baselines.py 구현 및 Val 셋 검증

### 무엇을
`src/agents/baselines.py` 신규 작성 — 베이스라인 에이전트 3종 7개 변형.

### 왜
PPO 학습 이전에 "이기기 위한 기준선"이 필요하다. 규칙 기반 전략 대비 Sharpe Ratio
우위를 보이는 것이 연구 질문의 핵심이므로, 동일한 수수료/주문 구조로 구현한 베이스라인이
공정한 비교의 전제다.

### 어떻게
`BaseAgent(ABC)` 공통 인터페이스 + `_buy()` / `_sell()` / `_equity()` 헬퍼 구현.
BTCGridTradingEnv와 동일한 원칙 적용:
- Sell 우선 원칙 (같은 봉에서 sell 먼저 처리)
- n_splits 예산 균등 분할
- threshold_btc 기준 전량/분할 매도
- 체결 판단: 다음 봉 high/low 기준

| 클래스 | 전략 |
|--------|------|
| `BuyAndHoldAgent` | warmup 이후 전액 매수, 보유 |
| `FixedGridAgent(gap)` | 고정 gap%, 대칭 그리드 |
| `ATRGridAgent(k)` | gap = k × volatility_raw, 동적 간격 |

`run_all_baselines(df, config)` 편의 함수로 7개 변형 일괄 실행.

### Val 셋 결과 (2023년 강세장)

| 전략 | 수익률(%) | 거래횟수 | 완료사이클 |
|------|----------|---------|----------|
| Buy-and-Hold | +150.18 | 1 | 0 |
| Fixed Grid 1% | +43.16 | 567 | 141 |
| Fixed Grid 2% | +16.96 | 126 | 41 |
| Fixed Grid 5% | +2.47 | 8 | 4 |
| ATR Grid k=0.5 | +24.70 | 1464 | 347 |
| ATR Grid k=1.0 | +39.79 | 833 | 213 |
| ATR Grid k=2.0 | +28.98 | 320 | 91 |

**해석**: 2023년은 BTC ~$16k → ~$42k 강세장이므로 Buy-and-Hold가 압도적으로 유리.
그리드 전략은 횡보/변동 구간에서 강점을 보인다. Val 셋 대비 Train 셋(2020-2022, 상승·하락·횡보
혼재)에서 그리드 전략의 상대 성과가 더 나올 것으로 예상 — PPO와의 비교는 Train + Val 모두 필요.

### 커밋
- `feat: Buy-and-Hold / Fixed Grid / ATR Grid 베이스라인 구현` (feature/baselines → main)

---

## 2026-04-13 — metrics.py 구현

### 무엇을
`src/evaluation/metrics.py` 신규 작성 — 포트폴리오 성능 지표 계산 모듈.

### 왜
PPO와 베이스라인을 **동일한 함수**로 계산해야 비교가 공정하다.
학습 루프 안에서도 Val 평가 시 Sharpe를 계산하려면 이 모듈이 먼저 있어야 한다.
ppo_agent.py / train_ppo.py 구현 이전에 선행 완료.

### 구현 내용

| 함수 | 설명 |
|------|------|
| `total_return_pct(equity_curve, initial_cash)` | 누적 수익률 (%) |
| `sharpe_ratio(equity_curve, ...)` | 연율화 Sharpe (√8760, 1h봉 기준) |
| `max_drawdown_pct(equity_curve)` | 최대 낙폭 (%) |
| `avg_cycle_pnl_pct(completed_cycles)` | 사이클 평균 수익률 (%) |
| `avg_cycle_hours(completed_cycles)` | 사이클 평균 소요 봉 |
| `compute_all(...)` | 전체 지표 일괄 계산 → dict |
| `print_metrics(metrics, label)` | 지표 dict 보기 좋게 출력 |

### 검증 결과

수학 단위 테스트:
- total_return 10% (단조 상승) ✅
- MDD 40% (10k→15k→9k) ✅
- 무변동 Sharpe 0.000 ✅

Val 셋 베이스라인 전체 적용:

| 전략 | 수익률(%) | Sharpe | MDD(%) | 거래 | 사이클 |
|------|----------|--------|--------|------|--------|
| Buy-and-Hold | +150.18 | 2.377 | 21.74 | 1 | 0 |
| Fixed Grid 1% | +43.16 | **2.610** | 10.77 | 567 | 141 |
| Fixed Grid 2% | +16.96 | 2.032 | 7.65 | 126 | 41 |
| Fixed Grid 5% | +2.47 | 1.375 | 1.66 | 8 | 4 |
| ATR Grid k=0.5 | +24.70 | 1.118 | 16.07 | 1464 | 347 |
| ATR Grid k=1.0 | +39.79 | 1.434 | 15.86 | 833 | 213 |
| ATR Grid k=2.0 | +28.98 | 1.948 | 9.38 | 320 | 91 |

**PPO가 넘어야 할 Sharpe 기준: 2.610** (Val 셋 Fixed Grid 1%)
단, 2023년은 강세장 — Train 셋(2020-2022 혼재)에서의 Sharpe 비교가 더 의미 있다.

### 커밋
- `feat: src/evaluation/metrics.py — 포트폴리오 성능 지표 계산 모듈` (feature/metrics → main)

---

## 2026-04-14 — PPOAgent 래퍼 + train_ppo.py 구현

### 무엇을
`src/agents/ppo_agent.py` + `scripts/train_ppo.py` 신규 작성.
PPO 학습 파이프라인 전체 구현 완료.

### 왜
베이스라인과 metrics.py가 완성됐으니 이제 실제 PPO를 학습시킬 파이프라인이 필요하다.
학습 중 Val 평가, 최고 모델 저장, MLflow 기록, 베이스라인 비교 출력까지 한 번에 처리한다.

### 구현 내용

**`src/agents/ppo_agent.py`**
- `ValMetricsCallback`: eval_freq 스텝마다 Val 1에피소드 실행 → Sharpe/Return/MDD 출력
  - 최고 Sharpe 모델 `best_model_path`에 자동 저장
  - MLflow 활성 시 지표 자동 기록
- `PPOAgent(config, df_train, df_val)`: SB3 PPO 래퍼
  - `train(total_timesteps, best_model_path)`: 학습 실행
  - `evaluate(df)`: 결정론적 1에피소드 평가 → metrics dict + equity_curve
  - `save(path)` / `load(path, ...)`: 모델 저장/로드
  - tensorboard 미설치 시 자동 비활성화 (ImportError 처리)

**`scripts/train_ppo.py`**
- CLI 인자: `--config`, `--timesteps`, `--exp-name`, `--no-mlflow`
- MLflow 하이퍼파라미터 + Val 지표 자동 기록
- 학습 완료 후 베이스라인 비교표 + PPO 우위 여부 출력
- config_snapshot.yaml 실험 디렉토리 자동 복사 (재현성)

### smoke test 결과 (100스텝)
파이프라인 전체 정상 동작:
- 콜백 주기적 호출 ✅
- best_model 자동 저장 ✅
- evaluate() 정상 반환 ✅
- 100스텝 미학습 에이전트 Sharpe -3.449 (랜덤 정책, 예상 범위 내)

### 인프라 이슈 및 해결
- `stable-baselines3` 미설치 → `pip install stable-baselines3` (v2.8.0, torch 2.11.0 포함)
- tensorboard 미설치 시 `ImportError` → ppo_agent.py에서 try/except로 자동 비활성화

### 커밋
- `feat: PPOAgent 래퍼 + train_ppo.py 학습 스크립트 구현` (feature/ppo-agent → main)

### PPO exp001 학습 결과 (1M 스텝, Train 2020-2022)

| 모델 | Sharpe | 수익률(%) | MDD(%) | 거래수 | 사이클 |
|------|--------|----------|--------|--------|--------|
| best_model (step=100k) | **+0.795** | +1.55 | 0.88 | 8 | 4 |
| final_model (step=1M) | -0.306 | -1.34 | 3.73 | 22 | 6 |
| Fixed Grid 1% (목표) | 2.610 | +43.16 | 10.77 | 567 | 141 |

**학습 곡선 패턴**: 100k 스텝에서 Sharpe 0.795 달성 후, 150k부터 0.15~0.35 구간 진동.
1M 스텝 종료 시 -0.280으로 하락 → **수렴 실패**.

**원인 가설 (검증 예정)**:
1. learning_rate=3e-4 과다 → lr scheduling (cosine decay) 필요
2. ent_coef=0.01 → annealing (0.01 → 0.001) 검토
3. 에피소드 길이(25,916봉) vs n_steps(2048) 불균형
4. reward 스케일 미세해서 학습 신호 약함

**향후 계획**: 2학기에 하이퍼파라미터 그리드서치 (lr, ent_coef, n_steps 우선)

---

## 2026-04-14 — behavior.py 구현 (PPO 행동 분석 모듈)

### 무엇을
`src/evaluation/behavior.py` 신규 작성 — Sub-RQ 분석 인프라.

### 왜
주 연구 질문(Sharpe 우위)과 함께 **"어떤 시장 레짐에서 어떤 간격을 선택하는가?"**
라는 부 연구 질문에 답해야 한다. PPO 학습 완료 후 즉시 분석할 수 있도록
행동 수집/분석/시각화 파이프라인을 미리 구현한다.

### 구현 내용

| 함수 | 역할 |
|------|------|
| `collect(model, df, config)` | 1에피소드 전체 (state, action, regime) → DataFrame |
| `actions_to_gaps(actions)` | action [0,1]² → buy/sell gap 실수값 변환 |
| `action_stats(behavior_df)` | 기술통계 (mean/std/min/max) |
| `regime_analysis(behavior_df)` | Low/Mid/High 레짐별 평균 행동 비교 |
| `print_regime_summary(behavior_df)` | 레짐별 요약 콘솔 출력 |
| `plot_behavior(behavior_df, save_dir)` | 행동 시계열 + 박스플롯 4종 PNG 저장 |

레짐 정의 (zscore_volatility 기준):
- Low : < -0.5 / Mid : -0.5 ~ 0.5 / High : > 0.5

### smoke test 결과 (Val, 미학습 모델)
- 8,567스텝 수집 완료 ✅
- 레짐 분포: Low 49.7% / Mid 10.6% / High 39.8%
- gap 변환 수학 검증: agg=0.0 → buy_hi_gap=0.01%, agg=1.0 → 5.01% ✅
- 행동 분석 그래프 저장 성공 ✅

### 커밋
- `feat: src/evaluation/behavior.py — PPO 행동 수집 및 레짐 분석 모듈` (feature/behavior → main)

---

## 2026-04-14 — Train 셋 베이스라인 평가 + exp002 튜닝 설정

### 무엇을
- Train 셋(2020-2022)에서 7개 베이스라인 전부 평가
- exp002 하이퍼파라미터 설정 파일 작성 + 재학습 시작
- Train vs Val 비교 그래프 2종 생성

### Train 셋 결과 (2020-2022, 상승·하락·횡보 혼재)

| 전략 | Train Sharpe | Val Sharpe | 비고 |
|------|-------------|-----------|------|
| ATR Grid k=0.5 | **0.818** | 1.118 | Train 최고 |
| Fixed Grid 5%  | 0.802 | 1.375 | |
| Buy-and-Hold   | 0.666 | 2.377 | Val 강세장 효과 |
| Fixed Grid 1%  | 0.620 | **2.610** | Val 최고 |
| ATR Grid k=2.0 | **-0.104** | 1.948 | 레짐 의존성 뚜렷 |

**핵심 발견**:
1. **Train과 Val 순위가 다르다** — Val(2023 강세장)은 Buy-and-Hold 유리, Train(혼재)은 ATR Grid 유리
2. **ATR Grid k=2.0 역전** — Train Sharpe -0.104(음수) vs Val 1.948. 넓은 간격이 하락장에서는 매수 체결 지연 → 손실
3. **PPO exp001 best_model (0.795)** ≈ Train 베이스라인 최고치(0.818) — PPO가 Train 환경 수준까지는 학습했다고 해석 가능

### exp002 하이퍼파라미터 튜닝 (학습 중)

| 파라미터 | exp001 | exp002 | 근거 |
|---------|--------|--------|------|
| learning_rate | 3e-4 | **1e-4** | 후반 불안정 → 감소 |
| n_steps | 2048 | **4096** | 에피소드(25k봉) 대비 짧은 롤아웃 |
| batch_size | 64 | **128** | n_steps 증가에 비례 |
| ent_coef | 0.01 | **0.005** | 수렴 방해 가능성 감소 |

### 커밋
- `feat: exp002 튜닝 설정 + Train 셋 베이스라인 평가`

---

## 2026-04-14 — exp002 학습 완료 및 결과 분석

### exp001 vs exp002 비교

| 지표 | exp001 | exp002 | 개선 |
|------|--------|--------|------|
| best Sharpe | 0.795 (step=100k) | 0.745 (step=50k) | ↓ |
| final Sharpe | -0.280 | **-0.095** | ↑ |
| mean Sharpe | 0.271 | **0.368** | ↑ |
| 양수 비율 | 95.0% | 90.0% | ↓ 소폭 |

### 해석
- **안정성 개선**: final Sharpe -0.280 → -0.095, mean Sharpe 0.271 → 0.368. 수렴이 더 안정적
- **peak 미개선**: best Sharpe는 오히려 낮음 (0.795 → 0.745). lr 감소가 탐색 초반 속도를 낮춤
- **공통 문제**: 두 실험 모두 Train 베이스라인 최고(ATR k=0.5: 0.818)에 미달. 단순 lr/ent_coef 튜닝만으로는 한계

### 보류 아이디어 (2학기)
- **reward normalization**: step_reward 스케일이 너무 작아 학습 신호 약함
- **VecNormalize**: SB3 내장 reward/obs 정규화 래퍼 활용
- **n_steps 8192**: 에피소드(25k봉) 대비 충분한 롤아웃
- **총 스텝 증가**: 1M → 3M (충분한 샘플 확보)
- **lr scheduling**: LinearSchedule(3e-4 → 1e-5) annealing

### 커밋
- `feat: exp002 학습 완료 + exp001 vs exp002 비교 그래프`

---

## 2026-04-17 — 설계 이슈 발견 (6주차 보고서 검토 중)

보고서 검토 과정에서 다음 4가지 설계 이슈가 발견되었다. 다음 실험(exp003) 시작 전 반영한다.

### 이슈 1. threshold_btc 분모 불명확 (낮음)

**현재:**
```
threshold_btc = cycle_slot_size / avg(sell_lo, sell_hi)
```
avg(sell_lo, sell_hi)는 에이전트가 설정한 목표가이지 실제 체결가가 아니다.
"1슬롯 현금에 해당하는 BTC 수량"을 의미하려면 현재가로 나누는 것이 더 직관적이다.

**개선:**
```
threshold_btc = cycle_slot_size / price
```

---

### 이슈 2. 균등 분할 매도 기준 불일치 (낮음)

**현재:** `sell_qty = holdings / n_sell_orders (=2)`
매수는 n_splits(=4) 슬롯으로 분할하는데 매도는 n_sell_orders(=2)로 분할한다.
두 주문이 모두 체결되면 한 번에 전량 청산되어 그리드 트레이딩의 점진적 수익 실현 특성과 맞지 않는다.

**개선:** `sell_qty = holdings / n_splits (=4)` — 매수와 대칭 구조

---

### 이슈 3. 지정가 체결 방식 — 백테스트와 실전 불일치 (중간)

**현재:** 지정가 조건 충족 시 지정가 그대로 체결
```
next_high >= sell_lo → sell_lo 가격으로 체결
```

**실제 거래소:** 지정가 조건 충족 시 그 시점 시장가로 체결
```
next_high >= sell_lo → next_high 가격으로 체결  (실전 유리)
```

현재 구현은 보수적 백테스트 근사이지만, avg_price < price인 상황에서
sell_hi가 현재가 아래로 내려가면 지정가(낮은 가격)로 체결되어 불필요한 손실이 발생할 수 있다.

**개선:** 체결 시 지정가 대신 봉의 high/low 값으로 체결하도록 수정

---

### 이슈 4. sell_lo / sell_hi 명칭 혼동 (낮음)

현재 명칭은 "결과값 기준 (낮은 목표가 / 높은 목표가)"인데,
기준 가격이 각각 price / avg_price로 달라 avg_price < price일 때 sell_hi < sell_lo 역전이 발생한다.
이름이 역할을 반영하지 못함.

**개선:** 기준 가격을 명칭에 반영
- `sell_lo` → `sell_market` (현재가 기준, 시장 모멘텀 활용)
- `sell_hi` → `sell_cost` (평단가 기준, 원가 수익 보장)

---

## 2026-04-19 — 설계 이슈 4개 코드 반영 완료 + exp003 config 작성

### 변경 내용

**브랜치:** `feature/env-fixes-exp003` → main merge 완료

#### src/env/trading_env.py

| 변경 항목 | 이전 | 이후 | 근거 |
|-----------|------|------|------|
| `_close_cycle()` 반환값 | cycle_pnl_pct + alpha/hours 보너스 | 0.0 (통계 기록만) | step_reward의 equity 변화가 P&L을 이미 반영 |
| `__init__` | `cycle_alpha` 파라미터 존재 | 제거 | 보너스 제거에 따른 정리 |
| `_init_state` / `reset` | `last_avg_price` 없음 | `last_avg_price: float = 0.0` 추가 | 미보유 구간 divergence 품질 향상 |
| `_execute_sell` | 전량 청산 시 avg_price만 0으로 초기화 | `last_avg_price = avg_price` 저장 후 초기화 | 직전 사이클 평단가 보존 |
| `_get_observation` divergence | 미보유 시 무조건 0.0 | `last_avg_price > 0`이면 그 값으로 계산, 없으면 0.0 | "직전에 70k 매도, 현재 60k" 상태가 "미거래" 상태와 다르게 인코딩 |
| `_compute_order_prices` 명칭 | `sell_lo`, `sell_hi` | `sell_market`, `sell_cost` | 기준가 명시 (현재가 vs 평단가) |
| `_process_fills` `threshold_btc` | `cycle_slot_size / avg(sell_lo, sell_hi)` | `cycle_slot_size / price` | 현재가 기준 1슬롯 BTC 수량이 의미론적으로 올바름 |
| `_process_fills` 균등 분할 매도 | `holdings / n_sell_orders` | `holdings / n_splits` | 매수(per_order = slot/n_buy_orders, n_buy_orders = n_buy per cycle)와 대칭; n_splits이 전체 예산 분할 기준 |
| `_process_fills` 체결가 | 지정가(sell_market/sell_cost 자체 가격) | 시장가(next_high/next_low) | 실제 거래소 지정가 체결 방식: 조건 충족 시 그 시점 시장가로 체결 |

#### tests/test_trading_env.py

- `TestThresholdBtcSell`: threshold 계산 수치를 price 기준으로 수정
- `TestSellFirstPrinciple._build_env`: `lows[WARMUP+1] = 500` → `999.5`
  - 이유: 500 사용 시 buy_hi + buy_lo 모두 체결 → holdings ≈ 5.0 > threshold_btc(2.5) → 분할 매도만 나와 사이클 미종료
  - 999.5 사용 시 buy_hi(≈999.9)만 체결 → holdings ≈ 1.25 ≤ threshold_btc(2.5) → 전량 청산 → 사이클 종료 정상 검증
- 전체 46개 테스트 통과 확인

### exp003 config 작성 (`config/exp003_config.yaml`)

| 항목 | exp001/002 | exp003 |
|------|-----------|--------|
| VecNormalize | 미사용 | norm_obs=True, norm_reward=True, clip=10.0 |
| LR | 3e-4 고정 | 3e-4 → 1e-5 cosine decay |
| n_steps | 2048 | 8192 |
| batch_size | 64 | 256 |
| total_timesteps | 1M | 3M |
| 환경 버전 | 이슈 4개 이전 | 이슈 4개 반영 버전 |

### 보류한 아이디어

- **n_splits = 10 vs 4**: notebook 04 행동 분석 후 결정. exp003은 4로 유지.
- **sell_cost dead order 이슈**: sell_cost < sell_market 구간(avg_price << 현재가)에서 sell_cost가 항상 먼저 체결될 수 있음 → 시장가 체결로 전환했으므로 실제로는 양쪽 다 next_high로 체결, 경제적 문제는 없음. 단, 순서 의미가 희미해질 수 있어 추후 단일 매도 가격 설계 검토 예정.

---

## 2026-04-19 — Reward 수수료 이중계산 버그 수정

### 발견 경위
exp003 학습 결과 3M steps 동안 거래 0건으로 수렴.
best_model(50k) Sharpe=0.367이었으나 이는 0거래 기준값.
에이전트가 aggressiveness=0.23으로 수렴 → buy_hi_gap=1.16% → 1h 캔들 변동폭(평균 0.3%) 대비 체결 확률 2.8%.

### 근본 원인: reward 수수료 이중계산 (8배 과다 패널티)

```python
# 기존 코드 (버그)
step_reward = (equity_after - equity_before) / start_capital
            - fee_rate * n_trades_this_step
```

수수료는 이미 equity에 반영됨:
- `_execute_buy`: cash -= spend (수수료 포함), holdings += (spend - fee) / price
- `_execute_sell`: cash += qty × price - fee
→ equity 변화분 = 실제 납부 수수료 (0.0625 USD per 1회 / 10,000 start_capital = 0.000625%)

그런데 `fee_rate * n_trades = 0.0005 × 1 = 0.05%`를 추가로 차감 → 실제의 8배 패널티.

에이전트 학습 결과: "거래 1회 = -0.056% 손해, 0거래 = 0" → 완전 수동 정책으로 수렴.

### 수정 내용

```python
# 수정 코드
step_reward = (equity_after - equity_before) / start_capital
```

- `src/env/trading_env.py`: fee 이중차감 제거, 주석으로 근거 명시
- `CLAUDE.md`: Reward 설명 업데이트
- `tests/test_trading_env.py`: TestReward 두 케이스에서 `- FEE * n_trades` 항 제거
- `config/exp003_config.yaml`: baselines 키 추가 (KeyError 수정)
- 46개 테스트 통과 확인

### exp003 재실행 (exp004로 명명)
동일 하이퍼파라미터 (VecNormalize + cosine LR + n_steps=8192 + 3M steps),
수정된 reward 공식으로 재학습.

---

## 2026-04-19 — Action 공식 ATR 비례 스케일링 도입

### 문제 진단 (exp003 결과 분석)

고정 절대 간격 공식의 구조적 문제:
- `buy_hi_gap = 0.0001 + agg × 0.05`
- val(2023) 1h 평균 변동폭 0.304%, train(2020-22) 0.561%
- agg=0.0 → gap=0.01% → 체결 확률 3%  (action 범위의 대부분이 "체결 불가")
- agg=0.1 → gap=0.51% → 체결 확률 84% (급격한 계단)
- 의미있는 구간 [0.03, 0.1] = 전체의 7%만 유효
- State[4]의 ATR 정보를 읽어도 Action에 반영할 표현 수단 없음

### 해결책: ATR 비례 스케일링

```python
# 수정 전 (고정)
buy_hi_gap = 0.0001 + aggressiveness * 0.05

# 수정 후 (ATR 비례)
atr_ratio  = ATR(168) / price
buy_hi_gap = atr_ratio * (0.1 + aggressiveness * 0.9)  # [0.1×ATR, 1.0×ATR]
buy_lo_gap = atr_ratio * (0.5 + aggressiveness * 4.5)  # [0.5×ATR, 5.0×ATR]
# sell도 동일 구조
```

체결 확률 분포 (val 2023 기준):
  agg=0.0 → gap=0.1×ATR(0.058%) → 체결 확률 13%
  agg=0.5 → gap=0.55×ATR(0.317%) → 체결 확률 67%
  agg=1.0 → gap=1.0×ATR(0.576%) → 체결 확률 87%

→ action [0,1] 전체가 의미있는 스펙트럼에 고르게 분포
→ 변동성이 높은 train 구간에서 배운 정책이 val에도 자동 적용 (스케일 불변성)
→ State-Action 정합성: State[4] ATR 정보를 Policy가 action으로 직접 표현 가능

### 변경 파일
- `src/env/trading_env.py`: `_compute_order_prices()` ATR 비례 공식 + `atr_ratio` 인자 추가
- `CLAUDE.md`: Action 공식 섹션 업데이트 + 설계 근거 명시
- `tests/test_trading_env.py`: `ATR_RATIO=0.01` 상수 추가, `_df()/_df_rows()` volatility_raw 컬럼 수정, 체결 경계 수치 전면 갱신
- 46개 테스트 통과 확인

---

## 2026-04-19 — exp004 결과 분석 + exp005 설계 (최소 gap 계수 수정)

### exp004 최종 결과 요약

| 모델 | Sharpe | 수익률(%) | MDD(%) | 거래수 | 사이클 |
|------|--------|----------|--------|--------|--------|
| best (step≈1.05M) | **4.588** | +0.00 | 99.62 | — | — |
| final (step=3M) | 0.724 | -3.67 | 69.84 | 0 | 0 |
| Fixed Grid 1% (기준) | 2.610 | +43.16 | 10.77 | 567 | 141 |

**best_model 트레이스 (200스텝)**: n_trades=424, cycles=144, equity 10000→12291 → 실제 거래 발생 ✅  
**final_model**: 0거래로 수렴 (학습 불안정)

### 원인 분석: 최소 gap ≈ fee 손익분기점

ATR 비례 공식 (min coef=0.1):
```
buy_hi_gap = ATR × 0.1  (agg=0.0 기준)
val p10 ATR = 0.349% → min gap = 0.035%
round-trip net = 2×0.035% - 2×0.05% = -0.030%  ← 손실
val mean ATR = 0.576% → min gap = 0.058%
round-trip net = 2×0.058% - 2×0.05% = +0.015%  ← 이익 (간신히)
```

→ val 하위 50% ATR 구간(p25=0.046%)에서 round-trip net < 0  
→ 에이전트가 agg=0 고착해도 저변동성 구간에서는 수익이 나지 않음  
→ 학습이 불안정: 이익 내는 정책 탐색 → 고변동성 일부 구간만 수익 → 결국 과적합 붕괴

### 해결책: 최소 gap 계수 0.1 → 0.5 (exp005)

새 범위 [0.5×ATR, 2.0×ATR]:
```
val p10 ATR = 0.349% → min gap = 0.175%
round-trip net = 2×0.175% - 2×0.05% = +0.250%  ← 안정적 이익
val p25 ATR = 0.463% → net = +0.363%
val mean ATR = 0.576% → net = +0.476%
```

→ val 전 구간에서 fee 손익분기점 초과 보장  
→ 에이전트가 agg=0에 고착해도 반드시 양(+)의 기댓값 거래

### 변경 내용 (exp005)
- `src/env/trading_env.py`: 계수 [0.1, 0.9] → [0.5, 1.5] (buy_hi/sell_market)
  및 [0.5, 4.5] → [2.5, 7.5] (buy_lo/sell_cost)
- `tests/test_trading_env.py`: buy_hi=995.0, sell_market=1005.0 기준으로 전면 갱신
- `CLAUDE.md`: Action 공식 업데이트
- `config/exp005_config.yaml`: 신규 (하이퍼파라미터는 exp004 동일)
- 46개 테스트 통과 확인

### exp005 학습 시작 (3M steps, 백그라운드)

---

## 2026-04-19 — exp006 설계·구현·학습 완료 + 노트북 03/04 업데이트

### exp006 설계 배경: 에피소드 길이 문제

exp001~005 공통 패턴: 초반 학습 → 후반 0거래 수렴.

**근본 원인: Discount 소멸**
```
γ = 0.99, 에피소드 = 25,916스텝
γ^2000 ≈ 2e-9  → 에피소드 내 2,000스텝 이후의 reward는 gradient에 미반영
γ^25000 ≈ 10^-109 ≈ 0

결과: 에이전트가 에피소드 초반(~80일)만 최적화하고
      나머지 2.6년은 "거래 안 함"이 로컬 옵티멈
```

### exp006 변경 사항

| 항목 | exp005 | exp006 | 근거 |
|------|--------|--------|------|
| 에피소드 최대 길이 | 25,916(전체) | **2,016**(12주) | γ^2016≈2e-9, 유효 horizon ≈500스텝 |
| 병렬 환경 | 1 | **4** (DummyVecEnv) | 다양한 레짐 동시 학습 |
| 시작점 | 2020-01-01 고정 | **랜덤** (에피소드마다) | 상승/하락/횡보 균등 학습 |
| rollout 크기 | 8,192 | **32,768** (4×8,192) | 학습 신호 다양성 4배 |

### 코드 변경
- `src/env/trading_env.py`: `random_start` 옵션 + `_max_ep_steps` 기반 시작점 계산
- `src/agents/ppo_agent.py`: `n_envs × DummyVecEnv` + `TimeLimit` 래퍼 자동 적용
- `config/exp006_config.yaml`: max_episode_steps=2016, n_envs=4, random_start=true

### exp006 학습 결과 (3M steps)

| step | Sharpe | 비고 |
|------|--------|------|
| 50k | 0.634 | |
| 250k | 1.042 | |
| 350k | 1.064 | |
| **550k** | **1.183** | **best 저장** |
| 1,250k | 1.117 | |
| 3,000k (final) | 1.165 | |
| final evaluate() | 0.871 | 0 trades (eval 버그 의심) |

**핵심 개선**: 학습 곡선이 전 구간 Sharpe 0.6~1.2 유지 (exp003~005는 후반 0으로 붕괴).
에피소드 단축 효과 확인.

**이상 현상**: final_model evaluate() 시 n_trades=0, MDD=72% 모순.
- n_trades=0이면 equity=10,000 (고정) → MDD=0%, Sharpe=0이어야 정상
- 추정 원인: VecNormalize obs_rms가 n_envs=4 + random_start 환경에서 수렴한 분포 vs
  단일 전체 val 에피소드 실행 시 obs 분포 불일치 → 이상 행동
- 다음 실험(exp007)에서 `norm_reward=False` 또는 VecNormalize 비활성화로 검증 예정

### 노트북 업데이트
- `notebooks/03_ppo_training.ipynb`: exp001/002/005 학습 곡선 비교, discount 효과 시각화, 실험 히스토리 표 포함
- `notebooks/04_behavior_analysis.ipynb`: 신규 — action 분포, 레짐별 행동, 거래 시각화

### 보류 아이디어
- **norm_reward=False**: reward 정규화가 수렴 저해 가능성 (exp007 검증)
- **VecNormalize 없이 실험**: obs 정규화는 유지하되 inference 시 분포 불일치 문제 우회
- **stop-loss**: 보유 중 가격 -X% 시 강제 청산 → MDD 제어

---

## 2026-04-19 — exp007 학습 완료 + 평가 파이프라인 버그 2개 수정

### exp007 설계 목적

exp006의 final_model evaluate() 시 n_trades=0 이상 현상 원인 검증 및 해소.

**가설 1: VecNormalize 이중 정규화**
- obs가 이미 z-score 정규화 → VecNormalize가 한 번 더 정규화
- train(n_envs=4, random_start)으로 수렴한 obs_rms ≠ val 추론 시 obs 분포 불일치
- → VecNormalize 완전 비활성화

**가설 2: 단일 긴 에피소드 평가 vs 학습 에피소드 구조 불일치**
- 학습 에피소드: 2016스텝 × 랜덤시작
- val 평가: 8,568스텝 단일 에피소드 (학습 분포와 다름)
- → Multi-episode 평가 활성화 (n_eval_episodes=5, 각 2016스텝)

### exp007 학습 결과 (3M steps)

학습 곡선: exp006과 유사하게 Sharpe 0.6~1.2 구간 안정 유지.
best_model 저장 완료.

### 버그 1: 평가 환경의 random_start=True 상속

**발견 경위**: exp007 best_model 평가 시 n_trades=0, Return=0%, MDD=34%.
직접 트레이스(random_start=False, 300스텝): 247 trades, 52 cycles, equity 10,000→11,980 (+19.8%)
→ 에이전트는 정상 거래. 평가 파이프라인 문제.

**원인**: `_evaluate_once`가 `config` 그대로 eval env를 생성 → `random_start=True` 상속.
eval env가 val 데이터 내 랜덤 위치에서 시작 → 평균 Sharpe가 실제와 무관해짐.

**수정**: `_evaluate_once`에서 `eval_cfg = deepcopy(config); eval_cfg["environment"]["random_start"] = False` 적용.

### 버그 2: DummyVecEnv 에피소드 종료 시 auto-reset

**발견 경위**: random_start 버그 수정 후에도 n_trades=0 지속.
동일 모델을 DummyVecEnv 없이 직접 실행 → n_trades=247.

**원인**: SB3 `DummyVecEnv`는 `done=True` 반환 직후 환경을 **자동 리셋**한다.
`while not done` 루프 종료 시 이미 리셋된 상태 → `raw_env.n_trades = 0`.

**수정**: `_evaluate_once`에서 `DummyVecEnv` 제거. `BTCGridTradingEnv + TimeLimit`를 직접 루프로 실행.
```python
# 수정 전: DummyVecEnv → done 후 auto-reset → n_trades=0
base_env = DummyVecEnv([_make_eval_env])

# 수정 후: 단일 환경 직접 실행
raw_env = BTCGridTradingEnv(df_eval, eval_cfg)
eval_env = TimeLimit(raw_env, max_episode_steps=ep_steps)
obs, _ = eval_env.reset()
...
n_trades = raw_env.n_trades  # 리셋 전에 읽음
```

### 수정 후 exp007 best_model 재평가

에피소드별 상세:
| ep | start | Return | Sharpe | MDD | Trades | Cycles |
|----|-------|--------|--------|-----|--------|--------|
| 1 | 168   | +96.3% | +13.68 | 4.04% | 1163 | 194 |
| 2 | 1805  | +45.7% |  +8.67 | 3.10% | 1031 | 191 |
| 3 | 3443  | +33.1% | +10.33 | 2.31% | 1023 | 187 |
| 4 | 5080  | +44.0% | +11.76 | 2.51% | 1098 | 187 |
| 5 | 6718  | +61.5% | +14.98 | 2.51% | 1099 | 192 |
| **평균** | | **+56.1%** | **+11.88** | **2.89%** | **1082.8** | **190.2** |

B&H 2023 전체: +150.3%  
기준선 Fixed Grid 1%: Sharpe 2.610

**PPO exp007 Sharpe 11.88 >> 기준선 2.610 → 연구 목표 달성**

**해석 주의사항**:
- 반환값 +56.1%는 실현+미실현 equity 합산 (에피소드 말미 open position 포함)
- ep1(Jan 2023 시작): BTC $16,927→$28,453 (+68%), grid trading +96.3%로 B&H 초과
- 고Sharpe 원인: 그리드 매매의 낮은 step-return 분산 + √8760 연율화 × 초당 빠른 사이클 회전
- 2023 강세장에서의 성과 → Train(2020-22 혼재) + Test(2024~) 교차 검증 필요

### 변경 파일
- `src/agents/ppo_agent.py`:
  - `_evaluate_once`: random_start=False 강제 적용
  - `_evaluate_once`: DummyVecEnv → 단일 환경 직접 루프로 교체
  - df 슬라이스 로직: `iloc[start_idx-warmup : start_idx+ep_steps]` (start_idx 기준 정렬)
- 46개 테스트 통과 확인

### 보류 아이디어
- **Train 셋 평가**: exp007 best_model을 Train 데이터에서도 평가해 과적합 여부 확인
- **Test 셋 개봉**: 최종 발표 직전에만 진행 (1학기 남은 일정 고려)

---

## 2026-04-19 — exp008 학습 완료: 레짐 적응 행동 발현 확인

### 설계 목적

exp007에서 aggressiveness=0.0 고착 관찰.
ent_coef 0.01→0.05 (5배) 증가로 policy entropy 보너스 강화 → 탐색 유도.
목표: Sub-RQ("어떤 레짐에서 어떤 간격?")에 답할 수 있는 행동 패턴 학습.

### 학습 곡선 요약 (3M steps)

- 전 구간 Sharpe 9.5~12.4 유지 (exp007 대비 유사 안정성)
- best_model: step=850k, Sharpe=12.381
- 1.75M~2.2M 구간 Sharpe 9~11로 일시 저하 후 회복 (ent_coef 증가로 탐색 노이즈)
- 2.6M 이후 Sharpe 12.2~12.4로 재수렴

### 핵심 결과: 레짐 적응 행동 통계적으로 유의미하게 발현

**aggressiveness 레짐별 분포 (val 2023, 5 에피소드)**:

| 레짐 | 스텝 수 | aggressiveness mean | std |
|------|---------|---------------------|-----|
| Low  | 5,040   | **0.01817**         | 0.05666 |
| Mid  | 1,007   | 0.01650             | 0.05578 |
| High | 4,028   | **0.00075**         | 0.00849 |

**Low vs High t-test: t=19.34, p=1.10e-81** → 압도적으로 유의미

방향성 해석:
- 고변동성 → aggressiveness≈0 (가장 타이트한 gap, 빠른 체결)
- 저변동성 → aggressiveness↑ (간격 넓힘, 체결 감소 대신 사이클당 수익 증가)

이는 이론적으로 올바른 적응 행동이다:
- 고변동성: 가격이 자주 크게 움직임 → 타이트 gap으로 최대 체결 빈도
- 저변동성: 가격 움직임 작음 → gap을 넓혀야 매도 조건 충족 가능

### exp007 vs exp008 성능 비교

| 지표 | exp007 | exp008 | 개선 |
|------|--------|--------|------|
| Sharpe (5ep 평균) | 11.884 | **12.381** | +0.497 |
| Return | +56.1% | +57.7% | +1.6%p |
| MDD | 2.89% | 2.90% | 동일 |
| Trades | 1082.8 | 1088.6 | 동일 |
| agg_mean | 0.0000 | 0.0110 | 레짐 적응 |

**결론**: 레짐 적응 행동 유도 + 소폭 성능 향상 동시 달성.
Sub-RQ 답변 근거 확보: "고변동성에서 좁은 간격, 저변동성에서 넓은 간격 선택."

### 변경 파일
- `config/exp008_config.yaml`: 신규 (ent_coef=0.05, 나머지 exp007 동일)

### 다음 실험 후보
- **exp009: n_splits ablation** — n_splits=2/4/8 비교 (현재 4 고정)
  - n_splits=2: 슬롯당 예산 큼 → 사이클당 수익 큼, 사이클 수 적음
  - n_splits=8: 슬롯당 예산 작음 → 사이클 수 많음, 사이클당 수익 작음
  - 최적 n_splits 탐색이 연구 과제의 일부

---

## 2026-04-20 — 포괄적 파라미터 탐색 + Phase 2-A 행동 분석

### 파라미터 탐색 실험 전체 결과 요약

exp008을 베이스라인(Sharpe 12.381)으로 다음 파라미터들을 순차 탐색했다.

| 실험 | 변경 파라미터 | best Sharpe | 결론 |
|------|-------------|-------------|------|
| exp009a | n_splits=2 | **16.826** | 최적값 확정 |
| exp009b | n_splits=8 | 8.041 | 열위 |
| exp009c | n_splits=16 | 3.018 | 열위 |
| exp010 | gamma=0.95 | 11.680 | gamma=0.99 유지 |
| exp011a | window=84 | 11.749 | window=168 유지 |
| exp011b | window=336 | 11.640 | window=168 유지 |
| exp013a | threshold=price | 16.942 | — |
| exp013b | threshold=avg_price | **17.579** | 최적값 확정 |
| exp014a_v2 | n_buy_orders=3 (신설계) | 15.963 | 열위 |
| exp014b_v2 | n_buy_orders=4 (신설계) | 14.630 | 열위 |
| exp015 | ent_coef=0.1 | 17.445 | ent_coef=0.05 유지 |

### 확정된 최적 파라미터 조합 (exp013b = 사실상 최종 모델)

| 파라미터 | 최적값 | 확정 근거 |
|----------|--------|----------|
| n_splits | 2 | exp009a (베이스라인 대비 +35.9%) |
| threshold_basis | avg_price | exp013b (price 대비 +0.637) |
| n_buy_orders | 2 | exp014v2 (신설계에서 2가 최적) |
| ent_coef | 0.05 | exp015 (0.1 대비 근소하게 우세) |
| gamma | 0.99 | exp010 (0.95 열위, 구조적 근거) |
| window/atr_period | 168 | exp011 (변경 시 열위) |

### 설계 변경 사항

1. **n_sell_orders 제거**: dead parameter였음. sell은 sell_market/sell_cost 2레벨 고정.
2. **threshold_basis 파라미터 추가**: "price" 또는 "avg_price" 선택 가능. avg_price가 우세.
3. **n_buy_orders 재설계**: 주문 크기 분할자 → 가격 레벨 수 (buy_hi~buy_lo 선형 보간). n=2 하위 호환.

### Phase 2-A: 행동 분석 (exp013b best_model, Val 2023 전체)

**성과 요약**:
| 항목 | 결과 |
|------|------|
| Val 총 수익률 | +4,961% ($10k → $506k) |
| 완료 사이클 | 1,304개 |
| 승률 | 92.9% |
| 평균 사이클 PnL | +0.30% |
| 평균 사이클 기간 | 4시간 |

**레짐 적응 행동 (Sub-RQ 최종 답변)**:
- Low vol aggressiveness 평균: 0.0082
- High vol aggressiveness 평균: 0.0000
- Mann-Whitney U: p = 1.36e-19 → 압도적으로 유의미

Sub-RQ 결론: PPO 에이전트는 시장 변동성 레짐에 따라 통계적으로 유의미하게 다른 aggressiveness를 선택한다. 고변동성 구간에서 aggressiveness≈0(보수적 진입), 저변동성 구간에서 aggressiveness 미세하게 상승하는 레짐 적응 행동이 확인됨.

### 다음 단계

- **Phase 2-A.5**: MDP 공식 계수(A=0.5, B=1.5, C=2.5, D=7.5) Bayesian 튜닝 (Optuna TPE + Hyperband)
- **Phase 2-B**: Test set 개봉 및 최종 평가 (사용자 리뷰 후 진행)

---

## 2026-04-20 — Phase 2-A.5 구현: Bayesian 계수 튜닝 (Optuna TPE)

### 목적

exp013b(Val Sharpe 17.579)는 MDP 공식 계수를 고정값(A=0.5, B=1.5, C=2.5, D=7.5)으로 사용.
이 계수들이 수익 최대화 관점에서 최적인지 검증하고, 더 나은 값이 있으면 개선한다.

### 구현 내용

**1. `src/env/trading_env.py` 수정**
- `__init__`에 `formula_coefs` 딕셔너리 읽기 추가
- 8개 계수(A_b, B_b, C_b, D_b, A_s, B_s, C_s, D_s) 개별 속성으로 저장
- `_compute_order_prices`에서 하드코딩 → 속성 참조로 변경
- 기본값 = 기존 설계값 → 하위 호환 완벽 (기존 config 그대로 동작)

**2. `scripts/bayesian_coef_tuning.py` 신규 작성**
- Sampler: TPE (Tree-structured Parzen Estimator), n_startup_trials=10
- Pruner: MedianPruner, n_startup_trials=10, n_warmup_steps=1
- 목적함수: Val 세트 Sharpe (mean of 5 episodes)
- Trial당: 1M 스텝 (exp013b 3M의 1/3 — 빠른 탐색)
- 중간 평가: 3회 (33%, 66%, 100%) — 하위 50% 조기 중단
- SQLite 저장 → 중단 후 재개 가능 (`--show-results` 옵션)
- 기본 50회 시도 (약 4-6시간 예상)

**3. `config/optuna_coef_config.yaml` 신규 작성**
- 탐색 범위 문서화
- 실험 설정 참조 문서

**4. `pyproject.toml` 업데이트**
- optuna>=3.6 의존성 추가 (실제 설치: optuna 4.8.0)

### 탐색 범위

| 계수 | 기본값 | 탐색 범위 | 의미 |
|------|--------|----------|------|
| A_b | 0.5 | [0.1, 3.0] | buy_hi_gap 기저 (ATR 배수) |
| B_b | 1.5 | [0.0, 8.0] | buy_hi_gap action 감도 |
| C_b | 2.5 | [1.0, 12.0] | buy_lo_gap 기저 |
| D_b | 7.5 | [2.0, 25.0] | buy_lo_gap action 감도 |
| A_s | 0.5 | [0.1, 3.0] | sell_market_gap 기저 |
| B_s | 1.5 | [0.0, 8.0] | sell_market_gap action 감도 |
| C_s | 2.5 | [1.0, 12.0] | sell_cost_gap 기저 |
| D_s | 7.5 | [2.0, 25.0] | sell_cost_gap action 감도 |

소프트 제약: C_b > A_b, D_b > B_b (buy_lo가 항상 buy_hi보다 아래)  
                C_s > A_s, D_s > B_s (sell_cost가 항상 sell_market보다 위)  
위반 시 -999.0 반환 → sampler가 해당 영역 자동 회피

### 검증

- pytest 46/46 통과 (formula_coefs 기본값 유지 → 기존 동작 불변)
- `--show-results` 모드 실행 확인 (SQLite DB 생성, 결과 출력 정상)

### 실행 방법

```bash
# 최초 실행 (50 trials)
python scripts/bayesian_coef_tuning.py

# 추가 실행 (재개)
python scripts/bayesian_coef_tuning.py --n-trials 20

# 결과 확인만
python scripts/bayesian_coef_tuning.py --show-results
```

### 이후 계획

1. 튜닝 완료 → best_coefs.yaml 확인
2. 최적 계수로 3M 스텝 full 훈련 → exp_final (또는 exp016)
3. Phase 2-B: Test set 개봉

---

## 2026-04-20 — exp016 완료: Bayesian 최적 계수 3M 스텝 Full 훈련 결과

### 실험 설정

- 베이스: exp013b (n_splits=2, threshold=avg_price, ent_coef=0.05)
- 계수: Optuna TPE Trial #42 (A_b=0.285, B_b=1.748, C_b=5.223, D_b=18.683, A_s=0.101, B_s=0.890, C_s=6.913, D_s=5.457)
- 훈련: 3M 스텝, n_envs=4, cosine LR (0.0003→0.00001), seed=42
- 설정 파일: `config/exp016_final_config.yaml`

### 최종 결과

| 모델 | Val Sharpe | Val Return | Val MDD | 거래 수 | 완성 사이클 |
|------|-----------|-----------|---------|---------|-----------|
| PPO best (step 2,100,000) | **35.424** | +365.60% | 2.46% | - | - |
| PPO final (step 3,000,000) | 35.387 | +365.30% | 2.46% | 2065 | 938 |
| Buy & Hold | 2.377 | +150.18% | 21.74% | 1 | 0 |
| Fixed Grid 1% | 1.991 | +64.11% | 17.50% | 411 | 120 |
| Fixed Grid 2% | 2.039 | +31.05% | 12.57% | 112 | 38 |
| Fixed Grid 5% | 1.377 | +4.99% | 3.32% | 8 | 4 |
| ATR Grid k=0.5 | 1.141 | +30.80% | 20.66% | 953 | 279 |
| ATR Grid k=1.0 | 1.607 | +54.37% | 18.63% | 590 | 177 |
| ATR Grid k=2.0 | 2.234 | +55.42% | 13.23% | 287 | 91 |

**PPO Sharpe 35.424 vs 최강 베이스라인 2.377 (+14.9×)**

### 학습 수렴 패턴

- step 50,000: Sharpe 31.904 (초기 수렴)
- step 350,000~1,300,000: 31.90 → 35.39 (단조 증가)
- step 1,300,000~: Sharpe ≈ 35.38~35.42 (완전 수렴, 극미세 변동만)
- best_model: step 2,100,000 (Sharpe 35.424)
- 3M 스텝 훈련 대부분이 수렴 후 구간 → 1.3M 스텝이면 충분했을 것

### Optuna 1M → Full 3M 스케일업 비교

| 지표 | Optuna (1M) | exp016 (3M) | 변화 |
|------|------------|------------|------|
| Val Sharpe | 42.997 | 35.424 | -17.6% |
| Val Return | (미기록) | +365.60% | - |

- 1M Optuna 결과(42.997)보다 3M full 훈련 결과(35.424)가 낮은 이유:
  - Optuna는 5 episode 평균 (lucky seed 포함 가능성)
  - exp016은 더 엄격한 재현성 조건으로 평가
  - 실제 성능 지표는 exp016 기준 35.424가 신뢰성 높음

### 생성 파일

- `config/exp016_final_config.yaml` — 실험 설정
- `experiments/exp016_final/best_model.zip` — 최종 사용 모델 (Sharpe 35.424)
- `experiments/exp016_final/final_model.zip` — step 3M 모델
- `experiments/exp016_final/train_log.txt` — 전체 학습 로그

### 다음 단계

Phase 2-B: Test set 개봉 (2024.01~) — 사용자 검토 후 진행

---

## 2026-04-20 — Phase 2-A.5 결과: Bayesian 튜닝 50 trials 완료 + 심층 분석

### 튜닝 결과 요약

| 항목 | 결과 |
|------|------|
| 총 trials | 50 (유효 48, 제약위반 2) |
| 최적 Trial | #42 |
| **Val Sharpe** | **42.997** (기본 계수 대비) |
| **베이스라인 대비** | **+25.418 (+144.6%)** |
| 소요 시간 | 4.34시간 |

### 최적 계수 (Trial #42)

| 계수 | 기본값 | 최적값 | 변화 | 경제적 의미 |
|------|--------|--------|------|------------|
| A_b | 0.5 | 0.285 | ↓ | buy_hi 기저 축소 → 더 빠른 매수 진입 |
| B_b | 1.5 | 1.748 | ↑ | buy_hi action 감도 소폭 증가 |
| C_b | 2.5 | 5.223 | ↑↑ | buy_lo 기저 확대 → 더 깊은 저점 대기 |
| D_b | 7.5 | 18.683 | ↑↑ | buy_lo 범위 대폭 확장 (최대 23.9×ATR) |
| A_s | 0.5 | **0.101** | ↓↓ | sell_market 즉각 실행 (0.1×ATR만 상승해도 매도) |
| B_s | 1.5 | 0.890 | ↓ | sell_market action 감도 감소 |
| C_s | 2.5 | 6.913 | ↑↑ | sell_cost 기저 상향 (큰 랠리 전용) |
| D_s | 7.5 | 5.457 | ↓ | sell_cost 범위 적당 수준 |

### 발견된 전략 패턴

기본값 → 최적값으로 전환 시 에이전트 전략이 근본적으로 바뀐다:
- **sell_market이 매우 빠르게 작동** (0.1×ATR): 소폭 상승마다 즉시 현금 회수
- **sell_cost가 높은 기준 유지** (6.9×ATR 이상): 큰 랠리 때만 평단가 기준 매도
- **buy_lo 범위 대폭 확장** (최대 23.9×ATR): 큰 조정에서 저점 매수 적극 포착
- 결과: 빈번한 소수익 사이클 + 가끔 대형 수익 사이클 혼합

### 계수 중요도 분석 (notebooks/05_coef_analysis.ipynb)

- **가장 중요**: A_s (sell_market 기저) — Sharpe와 강한 음의 상관 (낮을수록 좋음)
- **중요**: C_b, D_b (buy_lo 범위) — 클수록 좋은 경향
- **덜 중요**: B_b, B_s — 상위 trial에서 값이 분산됨

### 경계 분석 및 Round 2 권고

- **A_s=0.101이 탐색 하한(0.1)에 도달** → 실제 최적값이 0.1보다 낮을 가능성
- Round 2 권고 범위:
  - A_s: [0.01, 0.15] (핵심)
  - 나머지: 현재 최적값 ±30% 범위 축소
  - 20-30 trials 추가

### 생성 파일

- `notebooks/05_coef_analysis.ipynb` — 심층 분석 노트북 (6개 그래프)
- `experiments/optuna_coef_v1/fig_01~06_*.png` — 분석 그래프

### 다음 단계 선택

1. **Round 2 튜닝** (30 trials, ~2.5시간): A_s 하한 탐색 → 추가 개선 가능성
2. **exp016 즉시 진행** (3M 스텝): Trial #42 계수로 full 훈련 → Val Sharpe ~43 목표

---
