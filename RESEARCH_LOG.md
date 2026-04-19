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
