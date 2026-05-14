# CLAUDE.md — 프로젝트 브리핑

Claude Code가 이 프로젝트에서 작업할 때 반드시 숙지해야 할 규칙과 맥락이다.

**단일 기준점**: `docs/PROJECT_GOAL.md` — 모든 작업은 여기 정의된 RQ에 정렬되어야 한다.
상세 설계는 `docs/MDP.md`, `docs/FORMULAS.md`, `docs/RELATED_WORK.md`, `config/experiment_config.yaml`을 참고한다.

---

## 프로젝트 RQ (2026-05-14 확정)

> **BTC 그리드 트레이딩에서 reward 설계가 RL 정책의 알파에 어떤 영향을 미치는가?
> 특히, 어떤 reward 함수 하에서 RL이 ATR 규칙 기반을 초과하며 (혹은 초과하지 못하며), 그 메커니즘은 무엇인가?**

- **자산 범위**: BTC/USDT 1시간봉 **단일 자산**. 자산 확장은 졸업 논문 범위 외.
- **메인 비교 단위**: 4가지 reward 변형 (Symmetric / Asymmetric / DSR / Prospect-theoretic) × ATR baseline.
- **현재 위치**: Phase 3 진입. exp030~035 시리즈가 본 논문의 메인 실험.

⚠️ **RQ 표현 원칙**: 사전 증거(exp027_rl Test Sharpe 1.955)가 강하더라도 RQ는 열린 질문 형태로 유지.
강한 사전 증거는 **가설(H1~H4)** 의 정당성으로 흡수하지 RQ의 답으로 단정하지 않음 (학술 컨벤션 + 확증 편향 회피).

상세는 `docs/PROJECT_GOAL.md`, `ROADMAP.md` 참조.

---

## 절대 금지 규칙

1. **`data/processed/btc_test.parquet` 열람 금지**
   테스트셋은 exp035(최종 평가)까지 완전 봉인된다.
   어떤 코드, 노트북, 스크립트에서도 test 파티션을 로드하거나 출력해서는 안 된다.
   Gort et al.(2022) 백테스트 과적합 방지 설계의 핵심이다.

2. **`data/raw/`의 원본 파일 수정 금지**
   다운로드한 OHLCV 원본은 불변이다. 전처리는 반드시 `data/processed/`에 저장한다.

3. **RQ를 벗어나는 작업 자동 진행 금지**
   자산 확장, hierarchical RL, multi-asset portfolio 등 RQ에서 명시적으로 제외한 영역은
   사용자 명시적 합의 없이 시작하지 않는다. `docs/PROJECT_GOAL.md`의 "의식적으로 제외하는 것" 참조.

---

## 프로젝트 핵심 — 반드시 숙지

### 이 프로젝트는 방향성 트레이딩이 아니다

BTC 가격이 오를지 내릴지 예측하는 시스템이 아니다.
**변동성 자체에서 수익을 추구하는 동적 그리드 트레이딩**이다.

- 잘못된 접근: "RSI가 30 이하면 매수" 같은 방향성 신호
- 올바른 접근: "지금 변동성 수준 + 보상함수 설계에서 어떤 그리드 정책이 알파를 만드는가"

### 본 논문의 핵심 가설 (사전 증거 → exp032에서 검증)

- **H1 (사전 증거 있음)**: Symmetric reward + ATR 비례 공식 조합에서 RL은 ATR과 동등 (exp020~022)
- **H2 (사전 증거 있음, 검증 필요)**: Asymmetric / DSR / Prospect-theoretic reward 로 정식화하면 RL이 ATR 초과 가능 (exp027_rl)
- **H3 (검증 필요)**: 우수 reward의 효과는 "선택적 진입" 행동으로 나타남
- **H4 (검증 필요)**: 그 우위(또는 비우위)가 CPCV + Slippage에서도 유지

가설이지 결론이 아님. exp032에서 정식 검증한다.

### Action은 연속 다차원이다

```python
# 현 표준 (exp020~027)
action_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
# [0] aggressiveness  → 매수 간격 결정
# [1] profit_target   → 매도 간격 결정

# exp029 변형 (사이클 시작 시 1회 결정, 5D)
action_space = Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
# [0] n_splits_coef  [1] gap_b1  [2] gap_b2  [3] gap_s1  [4] gap_s2
```

이산 action (Discrete(3) Buy/Sell/Hold) 은 절대 사용 금지.

### Sell 우선 원칙

같은 봉에서 buy와 sell이 동시 체결 가능할 때 **sell을 먼저 처리**한다.

### RL 설계 무결성 체크 (메모리 feedback)

새 기능 추가, 최적화, 파라미터 튜닝 작업을 시작하기 전에 반드시:
> **"이 작업이 RL agent가 state를 보고 다르게 행동하는 능력을 유지하는가?"**

Bayesian 최적화가 RL action과 동일한 파라미터 (gap 크기) 를 동시 제어하면
정책이 [0,0]으로 포화 수렴한다 (exp016 사례). 계수와 RL action의 역할이 겹치지 않도록 설계.

---

## MDP 빠른 참조

### State (현 7D, exp029는 9D)

```
# 기본 7D (exp020 이후 표준)
[0] log_price            = log(price / price.rolling(168).mean())
[1] divergence           = (avg_price - price) / avg_price
                           # 보유 중: 현재 평단가 기준
                           # 미보유 + 직전 사이클 있음: last_avg_price 기준
                           # 미보유 + 거래 이력 없음: 0.0
[2] holdings_value_ratio = (holdings × price) / start_capital  # 미보유 시 0
[3] cash_ratio           = cash / start_capital
[4] volatility           = ATR(168) / price
[5] trend_short          = pct_change(72)
[6] trend_long           = pct_change(720)

# exp029 추가 2D (사이클 1회 결정 방식)
[7] idle_norm            = idle_steps / grace_period
[8] n_splits_norm        = (n_splits - n_splits_min) / (n_splits_max - n_splits_min)
```

모든 변수 rolling z-score (window=168) 정규화, clip [-5, 5].

### Action → 주문 변환 공식 (ATR 비례 스케일링, 현 baseline)

```python
atr_ratio = ATR(168) / price               # volatility_raw 컬럼

# 7D state + 2D action 표준
buy_hi_gap      = atr_ratio * (A_b + B_b * aggressiveness)
buy_lo_gap      = atr_ratio * (C_b + D_b * aggressiveness)
sell_market_gap = atr_ratio * (A_s + B_s * profit_target)
sell_cost_gap   = atr_ratio * (C_s + D_s * profit_target)

buy_hi      = price     * (1 - buy_hi_gap)
buy_lo      = price     * (1 - buy_lo_gap)
sell_market = price     * (1 + sell_market_gap)
sell_cost   = avg_price * (1 + sell_cost_gap)
```

ATR 고정 시스템(exp026 best)의 Bayesian 계수: A_b=1.921, C_b=5.719, A_s=0.688, C_s=9.673, n_splits=3.

### 체결 방식 (exp026 이후)

```
지정가(limit) 체결:
  next_low  <= buy_hi      → buy_hi  가격으로 매수 체결
  next_low  <= buy_lo      → buy_lo  가격으로 매수 체결
  next_high >= sell_market → sell_market 가격으로 매도 체결
  next_high >= sell_cost   → sell_cost   가격으로 매도 체결
```

⚠️ 이전 버전은 `next_low/next_high`로 체결 (favorable bias) → 수조% 가짜 수익 → exp026에서 수정.

### Reward (현 baseline)

```python
step_reward = (equity_t - equity_{t-1}) / start_capital  # symmetric
```

- 수수료는 `_execute_buy` / `_execute_sell`에서 cash에 이미 반영. 별도 차감 불필요.
- `fee_rate`: 0.05% (Binance maker fee)
- 사이클 종료 시 별도 보너스 없음. 통계만 기록.

### Phase 3 reward 변형 (exp032 비교 대상)

| 코드 | 정의 |
|---|---|
| `sym` | 현 baseline (위 공식) |
| `asym` | `r = sign(x) * abs(x) * (1 if x>=0 else β)`, β=2.0 |
| `dsr` | Differential Sharpe Ratio (Moody 2001) |
| `pt` | `r = sign(x) * abs(x)^α * (1 if x>=0 else λ)`, α=0.88, λ=2.25 |

---

## 데이터

- **소스**: ccxt Binance API (`BTC/USDT`, 1시간봉)
- **yfinance는 보조 용도만**: 최근 730일 제한으로 학습 기간 전체 커버 불가

| 구분 | 기간 | 파일 |
|------|------|------|
| Train | 2017-08-17 ~ 2022-12-31 | `data/processed/btc_train.parquet` |
| Validation | 2023-01-01 ~ 2023-12-31 | `data/processed/btc_val.parquet` |
| Test | 2024-01-01 ~ | `data/processed/btc_test.parquet` ← **봉인 (exp035까지)** |

---

## 다음 실험 시리즈 (Phase 3 진행 중)

| Exp | 목적 | 논문 챕터 | 상태 |
|---|---|---|---|
| exp030 | PPO 학습 안정화 | Method 3.3 | 대기 |
| exp031 | BC warm-start | Method 3.4 | 대기 |
| exp031b | CQL + mixed (조건부) | Method 3.4 | 조건부 대기 |
| **exp032** | **4가지 reward 비교 (메인)** | **§5 Positive finding** | 대기 |
| exp033 | Slippage + DR | §7.1 Robustness | 대기 |
| exp034 | CPCV 6-fold + DSR | §5, §7.2 | 대기 |
| exp035 | Test set 봉인 해제 | §7.3 최종 | 대기 |

기존 exp001~029 결과는 RQ-1 negative finding + RQ-2 가설 시발점으로 활용. `RESEARCH_LOG.md` 참조.
상세 설계는 `docs/study/rl_finance/project_continuation_plan.md` 참조.

---

## Git 작업 규칙

### 브랜치 전략
- `main` — 발표/제출 가능한 안정 버전만
- `feature/*` — 기능 단위 개발. 완성 후 main에 merge하고 브랜치 삭제

### 자동화 규칙 (Claude Code가 항상 따름)
- 기능 구현 시작 시 **자동으로 feature 브랜치 생성**
- 구현 완료 후 **자동으로 main에 merge + push + 브랜치 삭제**
- 사용자 확인 없이 진행 (혼자 하는 프로젝트이므로)
- 단, 보고서/문서 업데이트는 브랜치 없이 main에 직접 커밋

### RESEARCH_LOG.md 자동 기록 규칙
코드 변경을 동반하는 작업이 완료되면 **사용자 요청 없이** 반드시 기록한다.

기록 조건:
- `src/`, `scripts/`, `config/`, `tests/` 등 코드/설정 파일을 수정한 경우
- 설계 의사결정이 바뀐 경우 (MDP, Reward, State 구조 등)
- 실험을 실행하고 결과가 나온 경우

기록 내용 (날짜별 섹션 추가):
- **무엇을** 변경했는가 (파일, 함수, 파라미터)
- **왜** 변경했는가 (기존 문제, 결정 근거)
- **결과** (검증 수치, env_checker 통과 여부 등)
- **보류한 아이디어**가 있으면 "아이디어 기록" 항목에 남김

기록 후 `docs: RESEARCH_LOG.md 업데이트` 로 main에 직접 커밋 및 push한다.

#### 실험 (expXXX) 결과 기록 표준 템플릿

exp030 이후 모든 실험은 다음 6-section 구조를 따른다 (논문 작성 시 검색/인용 일관성 확보):

```markdown
## YYYY-MM-DD — expXXX_name 완료

### Objective (RQ 매핑)
- 무엇을 검증하려고 했는가
- 어떤 RQ/가설에 답하는가 (예: RQ-2 / H2a)

### Changes
| 파일 | 변경 |
|---|---|
| (경로) | (한 줄 요약) |

### Hyperparameter
- 기존 baseline 대비 어떤 hyperparameter 변경
- exp별 config 파일 경로

### Results
| Metric | Value | 비고 |
|---|---|---|
| Val Sharpe (mean ± std) | X.XX ± Y.YY | n_seed=5 |
| Cohen's d vs baseline | Z.ZZ | — |
| MDD / Calmar / Trades | ... | |

### Behavior Analysis (해당 시)
- Regime별 행동 분포
- (exp032c부터) Counterfactual / SHAP / Mediation 핵심 발견

### Decision
- 다음 실험으로 무엇? 왜?
- 가설 H1~H4 중 어느 것이 (부분) 지지 / 부정 되었는가

### Figures
- `reports/.../figures/expXXX_*.png` 파일명만 기록

### 보류 아이디어
- (있으면, 미래 작업 후보)
```

→ 매 exp 기록이 동일 구조라 `docs/RESULTS_SUMMARY.md` 갱신 시 그대로 옮겨갈 수 있음.

#### RESULTS_SUMMARY.md 동기 갱신

각 exp 완료 시 RESEARCH_LOG 기록과 함께 `docs/RESULTS_SUMMARY.md` 의 해당 Phase 섹션도 갱신한다 (수치만 옮겨가는 짧은 작업).

### Commit Message Convention (Conventional Commits)
```
feat:     새 기능 구현
fix:      버그 수정
docs:     문서, 보고서, 주석
test:     테스트 추가/수정
refactor: 동작 변경 없는 코드 정리
chore:    설정, 패키지, .gitignore 등
data:     데이터 관련 스크립트
```

---

## 코드 작성 원칙

### 구조
- **로직은 반드시 `src/`에** — 노트북은 `src/`를 import해서 사용, 로직 직접 작성 금지
- **스크립트는 `scripts/`에** — 커맨드라인 실행 진입점. `src/`를 호출하는 얇은 레이어
- **실험 파라미터는 `config/`에** — 코드 내 하드코딩 금지. exp별 yaml 분리.

### 의존성
- 기술 지표: ATR은 `src/data/preprocessor.py`에서 pandas로 직접 계산. 외부 라이브러리 불필요
- 데이터 다운로드: `ccxt` (Binance). `yfinance`는 보조 용도만
- RL: `stable-baselines3` + `gymnasium`. `gym` (구버전) 사용 금지
- Offline RL (exp031b 조건부): `d3rlpy` (CQL, IQL)
- 실험 추적: `mlflow` (로컬). Weights & Biases 사용 금지
- Hyperparameter: `optuna` (TPE + MedianPruner)

### 환경 검증
`trading_env.py` 수정 후 반드시 실행:
```python
from gymnasium.utils.env_checker import check_env
check_env(env)
```

또한 `tests/test_trading_env.py` 46개 테스트 통과 확인.

---

## 주요 문서 위치

| 문서 | 경로 | 내용 |
|------|------|------|
| **본 논문 목표 (단일 기준점)** | `docs/PROJECT_GOAL.md` | **RQ, 가설, 평가 방법, scope** |
| 로드맵 | `ROADMAP.md` | Phase별 진행 현황 + 의사결정 기록 |
| 실험 시리즈 상세 | `docs/study/rl_finance/project_continuation_plan.md` | exp030~035 설계 |
| 학습 자료 허브 | `docs/study/rl_finance/00_overview.md` | 24개 이론 노트 인덱스 |
| 공식 정의 | `docs/FORMULAS.md` | ATR 고정 / RL 버전 공식 |
| MDP 설계 | `docs/MDP.md` | State/Action/Reward 설계 근거 |
| 선행 연구 | `docs/RELATED_WORK.md` | 논문 7편 역할 및 차별점 |
| 실험 설정 | `config/experiment_config.yaml` | 모든 수치 파라미터 |
| 실험 기록 | `RESEARCH_LOG.md` | 날짜별 의사결정 |
| 제안서 v3.0 | `D:\PARA\Assets\Semesters\26-1\캡디\캡스톤_주제제안서_v3.0.docx` | 지도교수 제출용 공식 문서 (Pivot 2 반영 필요할 수 있음) |
