# Scripts — CLI 진입점

> 본 디렉토리는 커맨드라인 실행 진입점만 담는다. 로직은 `src/` 에.
> 본 논문 위치는 [`docs/PROJECT_GOAL.md`](../docs/PROJECT_GOAL.md) 참조.

## 데이터 파이프라인

| 스크립트 | 용도 |
|---|---|
| `download_data.py` | ccxt Binance API로 BTC/USDT 1h OHLCV 다운로드 |
| `preprocess_data.py` | ATR, log_price, trend_*, rolling z-score 계산 + parquet 저장 |

## 학습/평가

| 스크립트 | 용도 |
|---|---|
| `train_ppo.py` | PPO 학습 (config 지정). MLflow 자동 로깅. best_model 저장 |
| `eval_atr_test.py` | ATR 고정 정책 평가 (직접 시뮬레이션) |
| `eval_test.py` | RL 모델 Test set 평가 (봉인 해제 후) |
| `analyze_regime.py` | 학습된 정책의 regime별 행동 분석 (state-action 수집 + 통계) |

## Optuna 튜닝

| 스크립트 | 용도 |
|---|---|
| `tune_ppo_optuna.py` | PPO 하이퍼파라미터 TPE 탐색 |
| `tune_atr_optuna.py` | ATR 공식 계수 단독 Bayesian (PPO 분리) |
| `bayesian_coef_tuning.py` | (legacy) Phase 2 초기 Bayesian — 사용 안 함 |

## 배치 실행 (legacy, Phase 2)

| 스크립트 | 용도 |
|---|---|
| `run_experiments_batch.py` | exp009~011 묶음 실행 |
| `run_batch_013_014.py` | exp013, exp014 묶음 실행 |
| `run_batch_014v2_015.py` | exp014_v2, exp015 묶음 실행 |

## 시각화/보고서 자료

| 스크립트 | 용도 |
|---|---|
| `make_paper_figures.py` | 논문용 figure 일괄 생성 |
| `make_ppt.py` | 발표 자료 (pptx) 생성 |
| `make_project_flow.py` | project_journey.html 인터랙티브 생성 |

---

## Phase 3 추가 예정 스크립트

| 스크립트 (예정) | 용도 |
|---|---|
| `bc_pretrain.py` | exp031 BC warm-start (ATR trajectory → PPO policy MSE pretrain) |
| `cql_pretrain.py` | (조건부) exp031b CQL pretrain (d3rlpy) |
| `cpcv_eval.py` | exp034 6-fold CPCV 평가 + DSR 계산 |
| `compute_dsr.py` | Deflated Sharpe Ratio 단일 계산 (Optuna trial log 입력) |

상세 설계는 [`docs/study/rl_finance/project_continuation_plan.md`](../docs/study/rl_finance/project_continuation_plan.md).
