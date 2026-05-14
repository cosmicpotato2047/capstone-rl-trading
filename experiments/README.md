# Experiments — 실험 디렉토리 인덱스

> 본 디렉토리의 각 폴더는 단일 실험의 산출물 (best/final model, config snapshot, regime_analysis, log) 을 담는다.
> 상세 설계 의도와 결과는 [`RESEARCH_LOG.md`](../RESEARCH_LOG.md) 참조.
> 본 논문 위치는 [`docs/PROJECT_GOAL.md`](../docs/PROJECT_GOAL.md) 참조.

## Phase 1 — 환경 설계 + RL 학습 안정화 (exp001~016)

| 디렉토리 | 핵심 변경 | Val Sharpe (best) |
|---|---|---|
| `exp001_baseline` | 첫 PPO 학습, 1M steps | 0.795 |
| `exp002_tuned` | LR/ent_coef 보수화 | 0.745 |
| `exp005_min_gap_fix` | 최소 gap 0.1→0.5 (fee 손익분기) | — (학습 후반 0거래 수렴) |
| `exp006_short_episode_multienv` | 에피소드 단축 2016 + n_envs=4 | 1.183 |
| `exp016_final` | Bayesian 계수 (Trial #42) + 3M steps | **35.424** |

## Phase 2 — State 확장 + ATR vs RL 비교 (exp017~027)

| 디렉토리 | 핵심 변경 | Val Sharpe / Test Sharpe |
|---|---|---|
| `exp017_phase2_7d` | State 5D→7D (trend_1d, trend_1w 추가) | Val 38.186 |
| `exp020_budget_fraction` | action[0]=aggressiveness→budget_fraction | **Val 48.238 / Test 42.090** |
| `exp021_entry_gate` | action[0]=entry_gate | Val 48.074 |
| `exp022_optuna` | PPO 하이퍼파라미터 Optuna | Best Trial #25, Sharpe 56.4 |
| `exp022_rl_coef` | RL이 계수 결정 (1M 학습) | Val 55.777 / Test 52.802 |
| `exp023_atr_optuna` | ATR 계수 단독 Bayesian (PPO 분리) | Val 60.723 / Test 52.6 |
| `exp026_atr_limitfill` | **체결가 버그 수정** (next_low/high → 지정가) | Val 1.978 / Test 0.935 (현실화) |
| `exp026_rl_optuna` | exp026 환경에서 PPO Optuna | Best Trial #34 |
| `exp026_rl_final` | exp026 환경 RL 1M 학습 | Val 0.896 / Test 0.009 |
| `exp027_atr_direction` | ATR + direction multiplier | Val 2.348 / **Test -0.213** (Val 과적합) |
| `exp027_rl` | **asymmetric reward (beta=2.0)** | Val 2.444 / **Test 1.955** (ATR 2배) |
| `optuna_coef_v1` | (legacy) 초기 Bayesian 50 trials | — |

→ Phase 2의 핵심 발견은 **exp020 (RL=ATR)** 과 **exp027_rl (Asymmetric reward로 RL>ATR)**.

## exp028~029 (보류)

Worktree에 디렉토리 없음. RESEARCH_LOG 참조.
- exp028: early stopping + beta=1.5 안정화 시도
- exp029: MDP 전면 재설계 (5D action, 사이클 시작 시 1회 결정, Val Sharpe 1.440, **학습 불안정**)

## Phase 3 — Reward 변형 비교 (본 논문 메인, exp030~035, 진행 예정)

| 디렉토리 (예정) | 목적 |
|---|---|
| `exp030_stabilization` | PPO 학습 안정화 패키지 (LR schedule, target_kl, ent annealing) |
| `exp031_bc_warmstart` | BC warm-start (ATR 정책으로 PPO pretrain) |
| `exp031b_cql` (조건부) | CQL + mixed dataset |
| **`exp032_reward_variants/{sym,asym,dsr,pt}/`** | **4가지 reward 비교 (메인)** |
| `exp033_slippage_dr` | Slippage + Domain Randomization |
| `exp034_cpcv` | 6-fold CPCV + DSR 계산 |
| `exp035_test_eval` | Test set 봉인 해제 (1회) |

상세 설계는 [`docs/study/rl_finance/project_continuation_plan.md`](../docs/study/rl_finance/project_continuation_plan.md).

---

## 각 실험 디렉토리의 일반 구조

```
expNNN_name/
├── best_model.zip          # 최고 Val Sharpe 모델
├── final_model.zip         # 학습 종료 시점 모델
├── config_snapshot.yaml    # 재현용 config 사본
├── regime_analysis.csv     # (선택) state-action 수집 데이터
├── train_log.txt           # 학습 로그
└── test_eval_results.yaml  # (선택) Test 평가 (exp035 등)
```
