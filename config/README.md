# Config Files — 실험 설정 인덱스

> 각 yaml은 단일 실험의 모든 수치 파라미터를 담는다.
> 코드 내 하드코딩 금지 — 변경은 yaml에서만.
> 본 논문 위치는 [`docs/PROJECT_GOAL.md`](../docs/PROJECT_GOAL.md) 참조.

## 활성 (현 baseline)

| 파일 | 용도 |
|---|---|
| `experiment_config.yaml` | 기본 환경 설정 (ATR 계수 default, n_splits 등). exp별 config가 없는 경우 fallback. |
| `optuna_coef_config.yaml` | Bayesian coef 튜닝 탐색 범위 |

## Phase 1~2 실험 config (historical)

| 파일 | 실험 | 메모 |
|---|---|---|
| `exp002_config.yaml` | PPO 하이퍼파라미터 1차 튜닝 | LR 1e-4, n_steps 4096 |
| `exp003_config.yaml` | VecNormalize + cosine LR | reward fee 이중차감 버그 발견 후 폐기 |
| `exp004_config.yaml` | exp003 + reward fix | |
| `exp005_config.yaml` | 최소 gap 0.1→0.5 | val p10 ATR에서도 손익분기 |
| `exp006_config.yaml` | 에피소드 단축 + n_envs=4 | Discount 소멸 해결 |
| `exp007_config.yaml` | VecNormalize 비활성 | eval 분포 불일치 해결 |
| `exp008_config.yaml` | ent_coef 0.01→0.05 | regime 적응 발현 |
| `exp009{a,b,c}_config.yaml` | n_splits=2/8/16 ablation | n_splits=2 우세 |
| `exp010_config.yaml` | gamma=0.95 ablation | gamma=0.99 유지 |
| `exp011{a,b}_config.yaml` | window=84/336 ablation | window=168 유지 |
| `exp012_config.yaml` | (이력에 명시 없음, 점검 필요) | |
| `exp013{a,b}_config.yaml` | threshold_basis=price/avg_price | avg_price 우세 |
| `exp014{a,b}{,_v2}_config.yaml` | n_buy_orders ablation (v1, v2 신설계) | n_buy_orders=2 최적 |
| `exp015_ent_01_config.yaml` | ent_coef=0.1 ablation | 0.05 유지 |
| `exp016_final_config.yaml` | **Bayesian best (Trial #42) + 3M** | **Val Sharpe 35.4 / Test 43.0** |

## Phase 3 실험 config (예정, exp030~035)

작성 시 본 디렉토리에 추가. 명명 규칙:
```
exp030_stabilization_config.yaml
exp031_bc_warmstart_config.yaml
exp031b_cql_config.yaml         (조건부)
exp032_sym_config.yaml          # reward variant 4종
exp032_asym_config.yaml
exp032_dsr_config.yaml
exp032_pt_config.yaml
exp033_slippage_dr_config.yaml
exp034_cpcv_config.yaml
```

상세 설계는 [`docs/study/rl_finance/project_continuation_plan.md`](../docs/study/rl_finance/project_continuation_plan.md).

---

## Phase 2 후속 추가 config (Worktree에 일부만 있음)

Phase 2에서 추가된 실험 (exp017~027)의 config는 직접 코드 수정으로 처리되었거나
`experiment_config.yaml` 의 변형으로 진행됨. 일부는 `experiments/expXXX_*/config_snapshot.yaml` 에 보존됨.

본 디렉토리에 없는 config는 `experiments/expXXX_*/config_snapshot.yaml` 참조.
