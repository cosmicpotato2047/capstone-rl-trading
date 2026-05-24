# 실험 결과 요약 — 논문 인용용 Quick Reference

> 본 문서는 **졸업 논문 작성 시 핵심 수치를 빠르게 찾기 위한 단일 참조표**.
> 상세 의사결정과 시행착오는 [`../RESEARCH_LOG.md`](../RESEARCH_LOG.md) 참조.
> RQ 및 가설은 [`PROJECT_GOAL.md`](PROJECT_GOAL.md) 참조.
> **환경 변천 및 인용 가능성은 [`ENV_HISTORY.md`](ENV_HISTORY.md) 참조** (필수).

## 환경 태그 (본 표의 각 결과는 어느 환경에서 나왔는지 표시)

- **Env-v2** — 2D ATR 비례 + favorable bias 체결 (exp006~023)
- **Env-v3** — 4D 절대 gap + 지정가 체결 (exp024~028)
- **Env-v4** — 2D ATR 비례 + 지정가 체결 (canonical, exp030~)

| 환경 | 본 논문 직접 인용 가능? |
|---|---|
| Env-v2 | ⚠️ 정성적 only (수치는 favorable bias artifact 명시) |
| Env-v3 | ❌ 다른 환경 (Env-v4 에서 재현 필요) |
| **Env-v4** | ✓ **본 논문 환경 (canonical)** |

---

## Phase 1 — 환경 설계 + RL 학습 (exp001~exp016, 2026-04)

> **환경**: 대부분 Env-v2 (2D ATR 비례 + favorable bias 체결). 수치는 artifact 영향 있음.

### 베이스라인 (Val 2023, 강세장) — Env-v2

| 전략 | Sharpe | Return (%) | MDD (%) | Trades | Cycles |
|------|--------|-----------|---------|--------|--------|
| Buy-and-Hold | 2.377 | +150.18 | 21.74 | 1 | 0 |
| Fixed Grid 1% | 2.610 | +43.16 | 10.77 | 567 | 141 |
| Fixed Grid 2% | 2.032 | +16.96 | 7.65 | 126 | 41 |
| Fixed Grid 5% | 1.375 | +2.47 | 1.66 | 8 | 4 |
| ATR Grid k=0.5 | 1.118 | +24.70 | 16.07 | 1464 | 347 |
| ATR Grid k=1.0 | 1.434 | +39.79 | 15.86 | 833 | 213 |
| ATR Grid k=2.0 | 1.948 | +28.98 | 9.38 | 320 | 91 |

→ **PPO 목표선 (Val): Fixed Grid 1% Sharpe 2.610**

### PPO 학습 진화

| Exp | 핵심 변경 | Val Sharpe (best) | 비고 |
|-----|----------|------------------|------|
| exp001 | 첫 PPO 학습 (1M steps) | 0.795 (100k) | 후반 -0.28로 붕괴 |
| exp002 | LR 1e-4, ent 0.005 | 0.745 (50k) | 안정성 ↑, peak ↓ |
| exp005 | 최소 gap 0.1→0.5×ATR (fee 손익분기) | 4.588 (1.05M, 0거래 artifact) | 학습 후반 0거래 수렴 |
| exp006 | 에피소드 단축 2016 + n_envs=4 | 1.183 (550k) | Discount 소멸 해결 |
| exp007 | VecNormalize 비활성 + multi-eval | **11.884** | 5 episode 평균. 평가 버그 fix |
| exp008 | ent_coef 0.05 (regime 적응 유도) | **12.381** | Low vs High vol t-test p=1e-81 |
| exp013b | n_splits=2 + threshold=avg_price | 17.579 | n_splits ablation 완료 |
| exp016 | Bayesian 계수 (Trial #42) + 3M | **35.424** | Phase 1 best |

### Phase 1 Test Set 평가 (exp016, 2024+, 19,901봉) — Env-v2 ⚠️

| 모델 | **Sharpe** | Return (%) | MDD (%) | Trades | Cycles |
|------|-----------|-----------|---------|--------|--------|
| **PPO exp016 best** | **43.040** | (수조%, artifact) | 3.12 | 20,896 | 9,520 |
| Best baseline (Fixed Grid 5%) | 1.472 | — | — | — | — |

> ⚠️ **Env-v2 (체결가 favorable bias) artifact** (exp026에서 발견 + 수정). 본 논문 §4 Negative finding에서 "체결가 버그 사례" + 정성적 인용으로 활용. Sharpe 수치는 인용 불가.

### Phase 1 핵심 발견 — Policy Saturation

- exp016 best 정책의 deterministic action: `[aggressiveness=0.000, profit_target=0.000]` — 완전 포화
- raw network output `[-9.19, -4.30]` → tanh+rescale → `[≈0, ≈0]`
- 모든 state에서 동일 action → **RL이 학습한 게 사실상 상수**

---

## Phase 2 — ATR vs RL + Asymmetric Reward (exp017~exp027, 2026-04)

> **환경**:
> - exp017~023 (전반): Env-v2 — 2D ATR 비례 + favorable bias
> - exp024~028 (후반): Env-v3 — 4D 절대 gap + 지정가 체결
> 둘 다 본 논문 환경 (Env-v4) 와 다름. Env-v3 결과는 재현 필요.

### Decisive Ablation: Fixed Policy 비교 — Env-v2 ⚠️

| | Val Sharpe | Test Sharpe |
|---|---|---|
| RL exp020 | 45.390 | 42.090 |
| **Fixed [1.0, 0.0]** (수렴값 고정) | **45.390** | **41.769** |
| Fixed [1.0, 0.5] | 4.116 | 4.683 |
| Fixed [1.0, 1.0] | 1.060 | 1.843 |

→ **RL과 Fixed [1.0, 0.0] Val Sharpe 완전 일치.** RL의 학습 결과는 상수.
→ **본 논문 §4 Negative finding 핵심 인용**.

### 2-A. 체결가 버그 수정 (exp026, 2026-04-22)

- 기존: `next_low/next_high` 로 체결 → 매수는 봉의 최저, 매도는 봉의 최고 → **구조적 spread 수익 artifact**
- 수정: 지정가 (limit price) 로 체결

### Phase 2 결과 (체결가 수정 후) — Env-v3 (4D 절대 gap) ❌ 재현 필요

#### Val 2021-2023H1 (확장 데이터) — Env-v3

| 시스템 | Val Sharpe | Val Return (%) | Val MDD (%) | Trades |
|--------|-----------|---------------|-------------|--------|
| **ATR (exp026 Bayesian best)** | **1.978** | 34.81 | 6.01 | 1,176 |
| RL exp026 (symmetric reward) | 0.896 | 1.90 | 1.07 | 221 |
| ATR + direction (exp027) | 2.348 | 17.89 | 1.90 | — |
| **RL exp027_rl (asym β=2.0)** | **2.444** | **18.25** | **1.28** | **214** |

#### Test 2023H2-2026 (봉인 해제, Phase 2 시점) — Env-v3

| 시스템 | **Test Sharpe** | Test Return (%) | Test MDD (%) | Trades |
|--------|----------------|-----------------|--------------|--------|
| ATR exp026 | 0.935 | 7.65 | 2.43 | 1,591 |
| RL exp026 (sym) | 0.009 | 0.26 | 1.21 | — |
| ATR + direction (exp027) | -0.213 | -1.40 | 3.43 | 1,662 |
| **RL exp027_rl (asym β=2.0)** | **1.955** | **5.02** | **0.39** | **214** |

### Phase 2 핵심 발견 — Asymmetric Reward의 위력

> **Asymmetric reward (β=2.0) 만으로 RL이 ATR을 Test Sharpe 기준 2.1배 초과 (1.955 vs 0.935), MDD는 1/6 (0.39% vs 2.43%), 거래 횟수는 1/7 (214 vs 1,591).**

이 발견이 **본 졸업 논문의 메인 contribution (§5 Positive finding)** 의 사전 증거.

⚠️ **단, 이 결과는 Env-v3 (4D 절대 gap) 환경.** 본 논문 환경 (Env-v4, 2D ATR 비례) 에서 재현 검증 필요 (작업 중 — Step 11~13). 재현 결과에 따라 §5 의 톤 결정:
- 재현 성공 → strong positive
- 부분 재현 (효과 약화) → moderate
- 재현 실패 → asymmetric reward 의 환경 의존성 발견 자체가 §6 새 contribution

### Phase 2 시행착오 / 발견 timeline

| 날짜 | 발견 / 결정 |
|---|---|
| 2026-04-21 | Pivot 1 — RL 단독 → ATR vs RL 비교 연구로 전환 |
| 2026-04-22 | Bayesian 계수 최적화 (Trial #42) — Val Sharpe +144.6% |
| 2026-04-22 | exp020/021/022로 "RL = ATR" 확정 (Phase 1 발견 재현) |
| 2026-04-22 | 체결가 버그 발견 → 지정가로 수정 (이전 결과 모두 무효화) |
| 2026-04-22 | exp026 재최적화 — Val Sharpe 1.978로 현실화 |
| 2026-04-23 | exp027 ATR+direction — Val +0.65 / **Test -1.15 (Val 과적합)** |
| 2026-04-23 | **exp027_rl asymmetric reward — Test Sharpe 1.955 (ATR 2배 초과)** |

---

## Phase 2.5 — 환경 복원 (Env-v4 canonical 확립, 2026-05-14 완료)

> **목적**: Env-v3 (4D 절대 gap) 의 학술 정합성 약점 해결. Env-v4 (2D ATR 비례 + 지정가 체결) 로 본 논문 정식 환경 확립. 그 위에서 Phase 2 사전 증거 재현.

### Env-v4 ATR Baseline (Bayesian 50 trials, Val 2021-2023)

**최적 계수** (Trial #34):
- A_b=1.665, C_b=6.070, A_s=0.285, C_s=1.951, n_splits=2

| Metric | Value |
|---|---|
| Return | +35.80% |
| **Val Sharpe** | **1.505** |
| MDD | 9.83% |
| Trades | 2,121 |
| Cycles | 1,009 |
| Avg cycle PnL | +0.031% |
| Avg cycle hours | 1.2 h |

⚠️ exp023 Env-v2 계수 (A_b=0.285 등) 는 Env-v4 에서 **Sharpe -4.738** — 환경 의존성 강함. 환경 복원의 정당성 실증.

### Env-v4 RL Asymmetric Reward (β=2.0) 재현

**PPO 학습**: 1M steps, n_envs=4, exp026 Optuna PPO 설정 (lr 1.67e-4, n_steps 4096, clip 0.103).
Early stopping 발동 (100k peak 후 patience=6) → 400k 종료.

**학습 곡선** (eval_freq=50k):

| Step | Val Sharpe | Return | MDD |
|---|---|---|---|
| 50k | 2.223 | +7.95% | 1.80% |
| **100k** | **2.250 (best)** | +8.00% | 1.75% |
| 150-400k | 1.728~2.131 (oscillation) | 5.5~6.7% | 1.8~1.9% |

| Metric | Best (100k) | Final (400k) |
|---|---|---|
| Val Sharpe | **2.250** | 1.828 |
| Return | +8.00% | +5.83% |
| MDD | 1.75% | 1.92% |
| Trades | — | 107 (final) |
| Cycles | — | 51 (final) |

### 재현 vs 원본 비교

| Metric | 원본 (Env-v3) | 재현 (Env-v4, best) | Δ |
|---|---|---|---|
| Val Sharpe | 2.444 | 2.250 | -8% (재현 부분 성공) |
| Val Return | 18.25% | 8.00% | -56% (보수적) |
| MDD | 1.28% | 1.75% | +0.47p |
| Trades | 214 | 107 | -50% (더 선택적) |

→ **재현 부분 성공.** Sharpe 8% 감소했지만 본질적 효과 (asymmetric reward 가 ATR baseline 초과) 유지.

### 본 논문 §5 메인 결과 (Env-v4 canonical)

| 시스템 | Val Sharpe | vs ATR baseline |
|---|---|---|
| **ATR Baseline (Env-v4)** | **1.505** | — |
| **RL with asymmetric β=2.0 (Env-v4 best)** | **2.250** | **+0.745 (+49%)** |
| RL with asymmetric β=2.0 (Env-v4 final, early stop) | 1.828 | +0.323 (+21%) |

→ **시나리오 A (낙관) 확정**: Reward design 이 RL 알파의 채널임을 Env-v4 에서도 검증.

### 학술적 함의

1. **이전 Phase 2 발견 (asymmetric reward 효과) 가 환경 변경 후에도 유지** — 발견의 robustness 입증.
2. **효과 약화 (8%) 가 있지만 본질 유지** — 4D 자유도 → 2D ATR 비례로 일부 표현력 손실 가능. exp032 4 variant 비교에서 추가 검증.
3. **환경 의존성 발견** (exp023 Env-v2 계수가 Env-v4 에서 음수) — 본 논문 §3 Method 에서 "시뮬레이터 fill 가정의 결정적 영향" 명시.

---

## Phase 3 — 진행 중 (2026-05-14~)

> **환경**: Env-v4 (canonical). 본 졸업 논문 메인 챕터.

### exp030 — PPO 학습 안정화 패키지 (완료, 2026-05-14)

**Hyperparameter**: LR linear 3e-4→1e-5, target_kl 0.02, ent annealing 0.01→0.001, clip 0.2, n_steps 4096, patience 10, reward sym (β=1.0).

**1M steps 완주** (early stop 미발동):

| Metric | Best (550k) | Final (1M) |
|---|---|---|
| Val Sharpe | **1.974** | 1.209 |
| Return | +7.24% | +7.02% |
| MDD | 3.95% | 7.02% |
| vs ATR Baseline (1.505) | **+31%** | -20% |

**성공 기준 점검**:
- Val Sharpe ≥ 1.0 (floor): ✓
- 후반 변동 ± 0.3: △ (0.38)
- final ≥ best × 0.8: ✗ (61%)

→ **부분 성공.** Best 시점에서는 ATR baseline 31% 초과. 안정화 패키지로 best step 100k→550k 늦춤. 다만 700k 이후 붕괴 패턴 미해결 — exp031 (BC warm-start) 의 동기.

### 본 논문 메인 비교표 업데이트 (Env-v4)

| 시스템 | Val Sharpe (best) | vs ATR | 비고 |
|---|---|---|---|
| **ATR Baseline** | **1.505** | — | Env-v4 Bayesian Trial #34 |
| **RL sym (exp030 + 안정화)** | **1.974** | **+31%** | symmetric reward, 1M steps 완주 |
| **RL asym β=2.0 (exp_rl_replicate)** | **2.250** | **+49%** | asymmetric reward, 100k peak |

→ **두 가지 발견 확정**:
1. **§5 Positive finding 유지** — Reward variant (sym → asym) 가 RL 알파에 의미 있는 차이 (+14% 추가 우위)
2. **§4 Negative finding 약화** — Env-v4 에서 sym RL 도 best 시점에 ATR 초과 (학습 안정성 의존). exp020 Env-v2 "RL = Fixed [1.0, 0.0]" 가 Env-v4 에서는 정확히 성립 안 함.

### exp031 — BC Action Bias Init (Negative Result, 2026-05-14)

**시도**: PPO policy network 의 action_net bias 를 ATR Baseline 행동에 매칭 (bias=[-10,-10], 이후 [-3,-3] 으로 약화).

**결과**: 두 시도 모두 학습 정체 (Sharpe 1.526 정확히 동일, early stop).

**원인**: SB3 PPO + Box action_space [0,1] 의 clipping (deterministic action = clip(raw_mean, 0, 1)). bias 음수면 dead zone, gradient signal 약함.

**implication**:
- exp032~035 메인 챕터 영향 **없음** (random init 사용)
- 본 논문 §3.4 Method 또는 §8 Discussion 에서 "단순 bias init 의 한계 + future work" 한 줄
- 정석 BC pretrain 또는 SAC 전환은 future work

### exp032a — Reward Variant Hyperparameter Optuna 튜닝 (완료, 2026-05-15)

**Optuna TPE 30 trials × 200k steps × 3 variant** (sym 은 hyperparameter 없음).

| Variant | Best Trial | Best Params | Val Sharpe (200k single-seed) | vs ATR (1.505) |
|---|---|---|---|---|
| asym | #23 | β=3.4195 | 1.5166 | +0.8% |
| **dsr** | #1 | η=0.0352 (≈ 1/28h EMA) | **1.8883** | **+25%** |
| pt | #18 | α=0.6825, λ=3.3029 | 1.8035 | +20% |

→ **DSR 1위, PT 2위, asym 3위.** 단 200k single-seed 라 노이즈 큼. exp032b 가 본 검증.

→ 출력: `config/exp032b_{sym,asym,dsr,pt}_config.yaml` 4개 (exp032b 입력으로 사용)

### exp032b — 4 Reward Variant × 10 Seeds 본 비교 (완료, 2026-05-15) — §5 메인

**40 runs × 1M = 40M steps**, 3h 44min 소요.

| Variant | Best Sharpe (10 seeds) | Final Sharpe | MDD (%) | Calmar | Trades | Return (%) |
|---|---|---|---|---|---|---|
| **sym**  | **1.871 ± 0.22** | 1.015 ± 0.43 | 3.27 ± 0.82 | 0.60 | 120 | 6.60 |
| **dsr**  | 1.809 ± 0.21 | **1.204 ± 0.41** | 4.33 ± 1.70 | 0.47 | 117 | **7.40** |
| asym | 1.681 ± 0.10 | 1.101 ± 0.27 | **2.28 ± 0.31** | **0.755** | 96 | 5.23 |
| pt   | 1.667 ± 0.09 | 1.082 ± 0.18 | 2.31 ± 0.29 | 0.735 | 85 | 4.88 |
| (ATR baseline) | (1.505) | — | (9.83) | (0.153) | (2,121) | (35.80) |

**모든 4 variant 가 ATR baseline 을 best 기준 +11~24% 초과.** 단 4 metric 1위가 모두 다름.

#### 두 클러스터 (Cohen's d 기반)

- **Aggressive {sym, dsr}**: 높은 Sharpe + 높은 MDD + 많은 거래 (~120)
- **Conservative {asym, pt}**: 낮은 Sharpe + **낮은 MDD** + 적은 거래 (~90), Calmar 1·2위
- 그룹 내 Cohen's d ≈ 0.15~0.29 / 그룹 간 Cohen's d > 0.79

#### Pairwise Cohen's d (best Sharpe)

|  | sym | asym | dsr | pt |
|---|---|---|---|---|
| sym  | — | +1.10 | +0.29 | +1.19 |
| asym | -1.10 | — | -0.79 | +0.15 |
| dsr  | -0.29 | +0.79 | — | +0.89 |
| pt   | -1.19 | -0.15 | -0.89 | — |

#### 본 논문 §5 메인 결론 (시나리오 D — 사전 A/B/C 분기에 추가)

> **Reward variant 의 영향은 단일 metric (Sharpe) 의 alpha source 가 아니라, risk profile dimension 의 trade-off 로 나타난다. 4 variant 는 두 cluster (aggressive: sym, dsr / conservative: asym, pt) 로 통계적으로 분리되며, Sharpe-MDD 평면에서 Pareto-like frontier 를 형성한다.**

#### 가설 H1~H4 점검

- **H1** (sym ≈ ATR): 부정 — sym best Sharpe 1.871 >> ATR 1.505
- **H2 weak** (asym/dsr/pt > ATR): 지지 — 4 모두 ATR 초과
- **H2 strong** (asym/dsr/pt > sym): **부분 부정** — dsr ≈ sym, asym/pt < sym
- **H3** (selective entry → conservative): 지지 — asym/pt 거래 횟수 sym 대비 ~75%
- **H4** (CPCV+Slippage 유지): exp033/034 에서 검증 예정

#### exp027_rl 사전 증거와의 정합성

- Env-v3 (4D 절대 gap, asym β=2.0): Test Sharpe 1.955 (ATR 0.935 대비 **+109%**)
- Env-v4 (2D ATR 비례, asym β=3.42): Val Sharpe 1.681 (ATR 1.505 대비 **+12%**, sym 보다 낮음)

→ **환경 의존성 (4D → 2D) 효과가 reward variant 효과보다 큼.** 본 논문 §8 Discussion 에서 정직한 인정.

### exp032c — Mechanism Analysis (완료, 2026-05-15) — §6 메인

**5 메뉴 분석** on 1.04M step trajectories (40 모델 × ~26k val steps):

**Menu 1 — Pareto scatter**: 40 RL runs 가 Sharpe-MDD 평면에서 **두 cluster 형성, 5 개 Pareto frontier**. ATR (9.83, 1.505) 은 완전 dominated. → **§5 메인 figure 1순위**

**Menu 5 — Policy distance matrix** (가장 강력한 정량적 발견):

|  | sym | asym | dsr | pt |
|---|---|---|---|---|
| sym  | 0.000 | 0.209 | **0.123** | 0.326 |
| asym | 0.209 | 0.000 | 0.256 | **0.134** |
| dsr  | 0.123 | 0.256 | 0.000 | 0.353 |
| pt   | 0.326 | 0.134 | 0.353 | 0.000 |

- within-cluster mean: **0.129**
- across-cluster mean: **0.286**
- **Ratio: 2.22×** → 정책 수준 cluster 분리의 통계적 정량 입증.

**Menu 4 — Behavior per regime** (H3 강한 지지):

| Variant | Trade rate (high_vol) | **Hold rate (high_vol)** |
|---|---|---|
| sym  | 0.047 | 0.048 |
| **dsr**  | 0.047 | **0.120** |
| asym | 0.038 | 0.023 |
| pt   | 0.032 | 0.020 |

- Trade rate: aggressive > conservative (모든 regime) — **H3 selective entry 명확**
- **DSR hold rate 가 2~6배** — reward 형식 ↔ 행동 인과 직접 증거 (DSR window 가 짧은 hold 에서 noise 큼 → 정책이 더 오래 holding 학습)

### §6 메인 결론 (확정)

> **두 cluster 분리는 reward 의 손실 비대칭 (asym β, pt λ) 이 정책의 거래 빈도를 직접 결정한 결과 (H3 강한 지지). DSR 의 hold rate 우위는 sliding window risk-adjusted return 의 메모리 구조가 정책의 holding 시간을 늘린 결과 — reward 형식 → 행동 → 결과 의 메커니즘 인과 사슬 정량 확인.**

### exp033 — Slippage 0.02% Robustness (완료, 2026-05-15) — §7.1

**40 runs × 1M = 40M steps** with slippage_rate=0.0002, 3h 47min.

| Variant | exp033 Sharpe ± std | vs exp032b | Cohen's d | Slippage retention | vs ATR 1.505 |
|---|---|---|---|---|---|
| sym  | 1.658 ± 0.30 | -0.21 | -0.80 | 88.6% | +10% |
| dsr  | 1.551 ± 0.26 | -0.26 | -1.10 | 85.7% | +3% |
| asym | 1.478 ± 0.10 | -0.20 | -2.01 | 87.9% | -2% |
| pt   | 1.459 ± 0.10 | -0.21 | -2.18 | 87.6% | -3% |

**Cluster preservation 정량 입증**: within-cluster |d| 0.288 vs across-cluster |d| 0.630 → **ratio 2.19×** (exp032b 2.22× 와 거의 동일).

MDD 거의 변화 없음 (Δ +0.01~+0.55%). Slippage 가 거래 패턴은 안 바꾸고 마진만 깎음.

#### §7.1 메인 결론

> Slippage 0.02% 도입 후 모든 variant 가 ~12% Sharpe 감쇠, 단 **cluster 구조는 정량적으로 보존** (ratio 2.19×). Conservative cluster (asym, pt) 가 ATR-no-slippage 아래로 떨어져 H2 weak 약화 (ATR-with-slippage 재평가 필요).

### exp034 — CPCV 6-fold + DSR (완료, 2026-05-16) — §7.2

**60 runs × 1M**, 5h 27min. 6 groups × C(6,2)=15 paths, purge ±168h, seed=42.

| Variant | SR mean ± std (15 paths) | IQM | 5% CVaR | t-stat | p (one-sided) |
|---|---|---|---|---|---|
| sym  | 1.302 ± 0.48 | 1.282 | 0.579 | 10.61 | <0.001 |
| **dsr**  | **1.413 ± 0.38** | **1.433** | **0.890** | **14.49** | <0.001 |
| asym | 1.043 ± 0.47 | 0.954 | 0.503 | 8.52 | <0.001 |
| pt   | 1.093 ± 0.51 | 1.010 | 0.503 | 8.29 | <0.001 |

**4 variant 모두 다중검정 보정 후 p < 0.001** (진짜 알파). **DSR reversal 우위**: SR mean 1.413 (1위), std 0.378 (가장 낮음), 5% CVaR 0.890 (가장 높음).

#### Cluster preservation 강화

| 실험 | within \|d\| | across \|d\| | Ratio |
|---|---|---|---|
| exp032b | 0.30 | 0.79 | — |
| exp033 | 0.288 | 0.630 | 2.19× |
| **exp034 (CPCV)** | **0.179** | **0.636** | **3.55×** |

→ CPCV 환경에서 cluster 구분 **더 또렷** (다양한 시간 split → within-cluster 안정성 ↑).

#### §7.2 메인 결론

> CPCV 15 paths 평가에서 4 variant 모두 mean Sharpe > 1.0, DSR p < 0.001. **DSR variant 가 1위로 reversal** (1.413 vs single-split sym 1.871). 평가 방법 (single vs multi-split) 에 따라 winner 가 바뀌며, multi-split 환경에서는 DSR 의 sliding window formulation 이 가장 robust.

### exp035 — Test 봉인 해제 (완료, 2026-05-16) — §7.3 Final OOS

**100 RL 모델 + ATR baseline Test 2024+ 평가** (20,189 rows, BTC $42K→$75K bull market).

#### ATR Baseline Test: **Sharpe -0.055** (Val 1.505 → -1.56 gap)

#### RL Variants Test (양 source 일관)

| Variant | exp032b Test (n=10) | exp034 Test (n=15) | Val→Test gap (smallest) |
|---|---|---|---|
| sym  | 0.090 (p=0.015) | 0.001 (p=0.495) | -1.78 / -1.30 |
| asym | 0.173 (p=0.011) | 0.175 (p=0.009) | -1.51 / -0.87 |
| dsr  | **-0.122** (p=0.96) ⚠️ | 0.070 (p=0.15) | -1.93 / -1.34 |
| **pt** ⭐ | **0.367** (p=0.0015) | **0.339** (p=0.0004) | **-1.30 / -0.75 (best)** |

#### 🎯 세 개의 reversal — 평가 환경별 winner

| 평가 환경 | 1위 | Sharpe |
|---|---|---|
| Val (single-split) | sym | 1.871 |
| Multi-split CPCV | dsr | 1.413 |
| **Test (OOS)** | **pt** | **0.367 / 0.339** |

→ "단일 winner" 결론 평가 환경 의존. 3 환경 3 winner.

#### §7.3 메인 결론

> Test 2024+ OOS 에서 모든 시스템 (RL + ATR) ~1.5 Sharpe 감쇠 (BTC bull market 환경 grid trading 불리). 단 **prospect-theoretic reward (pt) 가 Test 1위, two sources 일관 (p<0.002), Val→Test gap smallest**. Loss aversion (λ=3.30) + concave gain (α=0.68) 의 정책이 unknown regime 에 robust. CPCV 1위였던 DSR 은 Test 꼴찌 — sliding window formulation 의 distribution shift 취약점 발견.

### 가설 H1~H4 최종 (Test 포함)

| 가설 | 최종 판단 |
|---|---|
| H1 (sym ≈ ATR) | 평가 의존 — Val sym >> ATR, Test 둘 다 near zero |
| H2 weak (variant > ATR) | **부분 지지** — Val/CPCV 강, Test 에선 pt/asym 만 |
| H2 strong (variant > sym) | **재해석** — 평가 환경 의존 (Val sym/CPCV dsr/Test pt) |
| H3 (selective entry) | 지지 (Menu 4/5 of exp032c) |
| H4 (Robustness) | **부분 지지** — slippage/CPCV cluster 유지, Test cluster 약화 |
| **H5 (사후, pt OOS robust)** | **강한 지지** (양 source p<0.002) |

### 본 논문 메인 thesis (확정, exp035 후)

> **Reward variant 의 영향은 risk profile dimension 의 trade-off (Val: Pareto frontier) + 평가 환경별 winner reversal (Val sym / CPCV dsr / Test pt) 로 나타난다. In-sample 우위 (DSR CPCV 1위) 가 OOS 에서 가장 큰 generalization 손실을 보이고, prospect-theoretic reward 가 unseen market regime 에 가장 robust — Kahneman-Tversky (1979) loss aversion 의 RL OOS 안전성 정량 확인.**

### 진행 예정

| Exp | 목적 | 결과 | 논문 챕터 |
|---|---|---|---|
| exp030 | PPO 학습 안정화 | (완료) | §3.3 |
| exp031 | BC warm-start | (완료, 폐기) | §3.4 |
| exp032a | Variant reward hyperparameter 튜닝 | (완료) | §3.5 |
| exp032b | Full 4 variant 비교 | (완료, 시나리오 D) | §5 Pareto frontier |
| exp032c | Mechanism analysis | (완료) | §6 Mechanism |
| exp033 | Slippage 0.02% | (완료) | §7.1 Robustness |
| exp034 | CPCV 6-fold + DSR | (완료) | §7.2 |
| exp035 | Test 봉인 해제 | **(완료, 위)** | **§7.3 Final OOS** |
| **Phase 15** | **ATR bootstrap + publication figures** | **(다음)** | **분석/논문 자료** |
| **Phase 16** | **논문 작성** | (대기) | §1~§9 본문 |

### exp032b 예상 결과 (시나리오)

본 논문은 결과가 어느 시나리오든 작성 가능 ([PROJECT_GOAL.md](PROJECT_GOAL.md) 참조):

- **A 낙관**: asym/pt 가 Cohen's d ≥ 0.5 로 sym보다 우위 → "Reward design이 RL 알파 핵심 채널" (strong positive)
- **B 중립**: variant 차이 있으나 ATR에는 못 미침 → "ATR 비례 공식이 강한 흡수력 보유" (interesting partial)
- **C 비관**: variant 간 차이 미미 → "Reward 형식 변형은 ATR 비례 공식 안에서 알파를 추가하지 못함" (negative 확장)

---

## 인용 가능한 핵심 수치 (논문 abstract / 결론용)

> 아래는 시나리오 A 가정. 실제 exp032b 결과 나오면 갱신.

- **Phase 1 발견**: "RL이 5D state로 학습한 정책이 Fixed [1.0, 0.0] (deterministic constant) 과 Val Sharpe 45.390 동일." (§4)
- **Phase 2 발견 (사전 증거)**: "Asymmetric reward (β=2.0) 하에서 RL이 ATR을 Test Sharpe 1.955 vs 0.935 (2.09×) 초과, MDD 0.39% vs 2.43% (0.16×), 거래 횟수 214 vs 1,591 (0.13×)." (exp027_rl)
- **Bayesian 효과**: "ATR 계수 단독 Bayesian 최적화 (Trial #42) 로 Val Sharpe 17.579 → 42.997 (+144.6%)." (exp013b → 50 trials)
- **체결가 정합성**: "next_low/next_high 체결의 favorable bias로 수조% return artifact 발생. 지정가 체결로 수정 후 Val Sharpe 60+ → 1~2 수준으로 현실화 (exp026 fix)."

→ 실제 exp032 결과 나오면 표 + abstract 갱신.

---

## 본 문서 사용 방식

1. **논문 작성 시**: 챕터별로 본 문서의 해당 섹션을 표/수치 그대로 인용
2. **디펜스 준비 시**: 핵심 수치 (Sharpe, MDD, 거래 횟수) 를 본 문서에서 한 번에 확인
3. **실험 결과 추가 시**: 각 Phase 섹션 갱신 (해당 exp 완료 직후)
4. **상세 history 필요 시**: [`RESEARCH_LOG.md`](../RESEARCH_LOG.md) 참조

## 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-05-14 | 본 문서 신설. Phase 1~2 결과 종합 |
| 2026-05-15 | exp032a 결과 추가 (4 reward variant Optuna 튜닝): dsr 1.89 / pt 1.80 / asym 1.52 (200k single-seed) |
| 2026-05-15 | **exp032b 결과 추가** (4 variant × 10 seeds × 1M, §5 메인): 시나리오 D — Pareto frontier in risk space, sym 1.87 / dsr 1.81 / asym 1.68 / pt 1.67 |
| 2026-05-15 | **exp032c 결과 추가** (Mechanism Analysis, §6 메인): Policy distance ratio 2.22× (cluster 통계적 분리 입증), DSR hold rate 2~6×, H3 강한 지지 |
| 2026-05-15 | **exp033 결과 추가** (Slippage 0.02% Robustness, §7.1): Cluster preservation ratio 2.19× ≈ exp032b 의 2.22×. 일률적 ~12% Sharpe 감쇠. asym/pt 가 ATR-no-slippage 아래로 marginal 하락. |
| 2026-05-16 | **exp034 결과 추가** (CPCV 6-fold + DSR, §7.2): 4 variant 모두 p < 0.001 진짜 알파. DSR variant reversal 우위 (mean 1.413, CVaR 0.890). Cluster ratio 3.55× (더 또렷). |
| 2026-05-16 | **exp035 결과 추가** (Test 봉인 해제, §7.3 Final OOS): ATR Test -0.055, pt 가 Test 1위 (0.367/0.339, p<0.002), 양 source 일관. DSR Test 꼴찌 (-0.122). 세 환경 세 winner reversal. H5 (pt OOS robust) 사후 발견. |
| 2026-05-16 | **Phase 15 완료**: Bootstrap P(RL>ATR) per (variant, env) + Val/Test distribution shift (KS p<1e-10, variance -27%) + three-env 종합 figure (abstract figure). 본 논문 모든 실험·분석 완료. |
| 2026-05-16 | **Phase 16a-d 완료**: ATR-with-slippage 재평가 (Val 0.835, exp033 RL 모두 초과), B&H Test 0.757 (정직 인정, MDD 50% caveat), Test trajectory 800K rows, **mechanism: DSR 7-day hold = OOS 실패 vs pt 6h hold = OOS robust**. |
