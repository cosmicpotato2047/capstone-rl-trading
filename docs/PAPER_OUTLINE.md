# 졸업 논문 골격 (PAPER_OUTLINE)

> 작성: 2026-05-16. **본 문서의 목적**: 본문 작성 시 nav 역할.
> 단위는 sentence 가 아니라 chapter-level main claim + figure/table mapping + reference pointer.
> 본문 작성 중 변경되는 부분은 outline 도 동기 갱신.

---

## 본 논문 메인 thesis (1 sentence)

> **Reward variant 의 영향은 risk profile dimension 의 trade-off 로 나타나며 (Pareto frontier), 평가 환경별로 winner 가 reversal (Val sym → CPCV dsr → Test pt). Prospect-theoretic reward (loss aversion λ=3.30) 가 unseen market regime 에 가장 robust — Kahneman-Tversky (1979) 의 RL OOS 안전성 정량 확인.**

## Abstract 핵심 수치 (4-5 줄)

- exp032b Val: 4 variants × 10 seeds, two clusters, ATR 1.505 대비 +11~24%
- exp032c Mechanism: policy distance ratio 2.22×
- exp034 CPCV: DSR p<0.001 reversal 1위 (1.413)
- exp035 Test OOS: pt 가 두 source 일관 1위 (0.367/0.339, p<0.002), DSR 꼴찌 (-0.122)
- Distribution shift: Val→Test KS p<1e-10, variance -27%

## Abstract figure

`reports/phase15_figures/menu_c_three_env.png` (4-panel boxplot, three environments)

---

## §1 Introduction (~2 페이지)

**Main claim**: BTC 그리드 트레이딩에서 reward design 의 영향을 정량 검증하며, 평가 환경에 따른 winner reversal + Prospect-theoretic reward 의 OOS robust 를 발견.

**핵심 contribution 3개**:
1. 4-variant 비교의 통계적으로 엄밀한 검증 (10 seeds + Cohen's d + Bootstrap)
2. 시나리오 D (Pareto frontier) 및 winner reversal 발견
3. Prospect-theoretic reward 의 OOS robust 정량 (H5, 양 source 일관 p<0.002)

**Figure**: phase15 menu_c (abstract figure 후보 또는 §1 후반부)

**Reference**: Avellaneda-Stoikov (2008), Zhang-Zohren-Roberts (2020), Gort et al. (2022)

**결론**: "본 논문은 reward design 의 효과가 단일 metric winner 가 아닌 risk profile trade-off + 평가환경 의존성으로 나타남을 정량 보인다."

---

## §2 Background (~3 페이지)

**Main claim**: 본 연구의 학술 토대 (grid trading + DRL + reward design + 평가 방법론).

**서브 섹션**:
- §2.1 Grid trading & market making (Avellaneda-Stoikov 2008)
- §2.2 DRL in trading (Zhang-Zohren-Roberts 2020, Gort 2022)
- §2.3 Reward design 이론
  - DSR (Moody-Saffell 2001)
  - Prospect Theory (Kahneman-Tversky 1979)
  - Reward shaping (Ng et al. 1999)
- §2.4 Evaluation methodology
  - CPCV (López de Prado 2018)
  - Deflated Sharpe Ratio (López de Prado 2014)
  - RL reproducibility (Henderson et al. 2018)

**Figure**: 없음 (또는 reward variant 4종 수식 비교 table)

**Table**: Reward variant 4종 수식 + hyperparameter table

---

## §3 Method (~5 페이지)

**Main claim**: Env-v4 (2D ATR 비례 + 지정가 체결) 위에서 PPO + 4 reward variant.

**서브 섹션**:
- §3.1 MDP 설계 (State 7D, Action 2D, Reward 4 variants)
- §3.2 Trading environment 구체 (ATR 비례 공식, 사이클 정의, fee/slippage)
- §3.3 ATR baseline (Bayesian Trial #34, formula_coefs)
- §3.4 PPO + 안정화 패키지 (target_kl, ent annealing, LR linear schedule) — exp030 결과 인용
- §3.5 4 reward variant 설계 + Optuna 튜닝 (exp032a)
- §3.6 평가 방법론: single-split Val, CPCV 6-fold, Test 봉인

**Figure**: 없음 또는 MDP 도식

**Table**:
- Reward variant 4종 수식 + Optuna best hyperparameter
- ATR baseline 의 Bayesian best coefs
- PPO 안정화 패키지 hyperparameter

**Reference**: Schulman et al. (2017, PPO), Moody-Saffell (2001), Kahneman-Tversky (1979), López de Prado (2018)

---

## §4 Phase 1 Negative Finding (~2 페이지)

**Main claim**: Symmetric reward + ATR 비례 공식 조합에서 RL 의 자유도가 흡수됨 (exp020-016).

**내용**:
- exp020/021/022 결과 (Env-v2 시기): "RL = Fixed [1.0, 0.0]" with Val Sharpe 45.390 일치
- Decisive ablation: Fixed [1.0, 0.0] = exp020 best
- Phase 1 발견의 동기 → reward design 정식화 필요

**Figure**: 없음 또는 §5 의 cluster figure 의 context

**Reference**: Henderson et al. (2018) for RL determinism

**결론**: "기존 reward (symmetric) + ATR 비례 공식 조합으로는 RL 알파가 한정적. 본 논문은 reward 정식화로 해결 가능성 검증."

---

## §5 Main Finding — Pareto Frontier Discovery (~6 페이지)

**Main claim**: 4 reward variant 가 단일 winner 가 아닌 risk profile dimension 의 trade-off 로 나타남 (시나리오 D).

**내용**:
- exp032b: 4 variants × 10 seeds × 1M = 40 runs 결과
- Per-variant Sharpe / MDD / Calmar / Trades / Return table
- Pairwise Cohen's d + P(A>B) bootstrap (4×4)
- 두 cluster 발견: {sym, dsr} aggressive vs {asym, pt} conservative
- 시나리오 분기 (사전 A/B/C + 사후 D)
- 가설 H1, H2 점검

**Figure 1** (메인): `reports/exp032c_figures/menu1_pareto_scatter.png` — 40 runs Sharpe-MDD plane
**Figure 2** (보조): `reports/exp032c_figures/menu5_policy_distance.png` — 4×4 distance heatmap

**Table**:
- 4 variants × {best Sharpe, final Sharpe, MDD, Calmar, Trades, Return}
- Pairwise Cohen's d / P(A>B)

**Reference**: López de Prado (2014) for multiple testing, Henderson et al. (2018)

**결론**: "Reward variant 의 영향은 단일 Sharpe metric 의 alpha source 가 아니라 multi-dimensional risk profile trade-off 로 나타남."

---

## §6 Mechanism Quantification (~5 페이지)

**Main claim**: 두 cluster 분리는 reward 의 손실 비대칭이 정책의 거래 빈도 + holding 시간을 직접 결정한 결과.

**내용**:
- exp032c 5-menu 분석 결과 (1.04M step trajectories, 40 models on Val)
- Policy distance ratio 2.22× → cluster 분리 정량
- Action distribution per regime → variant 별 행동 차이
- Counterfactual action grid → 같은 state, 다른 reward → 다른 action
- Behavior stats per regime → trade rate, hold rate
- 핵심 인과: reward 손실 비대칭 → 거래 빈도 → cluster

**Figure 3** (메인): `reports/exp032c_figures/menu5_policy_distance.png` (또는 §5 이미 사용시 §6 다른 figure)
**Figure 4** (메인): `reports/exp032c_figures/menu4_behavior_per_regime.png` — trade/hold rate per vol regime
**Figure 5** (보조): `reports/exp032c_figures/menu2_action_distribution.png` — variant × regime grid

**Table**: regime 별 behavior stats (trade rate, hold rate, return) per variant

**Reference**: Moody-Saffell (2001) for DSR explanation, Kahneman-Tversky (1979) for loss aversion

**결론**: "Reward 형식 → 행동 → 결과 의 인과 사슬 정량 확인 (H3 강한 지지). DSR 의 hold rate 우위 = sliding window memory 의 정책 영향."

---

## §7 Robustness Trio

### §7.1 Slippage Robustness (~3 페이지)

**Main claim**: Slippage 0.02% 추가 후 cluster 구조 보존, 단 absolute alpha 일부 잠식.

**내용**:
- exp033: 4 variants × 10 seeds × 1M with slippage
- Per-variant Sharpe / MDD 비교 (exp032b vs exp033)
- Slippage 일률적 ~12% 감쇠
- Cluster preservation ratio 2.19× ≈ exp032b 의 2.22×
- ATR-with-slippage 재평가 (Phase 16a 후 추가)
- Conservative cluster (asym, pt) 의 ATR 비교 결과

**Figure 6**: `reports/exp033_figures/menu1_side_by_side.png` — exp032b vs exp033 bar chart

**Table**: 4 variants × {Sharpe, MDD, retention, Cohen's d vs exp032b}

**Reference**: Sim2Real gap in RL trading

**결론**: "Cluster 구조 robust under slippage. Absolute alpha 잠식은 conservative cluster 에서 크고, ATR-with-slippage 와 fair 비교 시 RL family 는 여전히 ATR 초과."

### §7.2 CPCV (~3 페이지)

**Main claim**: 6-fold CPCV (15 paths) 평가에서 DSR variant 가 reversal 1위, 4 variant 모두 p < 0.001 진짜 알파.

**내용**:
- exp034: 4 variants × 15 paths × 1 seed = 60 runs CPCV
- Per-variant Sharpe distribution (mean, std, IQM, CVaR)
- DSR p-value (Bonferroni 4-way 보정 후 모두 < 0.004)
- Cluster preservation ratio 3.55× (더 또렷)
- **Winner reversal**: Val sym 1.871 → CPCV dsr 1.413
- DSR 의 sliding window 가 multi-split robust 한 이유

**Figure 7** (메인): `reports/exp034_figures/menu3_boxplot.png` — 15-path Sharpe distribution
**Figure 8** (보조): exp034 menu2_heatmap.png — 4 × 15 path-by-path heatmap

**Table**: 4 variants × {SR mean ± std, IQM, 5% CVaR, t-stat, DSR p}

**Reference**: López de Prado (2018) CPCV, (2014) DSR

**결론**: "Multi-split 환경에서 DSR reversal 우위. 평가 방법 (single vs multi) 이 winner 를 결정 — single Sharpe metric 한계."

### §7.3 Final OOS — pt 의 robust ★ (~5 페이지, 본 논문 main contribution)

**Main claim**: Test 2024+ OOS 에서 모든 시스템 ~1.5 Sharpe 감쇠하나, **prospect-theoretic reward (pt) 가 Test 1위 (양 source 일관 p<0.002, Val→Test gap smallest)** — Kahneman-Tversky loss aversion 의 RL OOS 안전성 정량 확인. CPCV 1위 DSR 은 Test 꼴찌.

**내용**:
- exp035: 100 RL 모델 + ATR baseline + B&H (Phase 16b 후 추가) Test 평가
- ATR Test -0.055 + B&H Test (예정)
- Per-variant Test 통계 (mean, std, IQM, 5% CVaR, t-test p)
- Val vs Test gap per variant
- pt 의 Test 우위 메커니즘 (Phase 16d 후): trajectory 기반 정량 인과
- DSR 의 Test 실패 메커니즘 (Phase 16d 후): hold rate 양면성
- 세 환경 세 winner reversal: Val sym → CPCV dsr → Test pt

**Figure 9** (메인 ★): `reports/exp035_figures/menu2_val_vs_test.png` — Val vs Test gap per variant
**Figure 10** (메인 ★): `reports/exp035_figures/menu3_boxplot.png` — Test Sharpe distribution per variant per source
**Figure 11** (보조): pt mechanism figure (Phase 16d 산출)

**Table**:
- 4 variants × {Test Sharpe mean ± std, t-stat, p (one-sided), vs ATR}, 양 source
- Val→Test gap per variant
- Bootstrap P(RL > ATR) per variant per env (Phase 15a)

**Reference**: Kahneman-Tversky (1979), Henderson et al. (2018)

**결론**: "pt 의 OOS robust = loss aversion + concave gain function 이 unknown regime 에서 정책의 entry 를 선택적으로 만듦 (Phase 16d mechanism 결과). DSR 의 Test 실패 = sliding window 의 in-sample 우위가 distribution shift 에 취약."

---

## §8 Discussion (~4 페이지)

**Main claim**: Three-environment winner reversal 의 학술적 해석 + 본 논문 limitation.

**서브 섹션**:
- §8.1 Distribution shift quantification (Val vs Test KS p<1e-10, Phase 15b)
- §8.2 exp027_rl 사전 증거 (Env-v3 asym 1.955) 의 환경 의존성 정직 인정
- §8.3 Single-metric winner 의 한계 + multi-environment frame 추천
- §8.4 Reward design 의 RL OOS 안전성 시사 (pt = Kahneman-Tversky)
- §8.5 Limitation:
  - BTC 단일 자산 (multi-asset 미검증)
  - Test 가 BTC bull market 1 환경 (다른 regime 추가 검증 필요)
  - Slippage 0.02% 1 level (sensitivity 추가 가능)
  - DR (Domain Randomization) 미적용 (future work)
  - 학습 step 1M (더 긴 학습 효과 미검증)

**Figure 12**: `reports/phase15_figures/menu_b_distribution_shift.png` — Val vs Test KDE overlay

**Table**: Val vs Test distribution stats (mean, std, KS p, Wasserstein)

**Reference**: Henderson et al. (2018), 본 논문 신규 contribution

**결론**: "본 논문의 발견은 RL trading 의 'in-sample 우위 ≠ OOS robust' 의 정직한 실증. Reward 형식 선택은 OOS regime 가정에 따라 다름 — pt 가 가장 안전, DSR 가 in-sample 다양화에 강함, sym 가 단일 split 우위."

---

## §9 Conclusion (~1 페이지)

**4 메인 메시지**:
1. Reward variant 의 영향 = risk profile dimension trade-off (Pareto frontier)
2. 단일 metric winner 가 평가 환경별로 다름 (3 환경 3 winner reversal)
3. **Prospect Theory (pt) 가 unseen regime 에 OOS robust** (양 source 일관, p<0.002)
4. In-sample 다양화 (CPCV) 의 우위가 OOS 일관성을 보장 안 함

**Future work**:
- Multi-asset 확장
- 다른 regime (bear, sideways) 검증
- DR (Domain Randomization) 통합
- Continuous reward learning (meta-RL)

---

## Figure Inventory (mapping)

| # | 파일 | 사용 § | 메모 |
|---|---|---|---|
| 1 | `phase15_figures/menu_c_three_env.png` | abstract / §1 / §7.3 | **abstract figure**, 세 환경 winner reversal |
| 2 | `exp032c_figures/menu1_pareto_scatter.png` | §5 | **§5 메인**, Pareto scatter |
| 3 | `exp032c_figures/menu5_policy_distance.png` | §5 또는 §6 | cluster 정량 |
| 4 | `exp032c_figures/menu4_behavior_per_regime.png` | §6 | trade/hold rate per vol regime |
| 5 | `exp032c_figures/menu2_action_distribution.png` | §6 (보조) | variant × regime grid |
| 6 | `exp033_figures/menu1_side_by_side.png` | §7.1 | exp032b vs exp033 |
| 7 | `exp034_figures/menu3_boxplot.png` | §7.2 | CPCV 15-path distribution |
| 8 | `exp035_figures/menu2_val_vs_test.png` | §7.3 | **§7.3 메인 1**, Val vs Test gap |
| 9 | `exp035_figures/menu3_boxplot.png` | §7.3 | **§7.3 메인 2**, Test Sharpe |
| 10 | `phase15_figures/menu_b_distribution_shift.png` | §8 | Val vs Test KDE overlay |
| 11+ | (Phase 16d 산출) | §7.3 또는 §8 | pt OOS mechanism, DSR 실패 |

→ **총 10~12 figure**. 학부 졸업 논문 표준.

---

## Reference List (per §)

- §1: Avellaneda-Stoikov (2008), Zhang-Zohren-Roberts (2020)
- §2: 위 + Gort (2022), Moody-Saffell (2001), Kahneman-Tversky (1979), Ng (1999), López de Prado (2014, 2018), Henderson (2018), Schulman (2017)
- §3: Schulman (2017) PPO, Optuna 인용
- §4: Henderson (2018) reproducibility
- §5: 본 논문 신규
- §6: Moody-Saffell + Kahneman-Tversky (mechanism)
- §7.1: 본 논문 신규
- §7.2: López de Prado (2018, 2014)
- §7.3: Kahneman-Tversky (1979), Henderson (2018)
- §8: 위 + 본 논문 신규
- §9: 본 논문 신규

**총 reference**: 약 10~15개.

---

## 분량 비중 (페이지 X, 비중 %)

| § | 비중 | 메모 |
|---|---|---|
| §1 Intro | 7% | 본 논문 motivation + contribution |
| §2 Background | 10% | 학술 토대 |
| §3 Method | 17% | MDP + PPO + reward |
| §4 Phase 1 | 7% | Negative finding |
| §5 Pareto | 20% | 메인 §1 |
| §6 Mechanism | 17% | 메인 §2 |
| §7.1 Slippage | 7% | |
| §7.2 CPCV | 10% | |
| §7.3 Test ★ | **17%** | 메인 §3 (본 논문 진짜 contribution) |
| §8 Discussion | 13% | Distribution shift + limitation |
| §9 Conclusion | 3% | |

**총**: 약 30~50 페이지 (학부 졸업 논문 표준).

---

## 작성 시 주의사항

1. **세 환경 세 winner reversal** 의 frame 일관 유지 — §5에서 시나리오 D, §7에서 reversal, §9에서 메인 메시지
2. **H5 (pt OOS robust)** 가 본 논문의 진짜 contribution — abstract + §1 contribution + §7.3 + §9 에서 강조
3. **exp027_rl 사전 증거의 환경 의존성** 정직 인정 (§8.2) — academic integrity
4. **B&H Test baseline (Phase 16b)** 결과 나오면 §7.3 + §8.5 에 추가
5. **Limitation 솔직히 적기** — BTC 단일 자산, Test 1 regime, 단일 slippage level

---

## Phase 16 발견 반영 (§7.1/§7.3/§8 갱신)

### §7.1 추가 자료 (Phase 16a)
- ATR-with-slippage Val 0.835 vs RL Val (exp033): **4 variant 모두 0.835 초과** → 이전 caveat 청소, fair comparison 에서 모두 ATR 초과
- ⚠️ ATR Val no-slippage 1.378 (재측정값) ≠ 1.505 (RESEARCH_LOG 초기). §3.3 의 footnote 로 evaluation setup 차이 명시.

### §7.3 추가 자료 (Phase 16b)
- B&H Test Sharpe **0.757** > 모든 RL (절대 Sharpe)
- 하지만 B&H Test MDD **50.08%** vs RL pt MDD ~2.3% → Calmar 22× pt 우위
- 정직 frame: "절대 Sharpe 면 B&H, risk-adjusted 면 pt 압도"
- §7.3 + §8.5 Limitation 에 반영

### §7.3 메커니즘 답변 (Phase 16d) — 본 논문 진짜 contribution
- **DSR Test 실패**: median 1h, p95 20h, **max 169h (7일)** holding → bull market sell-side timing risk
- **pt Test robust**: mean 1.4h, max 6h holding → bull market 가격 movement 회피
- 정책 행동 Val/Test 거의 동일 (action delta ~5%) → 결과 차이는 **시장 distribution shift** (Phase 15 KS p<1e-10 정합)
- **인과 사슬**: reward formulation → hold duration → OOS regime 적응 (in-sample 우위 vs OOS robust trade-off)

### 추가 Figure (Phase 16d)
- **Figure 11 (★ 신규 메인)**: `reports/phase16d_figures/menu3_hold_duration.png` — DSR 의 7-day hold visualization
- Figure 12: `reports/phase16d_figures/menu1_test_behavior.png` — Test trade/hold rate per regime
- Figure 13: `reports/phase16d_figures/menu2_val_test_shift.png` — 정책 안정성 (Val/Test 거의 동일)
- Figure 14: `reports/phase16d_figures/menu4_action_test.png` — Test action distribution

→ **총 figure 13~14개**. 학부 졸업 논문 표준 (10~15) 안.

### §7.3 메인 thesis 갱신 (확정)

> Test 2024+ OOS 에서 **prospect-theoretic reward (pt) 가 두 source 일관 1위** (p<0.002). 메커니즘: pt 의 loss aversion (λ=3.30) + concave gain (α=0.68) 이 정책의 **hold duration 을 매우 짧게 (mean 1.4h, max 6h) 학습** 시켜 unseen bull market 의 sell-side timing risk 를 회피. 반면 **DSR 의 sliding window reward 가 long hold (mean 4.58h, max 7일) 를 학습**, in-sample CPCV 우위와 정확히 같은 메커니즘이 OOS 에선 단점으로 작용. **in-sample 우위와 OOS robust 가 reward formulation 의 동일 차원에서 trade-off**.

---

## 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-05-16 | 본 문서 신설. 9 chapter outline + figure mapping + reference list |
| 2026-05-16 | Phase 16a~d 결과 반영: ATR-with-slippage, B&H baseline, hold duration mechanism, pt OOS robust 인과 사슬 |
