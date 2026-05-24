# 프로젝트 목표 (졸업 논문)

> 작성: 2026-05-14. 본 문서가 프로젝트의 단일 기준점이다.
> ROADMAP, CLAUDE.md, RESEARCH_LOG, project_continuation_plan은 모두 이 문서에 정렬되어야 한다.

---

## 핵심 연구 질문 (RQ)

> **BTC 그리드 트레이딩에서 reward 설계가 RL 정책의 알파에 어떤 영향을 미치는가?
> 특히, 어떤 reward 함수 하에서 RL이 ATR 규칙 기반을 초과하며 (혹은 초과하지 못하며), 그 메커니즘은 무엇인가?**

부 질문:
- **RQ-1 (reproduction of negative finding)**: Symmetric (단순 equity-change) reward + ATR 비례 공식 하에서 RL이 정말로 ATR과 동등한가? (exp020~022 재현)
- **RQ-2 (reward variant comparison)**: 어떤 reward 변형 (Asymmetric / DSR / Prospect-theoretic) 이 RL의 ATR 대비 우위 또는 비우위를 만드는가?
- **RQ-3 (mechanism)**: 우위가 발견된다면, 해당 reward가 정책 행동을 어떻게 변화시키는가? (Regime별 행동 분포, 사이클 통계)
- **RQ-4 (robustness)**: 그 결과(우위든 비우위든)가 CPCV + Slippage + 다중 seed에서도 유지되는가?

> ⚠️ **RQ 표현 원칙**: RQ는 열린 질문 형태로 유지한다. 사전 증거가 강하더라도(exp027_rl 등) 그 증거는 **가설(H1~H4)** 의 정당성으로 흡수하지 RQ의 답으로 단정하지 않는다.
> Thesis statement ("Reward 설계가 RL 알파의 핵심 채널이다") 는 결과가 가설을 지지할 때 §5 Positive finding과 abstract/conclusion에서 사용한다.

---

## 자산 범위

**BTC/USDT 1시간봉 단일 자산**.

자산 확장은 본 졸업 논문 범위에서 **명시적으로 제외**한다.
- 이유: BTC 하나만으로도 reward design 가설을 견고히 검증하기에 충분.
- 자산 확장은 사용자 개인 운용 차원에서 별도 진행 가능. 단, 논문에는 포함하지 않음.
- Discussion 섹션에서 "확장 시사점"으로만 짧게 언급.

---

## 핵심 가설

**H1**: Symmetric reward + ATR 비례 공식의 조합에서, RL의 자유도(state 인식 + 적응 행동)는 ATR이 이미 흡수한 변동성 차원의 알파를 넘어서지 못한다.

**H2**: 손실에 비대칭 가중을 주거나(Asymmetric, Prospect-theoretic), 위험조정수익을 직접 reward로 정의하면(DSR), RL이 ATR의 한계를 초과한다.

**H3**: 그 초과는 행동 측면에서 "선택적 진입" (낮은 거래 빈도 + 높은 사이클 승률)으로 나타난다.

**근거 (사전 증거)**:
- exp020/021/022: Symmetric reward에서 RL ≈ ATR (Test Sharpe 42.0 vs 41.8, exp026 수정 후 ATR > RL)
- exp027_rl: Asymmetric reward (beta=2.0)로 Test Sharpe 1.955, ATR(0.935) 대비 2.1배 + MDD 0.39% + 거래 214건 (ATR 1591건 대비 1/7)

→ H1, H2, H3는 사전 부분 증거가 있으나 정식 검증 필요.

---

## 결과 시나리오 분기 (2026-05-15 갱신, 시나리오 D 확정)

본 논문은 결과가 아래 어느 시나리오로 나오든 작성 가능한 RQ 구조를 채택. 사전에 4개 시나리오를 명시 등재하여 ex-post rationalization 을 회피한다.

### 사전 분기 (검증 전 — 2026-05-14)

- **시나리오 A (낙관, strong positive)**: asym/dsr/pt 가 sym 과 ATR 모두를 Cohen's d ≥ 0.5 로 유의하게 초과. → "Reward design이 RL 알파의 핵심 채널" thesis 직접 지지.
- **시나리오 B (중립, interesting partial)**: variant 간 차이는 있으나 ATR 에는 못 미침. → "ATR 비례 공식이 강한 변동성 흡수력 보유" — Negative finding 확장.
- **시나리오 C (비관, negative 확장)**: variant 간 차이 미미. → "Reward 형식 변형은 ATR 비례 공식 안에서 알파를 추가하지 못함" — Phase 1 발견의 강화.

### 사후 분기 추가 (검증 후 — 2026-05-15, exp032b 본 검증)

- **시나리오 D (다차원 trade-off, Pareto frontier)**: 4 variant 가 두 cluster {aggressive: sym, dsr / conservative: asym, pt} 로 통계적 분리. Sharpe/MDD/Calmar/Return 1위가 모두 다름. 단일 winner 없음. → "Reward design 의 영향은 단일 Sharpe alpha source 가 아니라 **risk profile dimension 의 trade-off**" — nuanced positive contribution.

### 현재 확정 시나리오: **D**

**근거** (exp032b, 40 runs × 1M steps, 2026-05-15):

| Variant | Best Sharpe | MDD% | Calmar | Cluster |
|---|---|---|---|---|
| sym  | 1.871 ± 0.22 | 3.27 | 0.60 | aggressive |
| dsr  | 1.809 ± 0.21 | 4.33 | 0.47 | aggressive |
| asym | 1.681 ± 0.10 | 2.28 | 0.755 | conservative |
| pt   | 1.667 ± 0.09 | 2.31 | 0.735 | conservative |
| ATR  | 1.505 | 9.83 | 0.153 | — |

- 4 variant 모두 ATR 을 +11~24% 초과 (H2 weak form 지지)
- sym 이 Sharpe 1위 → H2 strong form ("variant > sym") **부분 부정**
- 그룹 내 |Cohen's d| < 0.30 / 그룹 간 > 0.79 → 두 cluster 통계적 분리

**가설 H1~H4 점검**:
| 가설 | 결과 | 근거 |
|---|---|---|
| H1 (sym ≈ ATR) | 부정 | sym best 1.871 >> ATR 1.505 |
| H2 weak (variant > ATR) | 지지 | 4 모두 ATR 초과 |
| H2 strong (variant > sym) | 부분 부정 | dsr ≈ sym, asym/pt < sym |
| H3 (selective entry) | 지지 | asym/pt 거래 ~75% of sym |
| H4 (CPCV+Slippage robust) | 검증 예정 | exp033/034 |

**exp027_rl 사전 증거의 환경 의존성 인정**:
- Env-v3 (4D 절대 gap, asym β=2.0): Test Sharpe 1.955 (+109% vs ATR)
- Env-v4 (2D ATR 비례, asym β=3.42): Val Sharpe 1.681 (+12% vs ATR, sym 보다 낮음)
- → **환경 의존성 효과가 reward variant 효과보다 큼**. 본 논문 §8 Discussion 에서 정직한 인정.

### Frame 결정

본 논문 §5 메인 thesis (시나리오 D 위에서):

> "**Reward variant 의 영향은 단일 metric (Sharpe) 의 alpha source 가 아니라, risk profile dimension 의 trade-off 로 나타난다. 4 variant 는 두 cluster (aggressive vs conservative) 로 통계적으로 분리되며, Sharpe-MDD 평면에서 Pareto-like frontier 를 형성한다. 단순 'X reward 가 best alpha' 결론보다 multi-dimensional trade-off 의 정직한 정량화가 본 논문의 contribution 이다.**"

→ §5 는 "Pareto frontier discovery" 로, §6 (exp032c mechanism) 은 "왜 두 cluster?" 메커니즘 정량화로, §8 Discussion 은 exp027_rl 의 환경 의존성 정직 인정으로 구성.

---

## 평가 방법론

### 메인 비교 단위
**Reward variant × 평가 path** 의 2D 매트릭스. variant별 분포 비교.

### Reward 변형 (Treatment)
| 코드 | 정식명 | 학술 출처 |
|---|---|---|
| `sym` | Symmetric equity change | Baseline (현재 r_step) |
| `asym` | Asymmetric (loss penalty β) | exp027_rl, Prospect Theory 단순화 |
| `dsr` | Differential Sharpe Ratio | Moody & Saffell (2001) |
| `pt` | Prospect-theoretic (α, λ) | Kahneman & Tversky (1979) |

### Baseline (Treatment 비교 대상)
- **ATR 고정 (exp026 best)**: Bayesian 최적화된 ATR 비례 공식. 본 논문의 1차 비교 baseline.
- **Buy-and-Hold**: 절대 기준선.
- **Fixed Grid 1%/2%/5%**, **ATR Grid k=0.5/1.0/2.0**: 단순 규칙 베이스라인.

### 평가 metric
- **1차**: Sharpe Ratio (연율화, √8760)
- **2차**: MDD, Calmar, Sortino, 거래 횟수, 사이클 승률, 평균 사이클 PnL
- **분포 metric**: CPCV (Combinatorial Purged CV) 다중 path의 Sharpe 분포 (mean, std, 5% CVaR)
- **통계 검증**: DSR (Deflated Sharpe Ratio, López de Prado 2014) — 다중검정 보정 후 진짜 알파인지

### 검증 단계
1. **Single split (현 방식)**: Train 2017-2022 / Val 2023 / Test 2024+
2. **CPCV (메인 검증)**: Train+Val 안에서 6-fold → 15 paths Sharpe 분포 (Test 봉인 유지)
3. **Sim2Real**: Slippage 0.02% + Domain Randomization으로 학습 → 그 환경에서 평가
4. **Test set (최종, 1회만)**: CPCV 통과 후 봉인 해제

### Test set 봉인 원칙
- CLAUDE.md의 절대 금지 규칙 1 그대로 유지.
- exp035 단계에서만 봉인 해제.

---

## 논문 구조 (예상)

```
1. Introduction
   - Grid trading + RL의 자연스러운 만남
   - ATR 비례 공식의 변동성 흡수 가설
   - 본 논문의 contribution: reward design이 RL 알파의 핵심 채널

2. Background
   - Grid trading의 학술적 위치 (Avellaneda-Stoikov 2008)
   - DRL trading (Zhang Zohren Roberts 2020, Gort 2022)
   - Reward design 이론 (Moody 2001, Ng 1999, Kahneman 1979)

3. Method
   - 환경 설계 (MDP, 4-order grid, n_splits 사이클)
   - ATR 비례 공식 (Bayesian-optimized baseline)
   - RL 알고리즘 (PPO + BC warm-start)
   - 4가지 Reward 변형

4. Negative finding (RQ-1)
   - Symmetric reward 하에서 RL ≈ ATR (exp020~022, exp026 재실험)

5. Positive finding (RQ-2) — 메인 챕터
   - 4가지 reward 비교 (exp032 메인 결과)
   - CPCV 분포 (exp034)
   - DSR 통계 검증

6. Mechanism (RQ-3)
   - Regime별 행동 분석 (Bull/Bear/Sideways)
   - 사이클 통계 비교 (거래 빈도, 승률, PnL)
   - 정책 포화 vs 선택적 진입의 행동 패턴

7. Robustness (RQ-4)
   - Slippage + Domain Randomization (exp033)
   - Test set 평가 (exp035)
   - 한계 (BTC 단일 자산, 1시간봉, 호가창 미모델링)

8. Discussion
   - Reward design이 알파 채널인 이유
   - 자산 확장 시사점 (실제 확장은 본 논문 외)
   - 라이브 트레이딩 (Sim2Real gap)

9. Conclusion
```

---

## 실험 시리즈 매핑

| Exp | 논문 챕터 | 목적 |
|---|---|---|
| exp030 | Method (3.3 학습 안정화) | PPO 학습 안정화 패키지 |
| exp031 | Method (3.4 출발선 보장) | BC warm-start로 학습 초반 낭비 회피 |
| exp031b | Method (조건부) | exp031 부족 시 CQL + mixed dataset |
| **exp032a** | **Method (3.5 공정 비교)** | **Variant별 reward hyperparameter 튜닝** |
| **exp032b** | **Positive finding (5장 메인)** | **4 variant full 비교 + effect size** |
| **exp032c** | **Mechanism (6장)** | **counterfactual + SHAP + mediation** |
| exp033 | Robustness (7.1) | Slippage + Domain Randomization |
| exp034 | Positive finding (5장) + Robustness (7.2) | CPCV 분포 + DSR |
| exp035 | Robustness (7.3 최종 검증) | Test set 봉인 해제 |

기존 exp001~029의 결과는 RQ-1의 negative finding 근거 + RQ-2의 가설 시발점으로 활용.
이미 한 작업이지 새로 할 필요 없음. RESEARCH_LOG에 정리되어 있음.

---

## 의식적으로 제외하는 것 (Scope discipline)

본 문서가 명시한 RQ에서 벗어나는 모든 작업을 **본 논문 범위 외**로 처리.

- ❌ 다른 자산군 (주식, FX, 원자재) — 사용자 개인 운용으로 분리
- ❌ Hierarchical RL, Transformer state encoder — Discussion에서만 짧게 시사
- ❌ Multi-asset portfolio optimization
- ❌ Tick-level microstructure (호가창 큐)
- ❌ HFT, market making 실거래
- △ Live trading (Paper trading은 sim2real gap 측정 도구로 제한적 활용)

---

## 본 문서의 사용 방식

1. **새 작업 시작 전 본 문서 확인** — RQ에 부합하는가? Scope 안인가?
2. **불일치 발견 시 즉시 정리** — 다른 문서에 RQ와 모순되는 내용 있으면 그 문서를 본 문서에 정렬.
3. **본 문서 자체 수정은 신중히** — RQ나 가설을 바꾸려면 사용자와 명시적 합의 필요.

---

## 변경 이력

| 날짜 | 변경 | 근거 |
|---|---|---|
| 2026-05-14 (1차) | 본 문서 신설. RQ를 "RL이 고정 그리드 대비 우위?"에서 "Reward design이 RL 알파의 핵심 채널?" (단정문) 로 전환. 자산 확장 제외 결정. | 사용자와 합의 (2026-05-14 대화) |
| 2026-05-14 (2차) | RQ를 단정문 → **열린 질문 형태로 수정**. 사전 증거(exp027_rl)는 가설 H1~H4의 정당성으로 유지하되, RQ 자체는 검증 결과에 따라 어느 쪽이든 살아남는 형태로. | 학술 컨벤션 준수 + 확증 편향 회피. 사용자 지적으로 수정. |
| 2026-05-15 | **결과 시나리오 분기 섹션 신설**. 사전 시나리오 A/B/C 등재 + exp032b 본 검증 결과 시나리오 D (Pareto frontier) 사후 추가 + 현재 확정 시나리오 = D. 본 논문 §5 메인 thesis 갱신. | exp032b 결과 (40 runs × 1M, 두 cluster 발견). 시나리오 A 사전 증거 (exp027_rl Env-v3) 의 환경 의존성 정직 인정. |
