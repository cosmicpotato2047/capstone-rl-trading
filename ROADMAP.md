# Project Roadmap
# BTC 그리드 트레이딩 — Reward Design이 RL 알파에 미치는 영향

**핵심 RQ**: BTC 그리드 트레이딩에서 reward 설계가 RL 정책의 알파에 어떤 영향을 미치는가?
특히, 어떤 reward 함수 하에서 RL이 ATR 규칙 기반을 초과하며 (혹은 초과하지 못하며), 그 메커니즘은 무엇인가?

**자산 범위**: BTC/USDT 1시간봉 **단일 자산**. 자산 확장은 본 논문 범위 외 (사용자 개인 운용으로 분리).

상세 RQ, 가설, 평가 방법은 [docs/PROJECT_GOAL.md](docs/PROJECT_GOAL.md) 참조.

---

## 현재 위치

```
Phase 1  ██████████ 완료  (BTC 시스템 설계 + RL 학습 + 핵심 발견)
Phase 2  ██████████ 완료  (Bayesian 계수 + ATR vs RL 비교 + 체결가 버그 수정)
Phase 3  ██░░░░░░░░ 진행  (Reward design 본격 탐구 — 본 논문 메인)
```

---

## Phase 1 — BTC 시스템 설계 + 핵심 발견 ✅ 완료 (2026-04)

### 산출물
- BTCGridTradingEnv (Gymnasium 환경)
- 베이스라인 7종 (Buy-Hold, Fixed Grid 1/2/5%, ATR Grid k=0.5/1.0/2.0)
- ATR 비례 그리드 공식 + Bayesian 최적화 (Optuna)
- PPO 학습 파이프라인 (SB3)
- 행동 분석 (regime adaptation), metric 모듈

### 핵심 발견 (Negative finding)
**Symmetric reward + ATR 비례 공식의 조합에서 RL이 ATR과 동등.**

| | Val Sharpe | Test Sharpe |
|---|---|---|
| RL (exp020) | 45.390 | 42.090 |
| **Fixed [1.0, 0.0]** (RL 수렴 값 고정) | **45.390** | **41.769** |

→ RL이 학습한 것이 사실상 상수. ATR/price 항이 변동성 차원의 알파를 이미 흡수.

### 의의
- **RQ-1의 사전 증거 확보**: Symmetric reward 하에서 RL의 추가 알파 없음.
- 본 논문 5장 (Positive finding)의 출발점.

---

## Phase 2 — 시뮬레이션 정합성 + ATR 재확립 ✅ 완료 (2026-04)

### 산출물
- 체결가 버그 수정 (`next_low/next_high` → 지정가)
- ATR 계수 재최적화 (exp026, Val Sharpe 1.978)
- RL 재학습 (exp026 RL, Val Sharpe 0.896)
- **Asymmetric reward 시도 (exp027_rl)**: Test Sharpe 1.955 (ATR 0.935 대비 2배)
- exp028~029: 학습 안정화 시도 (진행 중 보류)

### 핵심 발견 (Positive finding 단초)
**Asymmetric reward (beta=2.0)로 RL이 ATR을 명확히 초과.**

| 시스템 | Test Sharpe | Test MDD | Test Trades |
|---|---|---|---|
| ATR (exp026) | 0.935 | 2.43% | 1,591 |
| RL exp027_rl best | **1.955** | **0.39%** | 214 |

→ beta=2.0이 Kahneman-Tversky λ≈2.25와 일치 → 학술적 출처 명확.
→ **RQ-2의 사전 증거 확보**. 본 논문의 메인 contribution.

### 의의
- 단순 hyperparameter가 아닌 학술적으로 정당화 가능한 reward 선택이 결정적임을 시사.
- Phase 3에서 정식화 (DSR, Prospect-theoretic) + 통계 검증 필요.

---

## Phase 3 — Reward Design 본격 탐구 (본 논문 메인) 🟡 진행

### 목표
4가지 reward 변형을 동일 환경에서 비교하고, CPCV + DSR로 통계적으로 검증.

### 실험 시리즈 (exp030 ~ exp035)

| Exp | 목적 | 논문 챕터 | 예상 기간 |
|---|---|---|---|
| **exp030** | PPO 학습 안정화 패키지 | Method 3.3 | 1주 |
| **exp031** | BC warm-start (학습 초반 낭비 회피) | Method 3.4 | 1주 |
| **exp031b** | (조건부) CQL + mixed dataset | Method 3.4 | +2~3주 |
| **exp032a** | **Variant별 reward hyperparameter 튜닝 (공정 비교 보장)** | **Method §3.5** | 1~2주 |
| **exp032b** | **4 variant full 비교 + effect size 분석 (메인)** | **§5 Positive finding** | 1~2주 |
| **exp032c** | **메커니즘 분석 (counterfactual, SHAP, mediation)** | **§6 Mechanism** | 1주 |
| **exp033** | Slippage + Domain Randomization | §7.1 Robustness | 1주 |
| **exp034** | CPCV 6-fold + DSR 계산 | §5, §7.2 | 1~2주 |
| **exp035** | Test set 봉인 해제 (1회) | §7.3 최종 검증 | 당일 |

상세 실험 설계는 [docs/study/rl_finance/project_continuation_plan.md](docs/study/rl_finance/project_continuation_plan.md) 참조.

### 완료 기준
- exp032에서 reward 변형 간 명확한 성능 차이 (CPCV 분포 비교)
- DSR p-value < 0.05 (다중검정 보정 후 진짜 알파)
- Test set Sharpe가 CPCV 분포 평균 ± 2σ 안에 (일반화 검증)
- 논문 7장 분량 초안

### 예상 종료 시점
8~11주 + (조건부 exp031b) 2~3주 = 약 2.5개월

---

## Phase 4 — 논문 작성 + 디펜스 준비 (Phase 3 후반과 병행)

### 산출물
- 졸업 논문 본문 (한글/영문)
- 발표 자료 (Phase 3 결과 + 행동 분석 시각화)
- 코드 저장소 정리 (재현 가능성)

### 핵심 인용 (이미 학습 노트로 정리됨)
- Avellaneda & Stoikov (2008) — 그리드 봇의 학술적 조상
- Zhang, Zohren, Roberts (2020) — DRL 트레이딩 baseline
- Gort et al. (2022) — PBO/CPCV (이미 인용 중)
- Moody & Saffell (2001) — DSR
- Kahneman & Tversky (1979) — asymmetric reward
- Ng, Harada, Russell (1999) — reward shaping safety
- Schulman et al. (2017) — PPO 알고리즘
- López de Prado (2018) — CPCV, DSR

---

## Phase 5 (선택, 본 논문 범위 외) — Live Trading

본 논문에는 결과로 포함하지 않으나, 학기 일정상 여유 있으면 진행 가능.
- Binance Testnet 1~2주 운영 (sim2real gap 측정)
- 소액 실거래 ($100) 1~2주
- 논문에는 "Future Work" 또는 "Discussion"에서만 언급

---

## 자산 확장 (의식적으로 제외) ⛔

**본 졸업 논문에는 포함하지 않음**.

이유:
- BTC 하나만으로도 reward design 가설을 견고히 검증 가능.
- 자산 확장 시 데이터 파이프라인/시간 처리/슬리피지 모델 재설계 필요 — 6주+ 추가 작업.
- 다른 자산군(주식, FX, 원자재)은 사용자 개인 운용으로 분리 — 단순 시장수익률 추종으로도 충분.

논문 8장 (Discussion) 에서 "자산 확장 시사점"으로만 짧게 언급.

---

## 의사결정 기록

| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-04 | ATR 비례 공식 채택 | 절대 간격 대비 변동성 적응 우수 |
| 2026-04 | Bayesian 계수 최적화 | A_b=0.285, C_b=5.223 등 |
| 2026-04 | 데이터 2017~현재로 확장 | Val 레짐 균형 확보 |
| 2026-04 | **Pivot 1**: RL 단독 → ATR vs RL 비교 | RL이 ATR 흡수 발견 |
| 2026-04 | 다자산 확장 방향 (당시) | 자산별 RL 가치 조건 탐구 가설 |
| 2026-04 | 체결가 지정가로 수정 | 시뮬레이션 정합성 |
| 2026-04 | exp027_rl asymmetric reward (beta=2.0) | Test Sharpe 1.955로 ATR 초과 |
| **2026-05-14 (1차)** | **Pivot 2**: 자산 확장 제외, RQ를 reward design 중심으로 재정의 (단정문) | 단일 자산 깊이 우선, exp027_rl 발견을 메인 contribution으로 |
| **2026-05-14 (2차)** | **RQ를 단정문 → 열린 질문 형태로 수정**. 사전 증거는 가설(H1~H4)의 정당성으로 흡수, RQ는 검증 결과에 따라 어느 쪽이든 살아남는 형태로. | 학술 컨벤션 준수 + 확증 편향 회피. 사용자 지적. |

---

## 관련 문서

| 문서 | 위치 | 내용 |
|---|---|---|
| **본 논문 목표** | [`docs/PROJECT_GOAL.md`](docs/PROJECT_GOAL.md) | **단일 기준점.** RQ, 가설, 평가 방법 |
| 실험 시리즈 상세 | [`docs/study/rl_finance/project_continuation_plan.md`](docs/study/rl_finance/project_continuation_plan.md) | exp030~035 상세 |
| 학습 자료 허브 | [`docs/study/rl_finance/00_overview.md`](docs/study/rl_finance/00_overview.md) | 23개 이론 노트 인덱스 |
| 공식 정의 | [`docs/FORMULAS.md`](docs/FORMULAS.md) | ATR 고정 / RL 버전 공식 |
| MDP 설계 | [`docs/MDP.md`](docs/MDP.md) | State / Action / Reward 설계 근거 |
| 선행 연구 | [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | 참고 논문 7편 역할 및 차별점 |
| 실험 기록 | [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | 날짜별 의사결정 및 실험 결과 |
| 에이전트 지침 | [`CLAUDE.md`](CLAUDE.md) | Claude Code 작업 규칙 |
