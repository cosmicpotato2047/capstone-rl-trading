# 알아야 할 것 — 항목화 (자가 점검용)

> 발표자가 *질문 받으면 막힘없이 답할 수 있어야 하는* 항목 리스트.
> 각 항목 옆 박스 `[ ]` 를 `[x]` 로 직접 채워가며 진행도 확인.

---

## A. 큰 그림 (Big Picture)
- [ ] 이 연구를 **30초 / 5분 / 30분 길이**로 각각 말할 수 있다
- [ ] **RQ 한 줄** 암기:
  > "BTC 그리드 트레이딩에서 reward 함수의 설계가 RL 정책의 행동 패턴과 일반화
  > 성능에 어떤 영향을 미치는가? 특히, 어떤 reward 함수 하에서 RL이 ATR 규칙
  > 기반을 초과하며 (혹은 초과하지 못하며), 그 메커니즘은 무엇인가?"
- [ ] **왜 reward design을 묻는가** — 기존 연구가 알고리즘·state 비교에 집중,
  reward 통제 비교는 드묾
- [ ] 핵심 발견 4개를 한 줄씩:
  - (1) Pareto Frontier / 시나리오 D — 단일 winner 없음
  - (2) Winner Reversal — 평가환경마다 우승자 다름 (sym→dsr→pt)
  - (3) PT OOS 강건성 (H5) — 본 논문 main contribution
  - (4) DSR 양면성 — 같은 메커니즘 정반대 결과
- [ ] **학술적 기여 4가지**:
  - Reward 변형의 통계적 비교 프레임워크
  - 시나리오 D 발견 (Pareto frontier)
  - Reward → 행동 → 결과 인과 사슬 정량
  - PT가 RL trading 정책에 부여하는 OOS 안전성 첫 정량 확인 (본 논문 알기로)

## B. 사전 개념 (Foundational Concepts)
- [ ] **강화학습 기본 loop**: env-agent-state-action-reward
- [ ] **PPO**가 뭔지 + 왜 PPO를 골랐는가 (안정성, 표준성)
- [ ] **그리드 트레이딩**: 방향성과 본질적 차이 (변동성 자체에서 수익)
- [ ] **ATR** 정의 (Wilder 1978, 168봉 윈도우 평균 진실 범위)
- [ ] **Sharpe / MDD / Calmar / IQM / 5% CVaR**의 의미
- [ ] **CPCV** vs **Deflated Sharpe** — *둘 다 López de Prado, 다른 개념!*
  - CPCV (2018, ch.8): 시계열 다중 분할 방법론
  - Deflated SR (2014): 백테스트 시도 횟수 보정한 Sharpe 유의성 검정
- [ ] **Cohen's d** 해석 (|d|<0.3 작음, 0.3-0.8 중간, |d|>0.8 큼)
- [ ] **OOS / distribution shift / domain randomization**의 의미

## C. 환경 설계 (MDP)
- [ ] **State 7차원** — 각 변수가 뭘 의미 + 왜 그걸 골랐는가
  - 0: log_price (vs 1주 평균)
  - 1: divergence (평단가 괴리율)
  - 2: holdings_value_ratio
  - 3: cash_ratio
  - 4: volatility (ATR/price)
  - 5: trend_short (3일)
  - 6: trend_long (30일)
- [ ] **Action 2차원** (aggressiveness, profit_target) — 왜 연속? 왜 이산 안 쓰는가
  - 이산 Discrete(3) Buy/Sell/Hold는 표현력 인위적 제한
- [ ] **ATR 비례 격자 공식** — 4개 호가 (buy_hi/lo, sell_mkt/cost) 결정
- [ ] **사이클 정의** — 보유 0 → 첫 매수 → 다시 0
- [ ] **지정가 체결 방식** — 다음 봉 high/low로 fill 판정. *Sell 우선 원칙*
- [ ] **fee 0.05%** (Binance maker fee 기준), `_execute_buy/sell`에서 cash 반영
- [ ] **데이터 분할** (train 2017-08 ~ 2020-12 / val 2021-2023 / test 2024-) + **Test 봉인 원칙**

## D. 4가지 Reward 함수
- [ ] 각각의 **수식 + 직관 + 함수 모양**:
  - `sym`: r = x. 베이스라인.
  - `asym`: x≥0 → x, x<0 → β·x. 위험회피 1차 근사.
  - `dsr`: 샤프비율의 미분 (Moody 2001). EWMA 1차/2차 모멘트 사용.
  - `pt`: sign(x)·|x|^α (이익), -λ·|x|^α (손실). Kahneman 1979.
- [ ] 왜 이 4개인가 (4가지 이론적 계보 대표):
  - sym: 표준 PnL
  - asym: 위험회피 효용 1차 근사
  - dsr: 위험 조정 수익 직접 최적화
  - pt: 행동경제학
- [ ] **하이퍼파라미터 best (Optuna 30 trials × 200k)**:
  - asym β = 3.420
  - dsr η = 0.0352 ≈ 1/28h (EWMA ~1.2일)
  - pt α = 0.683, λ = 3.303
- [ ] **인간 표준 vs RL best**:
  - pt 인간 표준 = α 0.88, λ 2.25 → **RL best는 더 극단적** ← 부수 발견!
- [ ] **dsr이 본질적으로 다른 이유** — *sliding-window memory가 state-extended reward*
  - 다른 3개: r(x) (1-step PnL만 입력)
  - dsr: r(x | A_prev, B_prev) (과거 평균/분산 의존)
- [ ] **Ng et al. 1999 reward shaping과 충돌 안 하는 이유**:
  - sym → asym/pt 는 *비-potential-based* 변환 (Ng 정리 적용 X)
  - dsr 은 *state-extended* (sliding window가 사실상 state 일부)

## E. Phase 1 Negative Finding + Pivot
- [ ] exp020/021/022 각각 무엇을 했고 무엇이 나왔는가
  - exp020: aggressiveness를 budget_fraction으로 재정의 → a^(0) ≈ 1.0 수렴
  - exp021: aggressiveness를 entry_gate로 재정의 → bear regime에서도 99.7% 진입
  - exp022: 원래 aggressiveness 복원 → a^(0)≈0, a^(1)≈0 수렴, raw output [-9.19, -4.30] saturated
- [ ] **"RL = Fixed [1.0, 0.0]"** 의 의미 — Val Sharpe 소수점 셋째 자리까지 일치
- [ ] **왜 그렇게 됐는가** (구조적 원인):
  - ATR 공식이 변동성을 *이미* 흡수 (g = ATR/price × (A + B·a))
  - sym reward는 사이클 수 극대화가 유일 최적해
  - → 매수 격자 가장 좁게 (a^(0)→1), 매도 격자 가장 좁게 (a^(1)→0)
- [ ] **왜 pivot 했고 무엇으로**:
  - 가정: sym reward + ATR 공식 조합이 RL 자유도 흡수
  - → reward를 비대칭(asym/pt) or 경로의존(dsr)으로 정식화하면 추가 가치?
  - 이게 본 논문 RQ로 정식화됨
- [ ] **왜 본 논문에서 Phase 1을 정직 인용**했는가:
  - 학술 윤리: pivot 정당화
  - 환경이 Env-v2 (favorable bias artifact) 였음을 caveat
  - 정성 결론 ("RL = Fixed") 만 인용, 절대 수치 (Sharpe 45.39) 는 비교에 안 씀

## F. Phase 2 환경 정상화
- [ ] **favorable bias가 뭐고 왜 제거**:
  - 이전: 다음 봉의 high/low로 *체결가 결정* → 가짜 수익 (수조%)
  - 본 논문: 다음 봉 high/low로 *체결 여부만 판정*, 가격은 호가 그대로
- [ ] **학습 안정화 패키지 4가지** (exp030):
  - LR linear decay (3e-4 → 1e-5)
  - Entropy coef annealing (0.01 → 0.001)
  - Target KL early stop (target_kl = 0.02)
  - Best checkpoint (50k step마다 Val Sharpe로 저장)
- [ ] **왜 best checkpoint를 쓰는가**:
  - 학습 후반 정책 붕괴 패턴: exp030 best step 550k (1.974) → final 1M (1.209)
  - final 사용 시 underestimate
- [ ] **Optuna TPE + MedianPruner** 작동 방식
- [ ] **Env 버전 차이**:
  - v2 (Phase 1): 다음 봉 극값 체결, favorable bias
  - v3 (Phase 2 중반): 4D 절대 gap action, 지정가 체결, asym Test 1.955 ← 사전 증거
  - v4 (본 논문 canonical): 2D ATR 비례 action, 지정가 체결

## G. exp032b 시나리오 D
- [ ] **실험 설정**: 4 var × 10 seed × 1M step = 40M (~3h 44m)
- [ ] **사전 등록 시나리오 A/B/C**:
  - A 낙관: asym/dsr/pt > sym과 ATR 모두 압도
  - B 중립: variant 차이 있으나 ATR에 못 미침
  - C 비관: variant 차이 미미
- [ ] **시나리오 D 정의** + 왜 사후로:
  - 결과가 A/B/C 어디에도 안 맞음
  - 4 var 모두 ATR 초과 but 단일 winner 없음
  - Sharpe-MDD 평면에서 Pareto-유사 frontier
- [ ] **두 클러스터**:
  - {sym, dsr} = aggressive (높은 Sharpe + 높은 MDD)
  - {asym, pt} = conservative (중간 Sharpe + 낮은 MDD)
- [ ] **통계적 분리 정량**:
  - within Cohen's d: sym-dsr 0.29, asym-pt 0.15 (작음)
  - across Cohen's d: sym-asym 1.10, sym-pt 1.19, dsr-asym 0.79, dsr-pt 0.89 (큼)
  - policy distance L2 비율 2.22× (across/within)
- [ ] **각 metric 1위가 다 다른 점**:
  - best Sharpe: sym (1.871)
  - final Sharpe: dsr (1.204)
  - MDD/Calmar: asym (2.28%, 0.755)
  - 누적 수익률: dsr (7.40%)
  - 거래수: sym (120)
- [ ] **가설 verdict**:
  - H1 (sym=ATR): 부정. sym 1.87 > ATR 1.38, t-test p<10⁻³
  - H2 weak (모두 > ATR): 지지
  - H2 strong (asym/dsr/pt > sym): 부분 부정
  - H3 (selective entry): 지지 (conservative 거래수 sym의 75%)

## H. 메커니즘 (exp032c)
- [ ] **인과 사슬**:
  ```
  Reward 형식 → 거래빈도 + Hold시간 → Risk profile cluster → (Sharpe, MDD) trade-off
  ```
- [ ] **거래 빈도 차이** (40 model trajectory 1.04M step):
  - sym/dsr ≈ 0.073, asym 0.058, pt 0.049 (low vol 기준)
  - 손실 비대칭(β=3.42, λ=3.30) → 매수 호가 멀리 → 거래 25-35% ↓
- [ ] **Hold rate 차이** ★ 가장 중요한 단서:
  - sym 0.074, asym 0.036, pt 0.031, **dsr 0.142** (low vol)
  - high vol에서는 dsr 0.120 vs pt 0.020 → **6배 차이**
- [ ] **DSR sliding-window memory가 긴 hold 학습시키는 이유**:
  - DSR step reward = (B·ΔA − ½A·ΔB) / (B−A²)^1.5
  - 짧은 holding → ΔA, ΔB에 큰 noise → 학습 신호 약화
  - → 긴 holding 선호 정책 학습
- [ ] **Action 평균** (메커니즘 확인):
  - sym/dsr: a^(0)≈0.12 (좁은 매수)
  - asym: a^(0)≈0.30, pt: a^(0)≈0.45 (멀리 매수)
  - dsr만: a^(1)≈0.15 (넓은 매도 = 긴 hold)
- [ ] **Counterfactual state-grid**:
  - 같은 (atr_ratio, divergence) 셀에서도 4 var의 평균 action 다름
  - → 분포 차이가 아닌 policy 차이가 cluster 원인

## I. 강건성 (exp033 slippage + exp034 CPCV)

### I.1 exp033 (Slippage 0.02%)
- [ ] σ = 0.02% — Binance taker 95% 분위
- [ ] **4 variant 모두 일률 ~12% Sharpe 감쇠**:
  - sym 1.871 → 1.658 (-0.21)
  - dsr 1.809 → 1.551 (-0.26)
  - asym 1.681 → 1.478 (-0.20)
  - pt 1.667 → 1.459 (-0.21)
- [ ] **MDD는 conservative cluster에서 거의 변화 없음** (거래 횟수 적음)
- [ ] **ATR baseline 공정 비교 (Phase 16a)**:
  - ATR with slippage = 0.835 (1.378에서 -39% 감쇠)
  - 4 RL variant 모두 ATR-slip 대비 **+75~99% 초과**
- [ ] **Cluster preservation ratio**: 2.19× (exp032b 2.22×와 거의 동일)

### I.2 exp034 (CPCV 6-fold 15 paths)
- [ ] **설정**: train 4 groups + test 2 groups, purge ±168h
- [ ] **결과**:
  - dsr 1위 (1.413 ± 0.378, IQM 1.433, 5% CVaR 0.890)
  - sym 1.302, asym 1.043, pt 1.093
- [ ] **모든 variant p < 4×10⁻⁷ 단측 t-test로 ATR 초과**
  - Bonferroni 4-way 보정 후에도 p<0.004
  - Deflated Sharpe z > 14 → numeric stability 유지
- [ ] **1차 winner reversal**: Val sym(1.87) → CPCV dsr(1.41)
- [ ] **Cluster preservation ratio 3.55×** (multi-split에서 *더 또렷*)

## J. OOS와 H5 (exp035 + phase16d) ★★ 본 논문 핵심
- [ ] **Test 봉인 이유** (Gort 2022):
  - 한 번 보면 더 이상 미공개 아님
  - exp035 (2026-05-16) 까지 어떤 분석에도 미사용
  - 본 논문이 사상 첫 1회 봉인 해제
- [ ] **평가 대상 100개 RL 모델**:
  - exp032b 40개 (4 var × 10 seed)
  - exp034 60개 (4 var × 15 path)
- [ ] **Test 결과 per source per variant**:
  - exp032b: sym +0.090, asym +0.173, **dsr -0.122**, **pt +0.367**
  - exp034: sym +0.001, asym +0.175, dsr +0.070, **pt +0.339**
  - ATR baseline = -0.055
- [ ] **2차 winner reversal**: CPCV dsr → Test **pt**
- [ ] **pt 양 source 일관 1위**: p<0.0015 / p<0.0004 (Bonferroni 후에도 유의)
- [ ] **dsr OOS 실패**: exp032b source -0.122 → ATR baseline (-0.055) 보다도 나쁨
- [ ] **세 환경 세 winner** (본 논문 frame):
  - Val: sym (1.871)
  - CPCV: dsr (1.413)
  - **Test: pt (0.367/0.339)** ★
- [ ] **PT OOS 강건성 메커니즘** (Phase 16d hold duration):
  - mean 1.39h, median 1h, max 6h (short-bounded)
  - → BTC bull market 매수 후 빠르게 청산 → sell-side timing risk 회피
- [ ] **DSR OOS 실패 메커니즘**:
  - mean 4.58h, max **169h (7일!)**
  - → bull market 강세장에서 7일간 가격 멀어짐 → sell 호가 fill 지연
- [ ] **정책 안정성 vs 시장 shift**:
  - 같은 model의 Val/Test 행동: Δtrade_rate ≤ 0.006, Δa^(0) ≤ 0.051
  - 시장 shift: KS p<10⁻¹⁰, 변동성 0.96% → 0.70% (-27%)
  - → 정책 변화 아님, 시장 분포 이동이 직접 원인
- [ ] **B&H 정직 인정 (Phase 16b)**:
  - B&H Sharpe 0.757 > pt 0.367 (단순 Sharpe)
  - B&H MDD 50% vs pt 2.3% → **Calmar는 pt 10배 우위**
  - 본 논문 "pt OOS robust" 주장은 *risk-adjusted* 의미에서 성립
- [ ] **H5 학술 의의**:
  - "Prospect theory가 RL trading 정책에 OOS 안전성 부여"
  - 본 논문 알기로 첫 정량 확인
  - 인간 표준값(α=0.88, λ=2.25)보다 더 극단적인 (α=0.683, λ=3.303)이 RL에 유리
  - → 응용 권고: "행동경제학 추정값 그대로 쓰지 말고 도메인 데이터로 재튜닝"

## K. 결론 + 한계
- [ ] **메인 frame 4가지** (본 논문 §9):
  - (1) Reward variant 영향 = risk profile trade-off (Pareto frontier)
  - (2) 단일 metric winner = 평가 환경 종속 (세 환경 세 winner)
  - (3) Prospect theory가 OOS robust 정책 산출 (H5)
  - (4) In-sample 다양화 우위 ≠ OOS 일관성 (DSR 양면성)
- [ ] **평가 권고**: 세 환경 동시 사용 (Val + CPCV + sealed Test)
  - 단일 환경 결과로 "X가 best"는 학술적으로 부정확
- [ ] **한계 5가지** 정직 인정:
  - (1) BTC 단일 자산
  - (2) Test 강세장 단일 regime
  - (3) Slippage 단일 수준 0.02%
  - (4) DR 미적용
  - (5) 학습 1M step 단일
- [ ] **exp027_rl 환경 의존성 정직 인정**:
  - Env-v3 asym Test Sharpe 1.955 (강한 사전 증거)
  - Env-v4 (본 논문) asym Val Sharpe 1.681 (< sym 1.871)
  - **환경 효과 > reward variant 효과** 인정
- [ ] **Future work 3개 우선순위**:
  - Multi-asset (주식 ETF, FX, 원자재) — pt 일반화
  - Meta-RL adaptive (α, λ) — 시변 환경
  - DR 통합 — dsr OOS 실패 회복 가능한지

## L. 선행 연구 좌표
- [ ] **각 인용 논문의 역할**:
  - Avellaneda 2008 — market making 원조 (그리드 학술적 조상)
  - Wilder 1978 — ATR 원본 (격자 폭 표준 방법)
  - Moody 2001 — DSR (본 논문 4 reward 중 하나)
  - Kahneman 1979 / Tversky 1992 — Prospect Theory (본 논문 pt reward 정식)
  - López de Prado 2018 ch.8 — CPCV (본 논문 exp034 방법론)
  - López de Prado 2014 — Deflated Sharpe (본 논문 exp034 보정)
  - Gort 2022 — crypto DRL backtest overfitting (본 논문 평가 프로토콜 근거)
  - Henderson 2018 — RL reproducibility (10시드, IQM 권장사항)
  - Ng et al. 1999 — reward shaping invariance (본 논문이 *경계* 밖임을 명시)
- [ ] **직접 선행 4편 + 차별점**:
  - Liu 2021 — LSTM 가격 예측 + PPO. 본 논문은 가격 예측 단계 *완전 제거*
  - Yasin 2024 — DQN/PPO/A2C + 20 기술지표, *이산 action*. 본 논문은 *연속 2D*
  - Pham 2025 — DQN으로 5개 사전 전략 선택. 본 논문은 *전략 자체를 학습*
  - Bandarupalli 2025 — PPO + 변동성 패널티. *단일 reward의 hyperparameter 조정*, 본 논문은 *reward 형식 4가지 비교*
- [ ] **본 논문의 한 줄 좌표**:
  > "기존 DRL 트레이딩 문헌이 단일 reward의 알고리즘/state 설계 비교에 집중한 반면,
  > 본 논문은 같은 알고리즘(PPO)·같은 환경 위에서 reward 함수 자체의 4가지 변형을
  > 통제 비교, 그 효과를 in-sample / multi-split / OOS 세 환경에서 분리 측정"

## M. 발표 흐름·청중 대응
- [ ] **청중 수준 가정**: RL/금융 비전문가 + 교수님 중 일부 전문가
  - 첫 5분은 비전문가용 비유 (낚시터·강아지 훈련 등)
  - 질문 받을 때 청중 수준 추정 후 답
- [ ] **시각자료 발표 시 1순위 그림 3장**:
  - Pareto scatter (exp032c menu1) — 두 클러스터
  - Hold duration (phase16d menu3) — dsr long-tail
  - Three env reversal (phase15 menu_c) — 한 장 frame 요약
- [ ] **예상 적대적 질문 대응**:
  - "B&H가 Sharpe 더 높은데?" → Calmar 10배 우위 + risk-adjusted 의미
  - "왜 PPO만? SAC/TD3는?" → 한계로 인정, future work
  - "10 시드는 부족하지 않나?" → Henderson 2018 권장 5-10, IQM/bootstrap CI 사용
  - "Test가 강세장 단일인데 일반화?" → 한계로 정직 인정, 메커니즘은 유효
- [ ] **시간 배분 (15분 발표 기준)**:
  - 큰 그림 1분 (RQ + 발견 4개)
  - 배경 2분 (그리드 + reward 중요성)
  - 방법론 3분 (MDP + 4 reward)
  - 결과 5분 (Pareto + Reversal + H5)
  - 한계·결론 3분
  - Q&A 1분 buffer

---

**총 11개 영역, 약 100개 자가 점검 항목.** 드릴 진행하면서 `[x]` 채워가기.
