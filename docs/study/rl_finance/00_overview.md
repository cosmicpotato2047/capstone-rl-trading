# RL × Finance — 학습 자료 허브

> 졸업 논문의 이론적 토대 보강 자료. 24개 노트, 4개 묶음.
> 작성: 2026-05-13. 개정: 2026-05-14 (RQ 확정, 자산 확장 제외 반영).
> **단일 기준점**: [[PROJECT_GOAL]] — 모든 노트는 그 RQ에 정렬되어야 함.

## 본 논문의 RQ (PROJECT_GOAL 발췌)

> **BTC 그리드 트레이딩에서 reward 설계가 RL 정책의 알파에 어떤 영향을 미치는가?
> 특히, 어떤 reward 함수 하에서 RL이 ATR 규칙 기반을 초과하며 (혹은 초과하지 못하며), 그 메커니즘은 무엇인가?**

자산 범위: BTC/USDT 1시간봉 **단일 자산**. 자산 확장은 본 논문 범위 외.

⚠️ 사전 증거(exp027_rl 등)는 가설 H1~H4의 정당성으로 흡수. RQ는 검증 결과에 따라 어느 쪽이든 살아남는 열린 질문 형태.

---

## 빠른 참조 — 어떤 노트를 언제 볼 것인가

### "exp032 (메인 챕터) — 4가지 reward 비교 설계"
→ **이론적 출처**: [[differential_sharpe_moody2001]] → [[prospect_theory]] → [[reward_shaping_ng1999]] → [[reward_hacking]]
→ **공정 비교**: [[hyperparameter_parity]] (variant별 hyperparameter 튜닝)
→ **통계 검증**: [[effect_size_rliable]] (Cohen's d, IQM, BEST, rliable)
→ **메커니즘 분석**: [[causal_counterfactual_rl]] (counterfactual, SHAP, mediation)

### "exp030 — PPO 학습 안정화"
→ [[policy_gradient_stabilization]] → [[ppo_schulman_2017]] → [[reward_shaping_ng1999]]

### "exp031 — BC warm-start"
→ [[offline_rl_warm_start]] → (조건부) [[cql_kumar_2020]]

### "exp033 — Slippage + Domain Randomization"
→ [[realistic_execution_simulation]] → [[sim2real_finance]] → [[curriculum_learning]]

### "exp034 — CPCV + DSR 통계 검증"
→ [[walk_forward_cv]] → [[bayesian_optimization_tpe]] → [[gort_2022_crypto_overfitting]]

### "exp020~022 negative finding 재정리 (RQ-1)"
→ [[avellaneda_stoikov_2008]] → [[optimal_grid_spacing]] → [[zhang_zohren_roberts_2020]]

### "RL 알고리즘 기초 (논문 §3 Method)"
→ [[mdp_bellman_pomdp]] → [[ppo_schulman_2017]] → [[ddpg_continuous_control]]

### "Phase 5 (선택) — 라이브 트레이딩"
→ [[realistic_execution_simulation]] → [[sim2real_finance]] → [[reward_hacking]]

---

## 묶음별 목차

### Bundle A — RL × 금융 (직결도 최고)

| # | 노트 | 핵심 | 본 논문 활용 |
|---|---|---|---|
| A1 | [[differential_sharpe_moody2001]] | Sharpe를 매 스텝 reward로 직접 사용 | exp032 DSR variant |
| A2 | [[zhang_zohren_roberts_2020]] | Volatility scaling reward, DRL trading 표준 | §2 Background |
| A3 | [[gort_2022_crypto_overfitting]] | PBO/CPCV 방법론 | exp034, §7.2 |
| A4 | [[finrl_framework]] | DRL trading 표준 라이브러리 | §2 Background (대비) |
| A5 | [[reward_hacking]] | RL trading의 함정. exp026 체결가 버그 사례 | §3.1, exp032 |
| A6 | [[sim2real_finance]] | 시뮬레이터-실거래 갭 | §7.1, Phase 5 |
| A7 | [[distributional_rl]] | CVaR-aware policy, fat tail 대응 | Discussion (시사) |
| A8 | [[hierarchical_rl_trading]] | 전략/전술 분리. exp029 사이클 단위 결정 | Discussion (시사) |

### Bundle D — 약점 보강 (현재 실험과 1:1 매칭)

| # | 노트 | 해결 문제 | 본 논문 활용 |
|---|---|---|---|
| D1 | [[bayesian_optimization_tpe]] | Optuna 정당화 + DSR 보정 | exp034 |
| D2 | [[walk_forward_cv]] | exp027 Val 과적합 → CPCV | exp034, §7.2 |
| D3 | [[policy_gradient_stabilization]] | exp028/029 학습 oscillation | exp030 |
| D4 | [[realistic_execution_simulation]] | 슬리피지, 부분체결 | exp033, §7.1 |
| D5 | [[volatility_modeling]] | ATR 너머 GARCH/RV/Jump | Discussion (시사) |
| D6 | [[offline_rl_warm_start]] | 학습 초반 낭비 회피 | exp031 |
| D6+ | [[cql_kumar_2020]] | Mixed-policy offline RL | exp031b (조건부) |
| D7 | [[curriculum_learning]] | 일반화 + DR | exp033 |

### Bundle E — exp032 정합성 보강 (2026-05-14 추가)

RQ 정합성 점검에서 발견된 세 가지 약점 (공정 비교 + 메커니즘 분석 + 통계 정직성) 해결.

| # | 노트 | 해결 문제 | 본 논문 활용 |
|---|---|---|---|
| E1 | [[effect_size_rliable]] | Variant 비교의 통계적 정직성 (Cohen's d, IQM, BEST, rliable) | exp032b 평가, §5, §7.2 |
| E2 | [[causal_counterfactual_rl]] | RQ-3 메커니즘 답변 강화 (counterfactual, SHAP, mediation) | exp032c, §6 Mechanism |
| E3 | [[hyperparameter_parity]] | Variant별 reward hyperparameter 공정 튜닝 | exp032a, §3.5 Method |

→ exp032가 (a) hyperparameter 튜닝 → (b) full 비교 + effect size → (c) 메커니즘 분석 3단계로 확장.

### Bundle B — RL 이론 기초

| # | 노트 | 핵심 | 본 논문 활용 |
|---|---|---|---|
| B1 | [[mdp_bellman_pomdp]] | RL 수학적 framework | §2 Background |
| B2 | [[ppo_schulman_2017]] | 채택 알고리즘 | §3.2 Method |
| B3 | [[ddpg_continuous_control]] | 연속 action 대안 비교 | §3.2 (선택 근거) |
| B4 | [[reward_shaping_ng1999]] | exp029 r_idle 안전성 | §3.3 + exp030, exp032 |
| B5 | [[prospect_theory]] | asymmetric reward 행동경제학 출처 | §3.4 + exp032 |

### Bundle C — 그리드의 학술적 뿌리

| # | 노트 | 핵심 | 본 논문 활용 |
|---|---|---|---|
| C1 | [[avellaneda_stoikov_2008]] | 마켓 메이킹 정석 — 학술적 조상 | §2 Background, §1 Intro |
| C2 | [[inventory_risk_adverse_selection]] | Glosten-Milgrom, Kyle, Ho-Stoll | §2 Background |
| C3 | [[optimal_grid_spacing]] | 그리드 학술적 정당화 + Short-vol 위험 | §2, §8 Discussion |

---

## 외부 참조 (기존 노트)

이미 사용자가 정리한 노트들 — 본 묶음과 cross-link.

- [[learning list]] — 학습 로드맵 원본
- [[퀀트투자]] — 큰 그림
- [[주요 방법론 상세]] — 방법론 7종
- [[factor]], [[Fama-French 3-5 factor model]], [[두 학파 상세]], [[멀티 팩터 부연 설명]] (Reward design과 직접 관련 없지만 배경 지식)
- [[risk-adjusted return]] — 평가지표 (논문 §5 평가)
- [[Backtesting]], [[Pitfalls]], [[Money and Risk Management]], [[Special Topics in Quantitative Trading]] (Ernie Chan)
- [[Backtest Statistics]], [[Financial Data Structures]], [[Fractionally Differentiated Features]], [[The Dangers of Backtesting]], [[Understanding Strategy Risk]] (López de Prado) — 특히 Backtest Statistics, Dangers of Backtesting이 §7.2 검증과 직결

---

## 프로젝트 진행 계획 (요약)

상세는 [[project_continuation_plan]] 참조.

### Phase 3 (현 진행) — Reward Design 본격 탐구

| Exp | 목적 | 주 참조 노트 |
|---|---|---|
| exp030 | PPO 학습 안정화 | [[policy_gradient_stabilization]], [[reward_shaping_ng1999]] |
| exp031 | BC warm-start | [[offline_rl_warm_start]] |
| exp031b | (조건부) CQL + mixed | [[cql_kumar_2020]] |
| **exp032 (메인)** | **4가지 reward 비교** | [[differential_sharpe_moody2001]], [[prospect_theory]] |
| exp033 | Slippage + DR | [[realistic_execution_simulation]], [[curriculum_learning]] |
| exp034 | CPCV + DSR | [[walk_forward_cv]], [[bayesian_optimization_tpe]] |
| exp035 | Test 봉인 해제 | [[gort_2022_crypto_overfitting]] |

### Phase 4 — 논문 작성 (병행)

논문 챕터별 핵심 인용은 위 "본 논문 활용" 열 참조.

### Phase 5 (선택) — Live Trading

본 논문 범위 외. Discussion에서만 짧게 언급.

---

## 본 자료의 사용 방식

1. **사용자**:
   - learning list의 1순위 (Sutton & Barto, PPO 논문, López de Prado) 정독
   - 본 노트는 빠른 참조용 — 깊이 파고 싶은 주제만 원본 논문으로
   - 새 실험 직전에 해당 Exp의 "주 참조 노트" 다시 확인

2. **클로드(나)**:
   - 새 실험 설계/코드 작성 시 본 노트를 cross-reference
   - 특히 reward 관련 변경 시 [[reward_shaping_ng1999]] + [[reward_hacking]] 점검
   - 새 작업이 RQ 안인지 [[PROJECT_GOAL]]에서 먼저 확인

3. **공동**:
   - 디펜스/논문 작성 시 인용 문헌
   - 위 "본 논문 활용" 열이 인용 위치 매핑
