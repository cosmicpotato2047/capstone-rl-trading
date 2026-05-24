# 발표 예상 질문 + 모범 답안 (60문)

> 각 질문에 **10초 답 / 1분 답 / 함정·후속 질문** 3단계로 정리.
> 드릴 시: 본인 답 → 모범 답과 비교 → gap 메모.

---

## A. 큰 그림 (Q1–6)

### Q1. 이 연구를 한 문장으로 설명해주세요.
**10초**: "BTC 그리드 트레이딩에서 reward 함수의 모양이 RL 정책의 행동과 일반화 성능을 결정한다는 것을 4가지 reward를 통제 비교해서 입증한 연구입니다."

**1분**: "비트코인 1시간봉 시장에서, 가격 방향 예측 없이 그리드 폭을 결정하는 PPO 강화학습 정책에 4가지 다른 reward 함수 — 대칭, 비대칭, Differential Sharpe, Prospect theory — 를 적용하고, 단일 분할 / CPCV / 봉인 해제 Test 세 환경에서 비교했습니다. 핵심 발견은 단일 winner가 없고 두 클러스터로 갈라지며, 평가 환경에 따라 우승자가 reversal된다는 것. 특히 Kahneman의 Prospect Theory를 reward로 직접 쓴 정책이 미공개 미래 시장에서 가장 robust 했습니다."

**후속 가능**: "왜 그리드 트레이딩인가?" → 변동성 자체에서 수익 추구, 방향 예측 불필요.

### Q2. RQ를 정확히 말해보세요.
**10초**: "BTC 그리드 트레이딩에서 reward 함수의 설계가 RL 정책의 행동 패턴과 일반화 성능에 어떤 영향을 미치는가, 그리고 그 메커니즘은 무엇인가."

**1분**: 위 한 줄 + "특히 RL이 ATR 규칙 기반 베이스라인을 초과하는 reward 형식이 있는지, 손실 비대칭(asym, pt)이나 위험 조정(dsr) 정식화가 의미 있는 차이를 만드는지를 정량적으로 검증합니다."

**함정**: "RQ는 결론적인 단정 형태입니까?" → No. **열린 질문** 형태. 사전 증거가 강해도 RQ는 thesis statement 아님 (학술 컨벤션).

### Q3. 학술적 기여 4가지는?
**10초**: "(1) Reward 변형 통계 비교 프레임워크, (2) Pareto frontier 발견, (3) 인과 사슬 정량, (4) Prospect theory의 RL OOS 안전성 첫 정량."

**1분**: 위 4개 각각 한 줄로:
- (1) 4 reward × 10 시드 × 1M, Cohen's d / bootstrap / IQM / CVaR
- (2) 사전 시나리오 A/B/C 어디에도 안 맞는 사후 시나리오 D
- (3) Reward → 행동 → risk profile → 환경별 winner의 1.04M step trajectory 분석
- (4) Kahneman 1979 prospect theory가 RL trading 정책에 부여하는 OOS 안전성을 본 논문이 알기로 첫 정량 확인

### Q4. 4가지 핵심 발견을 30초에 말해보세요.
**1분**:
1. **Pareto frontier**: 단일 winner 없음, 두 클러스터 {sym, dsr} vs {asym, pt}로 trade-off
2. **Winner reversal**: Val에서 sym, CPCV에서 dsr, Test에서 pt — 평가 환경마다 다른 1위
3. **PT OOS 강건성 (H5)**: 손실 회피 reward로 학습된 정책이 미공개 2024+ 강세장에서 양 source 일관 1위 (p<0.002)
4. **DSR 양면성**: 같은 sliding-window memory가 in-sample에서는 강점, OOS에서는 치명적 약점

### Q5. 이게 왜 학문적으로 중요한가요?
**1분**: "기존 RL 트레이딩 연구는 알고리즘 비교(PPO vs DQN), state 설계 변경, 가격 예측 통합에 집중했고, **reward 함수 자체를 통제 비교**한 연구가 드뭅니다. 본 논문은 같은 PPO·같은 환경 위에서 reward 4가지를 비교하여, reward 형식이 정책의 risk profile을 직접 결정함을 정량 입증합니다. 특히 Prospect theory가 RL 정책에 OOS robust를 부여한다는 발견은 행동경제학과 RL trading의 교차점에서 본 논문 알기로 첫 사례입니다."

### Q6. 이 시스템으로 돈을 벌 수 있나요?
**1분**: "현재 형태로는 권하지 않습니다. (1) BTC 단일 자산, 1시간봉 단일 시간단위만 검증, (2) Test 시기 2024+가 강세장이라 buy-and-hold가 단순 Sharpe는 더 높음, (3) 실 거래 슬리피지·수수료는 본 논문보다 더 클 수 있음. 본 논문 main contribution은 *실 거래 시스템 제안*이 아니라 *어떤 reward가 OOS robust한가의 메커니즘 정량*입니다. 단, pt의 Calmar(0.159)는 B&H Calmar(0.015)의 약 10배로, risk-adjusted 의미에서는 의미 있는 결과입니다."

---

## B. 사전 개념 (Q7–12)

### Q7. 강화학습(RL)이 뭔가요? 비전문가도 알아듣게.
**10초**: "강아지에게 '앉아'를 가르치는 컴퓨터 버전. 환경에서 행동하고 → 보상을 받고 → 보상이 높았던 행동을 더 자주."

**1분**: "에이전트가 환경에서 행동을 선택 → 환경이 보상과 다음 상태를 돌려줌 → 에이전트는 미래의 누적 보상이 최대가 되도록 행동 함수를 조금씩 수정. 본 논문에서는 BTC 1시간봉 시장이 환경, PPO 알고리즘으로 학습된 신경망이 에이전트, 매 시간 그리드 호가 폭 2개 숫자가 행동, 자본 변화가 보상입니다."

### Q8. PPO는 어떤 알고리즘이고 왜 PPO를 골랐나요?
**1분**: "Proximal Policy Optimization은 정책 업데이트의 크기를 clipping으로 제한하여 학습 안정성을 확보한 알고리즘. 현재 강화학습 표준에서 가장 안정적이고 reproducible한 방법 중 하나여서 reward 변형 비교의 변동성을 algorithm 효과가 아닌 reward 효과로 격리할 수 있습니다. SAC나 TD3도 가능하지만 본 논문은 PPO로 통일 — 다른 알고리즘 비교는 future work."

### Q9. 그리드 트레이딩과 방향성 트레이딩의 본질적 차이는?
**1분**: "방향성 트레이딩은 *가격이 오를지 내릴지를 예측*해서 사고팝니다 — '오를 것 같다 → 산다'. 그리드 트레이딩은 가격 방향과 무관하게 *흔들림 자체에서 수익*을 추구합니다. 현재가 위·아래로 매수·매도 호가를 깔아두고, 가격이 그 호가에 닿으면 자동 체결되어 작은 차익을 누적. 비유하면 낚시터에 여러 깊이로 낚싯대를 던져두는 것."

### Q10. ATR이 뭐고 왜 이걸 쓰나요?
**1분**: "Average True Range — Wilder(1978) 변동성 지표. 168시간(=1주일) 윈도우의 평균 진실 범위(true range)입니다. 그리드 폭을 시장 변동성에 적응시키는 표준 방법으로, 변동성이 클 때는 격자 폭을 자동으로 넓히고 작을 때는 좁힙니다. 본 논문에서는 (1) ATR 베이스라인 정책의 그리드 폭 결정, (2) RL 정책의 격자 공식 입력 — 두 역할."

### Q11. Sharpe, MDD, Calmar 차이는?
**1분**: "Sharpe = 평균 수익률 / 수익률 표준편차. 위험 대비 수익. MDD(Maximum Drawdown) = 자산이 고점에서 최대로 떨어진 비율(%). Calmar = 연환산 수익률 / MDD. Sharpe는 일상 변동성을 보고, Calmar는 *최악의 손실 시나리오*를 봅니다. 본 논문 핵심 비교 — B&H Sharpe 0.757 vs pt 0.367 (B&H 우위)이지만, Calmar는 B&H 0.015 vs pt 0.159 (**pt 10배 우위**) — risk-adjusted 의미에서 의미 있는 차이입니다."

### Q12. CPCV와 Deflated Sharpe Ratio (DSR\*)는 다른 거죠? 둘 다 DSR 약자 같은데.
**1분**: "**완전히 다른 두 개념이고 둘 다 López de Prado 저자라서 헷갈리기 쉽습니다.** (1) **CPCV** = Combinatorial Purged Cross-Validation, 2018 책 ch.8. 시계열 데이터에서 train/test 누설을 차단한 다중 분할 방법. (2) **Deflated Sharpe Ratio** = 2014 논문. 백테스트 시도 횟수가 많을수록 발생하는 우연 Sharpe를 보정하는 통계 검정. 본 논문은 (1)로 평가 분할을 만들고 (2)로 결과 유의성을 보정 — *둘 다 사용*. 그리고 본 논문 reward 변형 중 하나인 **Differential Sharpe Ratio (Moody 2001)**도 약자가 DSR이라 *세 번째* 혼동 — 항상 풀어서 답할 것!"

---

## C. 환경 설계 (MDP) (Q13–17)

### Q13. State 7개를 왜 그렇게 골랐나요?
**1분**: "시장 상태와 포지션 상태를 모두 포착해야 합니다. 시장: log_price (1주 평균 대비), volatility (ATR/price), trend_short (3일), trend_long (30일). 포지션: divergence (평단가 괴리), holdings_value_ratio, cash_ratio. **divergence가 핵심** — 보유 중일 때만 의미를 가지며 미보유 시 0. 모든 변수는 168봉 rolling z-score로 정규화 후 [-5, 5] 클리핑하여 학습 안정성 확보."

### Q14. Action을 왜 연속 2차원으로 했나요? 이산 buy/sell/hold가 단순하지 않나요?
**1분**: "이산 Discrete(3) Buy/Sell/Hold는 *그리드의 미세 폭 조정과 변동성 적응을 인위적으로 제한*합니다. 그리드 폭이 ATR 비례로 연속 결정되어야 시장 변동성에 자연스럽게 적응할 수 있어 연속 action이 필수. 2차원으로 한 이유는 aggressiveness(매수 호가 위치)와 profit_target(매도 호가 위치)이 의미적으로 독립적이기 때문 — 같은 신호에서 두 호가를 따로 조정 가능."

### Q15. ATR 비례 격자 공식을 설명해보세요.
**1분**: "현재가 p_t, ATR ratio r_t = ATR(168)/p_t, 평단가 p_avg에 대해:
- buy_hi 가격 = p_t × (1 − r_t × (A_b + B_b × aggressiveness))
- buy_lo 가격 = p_t × (1 − r_t × (C_b + D_b × aggressiveness))
- sell_mkt 가격 = p_t × (1 + r_t × (A_s + B_s × profit_target))
- sell_cost 가격 = **p_avg** × (1 + r_t × (C_s + D_s × profit_target))

핵심은 (1) r_t 항이 시장 변동성을 *자동 흡수*하므로 RL은 그 위에서 *추가 조정*을 학습, (2) sell_cost만 평단가 기준 — '원가 회복'을 정책에 명시적으로 노출."

### Q16. Sell 우선 원칙이 뭐고 왜 필요한가요?
**1분**: "같은 1시간봉에서 매수 호가와 매도 호가가 동시에 다음 봉의 high/low 범위에 들어와 둘 다 fill 조건을 만족할 수 있습니다. 이 경우 **매도를 먼저 처리** — 보유분 청산이 자본 회수로 이어져 다음 매수 여력을 확보. 만약 매수를 먼저 처리하면 자본이 묶여서 시뮬레이션에 인위적 편향이 생깁니다."

### Q17. 왜 Test partition을 봉인했나요?
**1분**: "백테스트 과적합(backtest overfitting)을 방지하기 위해서입니다. Test를 한 번 보면 결과에 따라 'hyperparameter 살짝 조정해볼까' 하는 유혹이 생기고, 그 순간 Test는 더 이상 미공개 미래가 아니라 *추가 validation*이 됩니다. Gort et al.(2022)의 crypto DRL 권고에 따라 exp035 단계(2026-05-16)까지 어떤 분석에도 사용 안 했고, 본 논문이 사상 첫 1회 봉인 해제입니다."

---

## D. 4가지 Reward 함수 (Q18–23)

### Q18. 4가지 reward 각각의 정의를 수식으로?
**1분**:
- **sym**: r = x, where x = (E_t − E_{t-1})/C_0 (1-step PnL ratio)
- **asym**: r = x if x≥0 else β·x, **β = 3.42**
- **dsr** (Moody 2001): r = (B_{t-1}·ΔA_t − ½·A_{t-1}·ΔB_t) / (B_{t-1} − A_{t-1}²)^1.5, where A, B는 EWMA 1차/2차 모멘트, **η = 0.0352 ≈ 1/28h**
- **pt** (Kahneman 1979): r = sign(x)·|x|^α if x≥0 else −λ·|x|^α, **α = 0.683, λ = 3.303**

### Q19. 왜 이 4개를 골랐나요? 다른 reward도 많지 않나요?
**1분**: "4가지 *이론적 계보*를 대표합니다. sym은 표준 PnL 베이스라인. asym은 위험 회피 효용의 1차 근사 (β로 손실 가중). dsr은 위험 조정 수익을 직접 reward로 사용하는 Moody 2001의 정식. pt는 Kahneman의 행동경제학 효용 함수. 더 많은 변형(CVaR reward, hyperbolic discounting 등)은 졸업 논문 범위 제약 + future work."

### Q20. Optuna best 값이 인간 표준값과 다른 점이 흥미롭다는데 무슨 의미죠?
**1분**: "Prospect theory 표준값 (Tversky 1992 기반)은 α=0.88 (약한 오목성), λ=2.25 (중간 손실 회피). 본 논문 Optuna best는 α=0.683 (더 강한 오목성), λ=3.303 (47% 더 강한 손실 회피). 즉 **RL 정책 입장에서는 인간보다 더 보수적인 효용함수가 OOS robust 수익에 유리**합니다. 이는 인간 트레이더의 '이익을 너무 빨리 실현한다' 패턴이 OOS 안전성 측면에서는 합리적임을 시사하며, 실용 권고: 행동경제학 추정값을 그대로 RL에 쓰지 말고 도메인 데이터로 재튜닝하라."

### Q21. dsr이 다른 3개와 본질적으로 다른 이유는?
**1분**: "다른 3개(sym, asym, pt)는 입력이 1-step PnL `x` 하나뿐인 *memoryless* 함수입니다. **dsr만 입력이 x + 과거 윈도우의 평균 A_prev와 분산 B_prev**입니다. 즉 step reward가 *state-extended* — sliding window가 사실상 state의 일부. 이 메모리 구조가 정책에 시간 의존성을 학습시키고, 결과적으로 다른 변형의 2-6배 긴 holding을 선호하게 만듭니다. 이게 본 논문 §6.2의 메커니즘이며, in-sample 우위(CPCV 1위)와 OOS 실패(Test 꼴찌)의 양면을 동시에 결정합니다."

### Q22. Ng et al. 1999 reward shaping invariance 정리에 위배되지 않나요?
**1분**: "Ng et al.은 *potential-based* reward shaping의 단조 변환은 최적 정책을 변경하지 않는다는 정리입니다. 본 논문 4 reward 비교는 이 정리의 *적용 범위 밖*입니다: (1) sym → asym/pt는 음의 영역에 비대칭 스케일을 부여하는 *non-potential-based* 변환, (2) dsr은 *state-extended* (sliding window가 state의 일부). 따라서 변형 간 정책 행동이 통계적으로 유의하게 차이남이 정리와 모순 아니며, §6.5에서 정량 확인됩니다."

### Q23. 왜 Optuna 30 trials × 200k step으로 골랐나요? 더 많이 안 했나요?
**1분**: "총 학습량 3 variants × 30 trials × 200k = 18M step (~2시간 CPU). 계산 자원과 변형 비교의 정확성 사이 trade-off입니다. TPE sampler + MedianPruner로 효율적 탐색, seed=42 고정으로 재현성 확보. 200k는 1M 학습의 20%지만 best Sharpe 추세를 안정적으로 잡기 충분 (exp030 결과로 검증). hyperparameter 정밀도가 cluster 분리 결과에 큰 영향 없을 것이라는 가설은 cluster 보존성(slippage 2.19×, CPCV 3.55×)으로 사후 확인됨."

---

## E. Phase 1 negative + pivot (Q24–27)

### Q24. Phase 1에서 정확히 무슨 일이 있었나요?
**1분**: "2026년 4월 exp020~022 세 실험에서 PPO + sym reward로 1M step 학습 결과, 학습된 정책이 사실상 상수 행동 [1.0, 0.0]으로 수렴했습니다. exp020에서는 a^(0)≈1.0, exp021에서는 entry gate 99.7% 열림, exp022에서는 a^(0)≈0, a^(1)≈0 + raw policy output [-9.19, -4.30]로 saturation 영역에 포화. 결정적 ablation: RL과 Fixed[1.0,0.0] 정책의 Val Sharpe가 *소수점 셋째 자리까지 일치*."

### Q25. "RL = Fixed [1.0, 0.0]"의 메커니즘을 설명해보세요.
**1분**: "구조적 원인 두 가지. (1) **ATR 격자 공식 자체가 변동성을 이미 흡수**: g = ATR/price × (A + B·a). 하락장(ATR↑)에는 격자 폭이 자동 확대되어 보수적 포지션 자동 달성, 상승장(ATR↓)에는 격자 폭 자동 축소. RL에 남은 자유도가 매우 작음. (2) **sym reward 하에서 최적해가 시점-독립**: 사이클 수 극대화 = 복리 누적 극대화가 유일 최적화 방향. 그 결과 매수 격자 가장 좁게(a^(0)→1), 매도 격자 가장 좁게(a^(1)→0)가 단일 최적해. 상태 의존 행동 학습 인센티브 사라짐."

### Q26. 그래서 왜 본 논문에서 Phase 1을 인용하나요?
**1분**: "두 가지 이유. (1) **연구 narrative**: Phase 1 negative가 본 논문 RQ를 동기화하는 출발점입니다. 'sym reward + ATR 공식 조합이 RL 자유도 흡수 → reward를 비대칭/경로의존으로 정식화하면?'이라는 가설이 본 논문 §5의 4 reward 비교로 직결됩니다. (2) **학술 윤리**: pivot 정당화. 다만 Phase 1은 Env-v2(favorable bias)였으므로 절대 Sharpe 수치(45.39)는 비교에 안 쓰고, 정성 결론(RL = Fixed)만 인용 — caveat 명시."

### Q27. Phase 1을 정직하게 보고하면 점수 깎이지 않나요?
**1분**: "오히려 **점수 올라간다고 봅니다**. (1) negative finding은 학술 연구에서 가치가 큽니다 — null result 보고가 메타사이언스 표준 권장사항. (2) Phase 1 → Phase 3 narrative가 본 논문의 *왜 그 RQ인가*를 자연스럽게 정당화. (3) 정직성은 학술 평가의 핵심 — 결과를 미화하면 도리어 의심받습니다. 본 논문은 Phase 1 absolute numbers를 caveat로 명시하고 정성 결론만 인용하여 *학술적 엄밀성을 더 강하게* 만듭니다."

---

## F. Phase 2 환경 정상화 (Q28–30)

### Q28. favorable bias가 무엇이고 어떻게 제거했나요?
**1분**: "Phase 1에서는 다음 봉의 high/low *극값을 체결가로* 사용했습니다 — 매수는 next_low로, 매도는 next_high로. 이는 *시뮬레이션 favorable bias artifact*로 비현실적 수익(수조%)을 만들었습니다. 본 논문에서는 다음 봉 high/low로 **체결 여부만 판정**하고, 체결가는 *호가 가격 그대로* 사용. 이는 실제 지정가 주문의 작동 방식과 일치합니다."

### Q29. 학습 안정화 패키지 4가지가 뭔가요?
**1분**:
- (1) **LR linear decay**: 3e-4 → 1e-5. 학습 후반의 큰 정책 업데이트 제한.
- (2) **Entropy coef annealing**: 0.01 → 0.001. 초기 탐색 충분, 후반 deterministic 수렴.
- (3) **Target KL early stop**: 0.02 초과 시 mini-batch update 조기 종료. PPO 근시안 큰 업데이트 방지.
- (4) **Best checkpoint**: 50k step마다 Val Sharpe 평가, best 저장. 후반 정책 붕괴 대비.

이 패키지는 exp030에서 검증 (best 1.97 / final 1.21 — best와 final이 큰 차이라 best ckpt 사용이 필수).

### Q30. Env 버전 v2/v3/v4 차이는?
**1분**: "(1) **v2** (Phase 1): 다음 봉 극값 체결 + favorable bias artifact. (2) **v3** (Phase 2 중반): 지정가 체결로 수정 + 4D 절대 gap action. asym Test Sharpe 1.955로 강한 사전 증거 (exp027_rl). (3) **v4** (본 논문 canonical): 2D ATR 비례 action + 지정가 체결. ATR 공식이 변동성 흡수, action은 추가 조정만. v3 → v4는 학술적 정합성(공식의 자연스러움) 위해 변경. 환경 의존성 정직 인정 — Env-v3의 강한 asym 우위가 Env-v4에서 재현되지 않음."

---

## G. exp032b 시나리오 D (Q31–35)

### Q31. exp032b 실험 설정을 설명해보세요.
**1분**: "4 reward variant (sym, asym, dsr, pt) 각각 10 시드(42-51) × 1M step 학습. PPO 표준 hyperparameter + 학습 안정화 패키지 + asym/dsr/pt의 Optuna best. 총 학습 40M step (~3h 44m). 평가는 Val 2021-2023 단일 분할의 best_checkpoint, n_eval_episodes=5. metric 6개 보고: best Sharpe, final Sharpe, MDD, Calmar, 거래수, 누적 수익률."

### Q32. 왜 시나리오 A/B/C를 사전 등록했나요?
**1분**: "**사후 합리화(post-hoc rationalization)를 방지**하기 위해서입니다. 실험 결과를 본 후에 '이 결과가 사실 이런 시나리오를 입증한 거다'라고 둘러대는 학술 부정직성을 차단하려고 결과 *받기 전*에 가능한 결과 형태를 시나리오로 등록해두고 사후에 매칭합니다. 본 논문은 A 낙관(asym/dsr/pt 압도) / B 중립(variant 차이 있으나 ATR 못 미침) / C 비관(variant 차이 미미) 세 가지를 등록했습니다."

### Q33. 시나리오 D는 사후 정의인데 사후 합리화 아닌가요?
**1분**: "**좋은 적대적 질문**입니다. 답: 사후 시나리오 D는 사후 합리화가 아닌 *honest reporting*입니다. (1) A/B/C 어디에도 안 맞는다는 사실을 정직 보고하고, (2) 사후 시나리오 D를 *새로운 가설*로 명시하지 *기존 가설의 증거*로 위장하지 않습니다. (3) D는 통계적으로 객관 측정 가능 — Cohen's d 행렬 + policy distance ratio로 객관적 cluster 분리를 보였습니다 (within d<0.30, across d>0.79). 사후 합리화라면 D의 정의가 임의(arbitrary)일 텐데, D는 *데이터가 직접 만든 분류*입니다."

### Q34. 두 클러스터의 통계적 분리를 어떻게 확인했나요?
**1분**: "두 수준에서 확인. (1) **Sharpe 분포 수준**: Pairwise Cohen's d 행렬. within-cluster (sym vs dsr) |d|=0.29, (asym vs pt) |d|=0.15 — 모두 *작은 차이* 범위. across-cluster |d| = 0.79–1.19 — 모두 *큰 차이* 범위. (2) **정책 행동 수준**: 40개 best_model의 Val trajectory 위 L2 action distance. within 평균 0.129, across 평균 0.286. 비율 **2.22×** — 같은 클러스터끼리 행동이 다른 클러스터보다 2.22배 더 유사."

### Q35. 4개 metric에서 1위가 다 다른 점이 핵심이라는데, 정확히 어떻게 다른가요?
**1분**:
- **Best Val Sharpe**: sym 1위 (1.871)
- **Final Val Sharpe**: dsr 1위 (1.204) — 학습 후반 안정성
- **MDD**: asym 1위 (2.28% 최소)
- **Calmar**: asym 1위 (0.755 최대)
- **거래 수**: sym 1위 (120) — 가장 활발
- **누적 수익률**: dsr 1위 (7.40%)

→ "Reward design은 single alpha source가 아니라 *multi-dimensional trade-off*"라는 frame의 직접 증거. 어떤 metric을 우선하느냐에 따라 best variant가 달라집니다.

---

## H. 메커니즘 exp032c (Q36–39)

### Q36. 두 클러스터가 갈리는 메커니즘을 한 줄로?
**1분**: "**Reward 형식 → 거래 빈도 + Hold 시간 → Risk profile → Sharpe-MDD trade-off**라는 인과 사슬. 손실 비대칭 (asym β=3.42, pt λ=3.30) → 매수 호가를 멀리 배치 → 거래 빈도 25-35% 감소 → conservative cluster. DSR의 sliding-window memory → 긴 holding 학습 → hold rate 2-6배 → 같은 거래 빈도에서도 다른 risk profile."

### Q37. DSR sliding-window memory가 정확히 어떻게 긴 hold를 학습시키나요?
**1분 — 가장 어려운 답**: "DSR step reward = (B·ΔA − ½A·ΔB) / (B − A²)^1.5, 여기서 A, B는 EWMA 1.2일 윈도우 1차/2차 모멘트. **짧은 holding 시 ΔA와 ΔB 둘 다에 큰 noise 발생**, 분모 (B − A²)^1.5가 작을 때 noise가 reward signal에 과대 증폭됩니다. 정책 입장에서 '짧은 holding은 학습 신호가 약하고 변동이 크다'로 인식되어 결과적으로 **더 긴 holding을 선호**하는 정책이 학습됩니다. 다른 세 변형(sym, asym, pt)은 step reward가 1-step PnL의 즉시 함수라 holding 길이에 직접 의존하지 않습니다."

### Q38. counterfactual state-grid가 뭐고 왜 중요한가요?
**1분**: "위 거래 빈도/hold rate 통계는 *실제 시장 분포 위에서 정책이 만난 state*에 한정됩니다. counterfactual은 (atr_ratio, divergence) state-grid를 격자로 나누고 *같은 state 셀에서 4 variant의 평균 action*을 비교 — 시장이 다른 분포로 보냈을 때도 변형 간 차이가 있는지 확인. 결과: 같은 state에서도 변형 간 다른 action 출력, 특히 divergence < 0 + 중간 atr_ratio (평단가 위쪽 보유 + 변동성 보통) 셀에서 sym/dsr은 좁은 매도 vs asym/pt는 hold/넓은 매도. 즉 cluster 분리가 *state 분포 차이가 아닌 정책 매핑 차이*."

### Q39. 거래 빈도 차이의 원인과 hold rate 차이의 원인은 *독립적*이라고 했는데, 무슨 뜻이죠?
**1분**: "두 메커니즘이 **separable**합니다: (1) **거래 빈도 차이**는 손실 비대칭 (asym, pt)의 효과 — 매수 호가 멀리 배치. (2) **Hold rate 차이**는 DSR sliding window의 효과 — 긴 매도 대기. 증거: dsr은 거래 빈도가 sym과 거의 동일(둘 다 ~0.073)이면서도 hold rate만 sym의 1.9배. 즉 두 효과가 다른 변형들에서 독립적으로 작용함을 표 4의 행 비교로 확인할 수 있습니다. 이 separability가 본 논문 인과 사슬의 정밀성을 보장."

---

## I. 강건성 exp033/034 (Q40–43)

### Q40. Slippage 0.02% 추가 시 결과가 어떻게 바뀌었나요?
**1분**: "4 variant 모두 **거의 일률적인 ~12% Sharpe 감쇠**: sym 1.871→1.658, dsr 1.809→1.551, asym 1.681→1.478, pt 1.667→1.459. retention 85.7-88.6%로 약 3%p 폭 안. 절대 감쇠는 변형 효과보다 슬리피지 효과가 큼. 그러나 cluster 구조는 보존 (preservation ratio 2.19× ≈ exp032b의 2.22×). MDD는 conservative cluster (asym, pt)에서 거의 변화 없고(거래 빈도 적음), aggressive cluster에서 미세 증가."

### Q41. ATR baseline을 같은 슬리피지로 재평가한 이유는?
**1분**: "초기 분석은 ATR baseline을 *슬리피지 없음* Sharpe 1.378로 두고 RL을 슬리피지 환경에서 비교하는 unfair였습니다. Phase 16a (2026-05-16)에서 ATR baseline을 같은 슬리피지(0.02%)로 재평가 → ATR Val Sharpe **0.835** (-39% 감쇠), MDD 15.27%로 증가. ATR baseline이 매 시점 4개 호가 활용으로 슬리피지 노출이 RL보다 *더 큼*. 공정 비교에서 4 RL variant 모두 ATR-slip 대비 **+75~99% 초과**, 'conservative cluster가 ATR 못 미치는 caveat'가 깨끗하게 정리됩니다."

### Q42. CPCV 6-fold 15 paths를 어떻게 만들었나요?
**1분**: "Train + Val 구간(2017-10 ~ 2023-12, ~54,087 봉)을 시간 순서로 6 group 분할 (각 ~9,014 봉 = ~12.5개월). C(6,2) = 15가지 (train 4 groups, test 2 groups) 조합 생성. train ~36,000봉 (~4.1년), test ~18,000봉 (~2년). **Purge ±168시간**(=1주, state rolling window와 정합)을 train/test 경계에 두어 정보 누설 차단. 총 학습량 60M step (~5h 27m)."

### Q43. CPCV에서 왜 dsr이 1위로 reversal인가요?
**1분**: "**메커니즘 답변**: DSR의 sliding-window memory가 학습시키는 *긴 holding*은 단일 분할의 특정 시기 의존성을 감소시키고, *다양한 시기에 걸쳐 robust한 정책 행동*을 산출합니다. CPCV의 15 path는 BTC 2017-2023의 다양한 시장 레짐 (2018 bear, 2019 sideways, 2020 COVID, 2021 ATH, 2022 crash, 2023 recovery)을 포함, 시기-의존성이 약한 정책이 path 평균에서 우위. DSR mean Sharpe 1.413 (IQM 1.433, 5% CVaR 0.890) 모두 1위, std도 가장 작음(0.378). 단, 이 강점이 OOS bull market에서는 정확히 반대로 작동 → Test 꼴찌 (exp035 §J 참조)."

---

## J. OOS와 H5 (본 논문 핵심) (Q44–51)

### Q44. Test 결과를 정확한 수치로 말해보세요.
**1분**:
- **exp032b source** (n=10): sym +0.090, asym +0.173, **dsr -0.122**, **pt +0.367** ★
- **exp034 source** (n=15): sym +0.001, asym +0.175, dsr +0.070, **pt +0.339** ★
- ATR baseline = -0.055
- pt 양 source 일관 1위, **p<0.0015 (exp032b), p<0.0004 (exp034)** — Bonferroni 4-way 후에도 유의 (<0.008)
- dsr이 exp032b source에서 음수 Sharpe로 ATR보다 *더 나쁨* — CPCV 1위에서 Test 꼴찌로 완전 reversal

### Q45. 세 환경 세 winner reversal의 의미는?
**1분**: "Val에서 sym(1.871), CPCV에서 dsr(1.413), Test에서 pt(0.367/0.339) — 각 환경마다 다른 winner. **'X variant가 reward design의 최선'이라는 단일 결론은 평가 환경 선택에 강하게 종속**되며, 단일 환경 결과를 보고하는 관행은 학술적으로 부정확합니다. Henderson 2018의 reproducibility 우려(같은 알고리즘이라도 시드별 결과 변동)를 *reward 비교의 차원*으로 확장한 결과입니다. 본 논문 권고: RL 트레이딩 평가 표준 프로토콜로 **Val + CPCV + sealed Test 세 환경 동시 사용**."

### Q46. PT가 OOS에서 robust한 메커니즘은?
**1분 — 본 논문 main contribution**: "PT의 reward 함수는 loss aversion λ=3.30 + concave gain α=0.683을 포함. 강한 손실 패널티와 오목한 효용은 정책에 *'불확실한 미래 이익보다 지금의 확실한 작은 이익을 실현'* 인센티브 부여. Phase 16d 분석 결과 학습된 정책의 hold duration은 **mean 1.39h, median 1.0h, max 6h (short-bounded)**. Test의 BTC bull market 구간 (\$42K → \$75K) 에서 이 행동은 매수 후 가격이 크게 움직이기 *전*에 청산 완료 → sell-side timing risk 자연 회피. MDD ~2.3% 유지, Sharpe 0.34-0.37 양수."

### Q47. DSR이 OOS에서 실패한 메커니즘은?
**1분**: "DSR sliding-window가 학습시킨 *긴 holding*이 정확히 반대 부호로 작동. Test의 hold duration: **mean 4.58h, median 1.0h, max 169h (= 7일!)**. BTC가 7일 동안 크게 움직이면 정책의 sell 호가가 시장에서 멀리 떨어져 fill이 지연 → sell-side timing risk 폭증 → Test Sharpe 음수 (exp032b -0.122, exp034 +0.070 near zero). **같은 reward formulation의 같은 행동 효과(긴 holding)가 in-sample(CPCV 1위)과 OOS(Test 꼴찌)에서 반대 부호로 작동** — 본 논문 frame의 핵심 입증."

### Q48. OOS 차이가 정책이 변한 탓인지 시장이 변한 탓인지 어떻게 확인했나요?
**1분**: "Phase 16d Menu 2에서 **같은 best_model의 Val/Test 행동 통계를 비교**. 결과: 4 variant 모두 trade rate Δ ≤ 0.006, hold rate Δ ≤ 0.004, action mean Δ ≤ 0.051 — **모두 5% 미만**. 즉 같은 정책이 *거의 동일한 행동*을 출력. 그런데 시장은: Phase 15b 비교 결과 변동성 0.96% → 0.70% (**-27%**), KS test **p < 10⁻¹⁰**. **결론: 정책은 robust하게 같은 행동을 출력하지만 그 행동이 다른 시장에서 다른 결과를 낳음** — OOS gap의 원인은 정책이 아닌 *시장 distribution shift*."

### Q49. B&H가 Test Sharpe 0.757로 모든 RL을 초과하는데 본 논문 주장이 의미 있나요?
**1분 — 핵심 적대 질문**: "Phase 16b에서 정직하게 인정했습니다. **단순 Sharpe 기준에서는 B&H가 0.757로 모든 RL을 초과** — 2024+ BTC가 +78% 상승하는 강세장에서 보유만 해도 risk-adjusted 수익이 양수입니다. 그러나 **B&H MDD는 50.08%로 극단적**, Calmar (Sharpe/MDD)는 0.015로 매우 낮습니다. pt의 Calmar는 0.159로 **B&H 대비 약 10배 우위**. 본 논문의 'pt OOS robust' 주장은 *risk-adjusted return의 의미에서 성립*하며 *absolute return의 의미가 아닙니다*. 이는 발표에서 정직하게 말해야 하는 부분."

### Q50. H5가 본 논문의 *진짜* main contribution이라는 의미는?
**1분**: "H1~H4는 사전 등록 가설이지만, H5는 *사후 발견*입니다. 사전 등록을 안 했다는 사실은 학술적으로 무게가 작아질 수도 있지만, 본 논문은 (1) H5를 *새 가설*로 명시하고 *기존 가설의 증거*로 위장 안 함, (2) Test 봉인 해제 *후* 발견했음을 timeline에 정직 명시, (3) 두 source 일관 (exp032b + exp034) p<0.002로 robust 확인. **그리고 H5는 본 논문이 알기로는 Kahneman의 prospect theory가 RL trading 정책에 부여하는 OOS 안전성의 첫 정량 확인**이라는 학술적 의의를 가집니다. 행동경제학과 RL trading 교차점에서 새로운 응용 차원."

### Q51. 인간 표준값보다 더 극단적인 (α, λ)가 RL에 유리한 이유는?
**1분**: "Prospect theory 인간 표준값은 α=0.88 (약한 오목성), λ=2.25 (중간 손실회피) — 사람이 실제 의사결정에서 보이는 행동에 fit된 값. 본 논문 Optuna best는 α=0.683 (더 강한 오목성), λ=3.303 (47% 더 강한 손실회피). 해석: BTC 그리드 트레이딩의 *OOS robust 수익을 위한 RL 정책은 인간보다 더 보수적인 효용함수가 유리*. 이는 인간 트레이더가 자주 표현하는 '이익을 너무 빨리 실현한다' 행동 패턴이 OOS 안전성 측면에서 합리적임을 시사합니다. **실용 권고**: 행동경제학의 인간 추정값을 RL reward로 그대로 채택하지 말고 도메인 데이터로 hyperparameter 재튜닝하라."

---

## K. 결론 + 한계 (Q52–55)

### Q52. 본 논문의 메인 메시지 4가지를 한 줄씩?
**1분**:
- (1) **Reward variant 영향 = risk profile trade-off** (Pareto frontier, 시나리오 D)
- (2) **단일 metric winner = 평가 환경 종속** (세 환경 세 winner)
- (3) **Prospect theory가 OOS robust 정책 산출** (H5, 양 source 일관 p<0.002)
- (4) **In-sample 우위 ≠ OOS 일관성** (DSR 양면성 — 같은 메커니즘 반대 부호)

### Q53. 한계를 정직하게 5가지 말해보세요.
**1분**:
- (1) **자산 단일**: BTC/USDT 1시간봉만. 주식 ETF, FX, 원자재 미검증.
- (2) **Test regime 단일**: 2024+가 강세장 단일 → bear, sideways에서도 성립할지 미확인.
- (3) **Slippage 단일 수준**: 0.02%만. 다단계 sensitivity 미수행.
- (4) **DR 미적용**: domain randomization으로 dsr OOS 실패 회복 가능한지 미확인.
- (5) **학습 길이 단일**: 1M step만. 더 긴 학습이 결과 변경 가능성.

추가로: **exp027_rl 환경 의존성** 정직 인정 — Env-v3에서 asym Test Sharpe 1.955였으나 Env-v4에서 1.681로 sym보다 낮음. 환경 효과 > reward variant 효과.

### Q54. Future work 우선순위 3개는?
**1분**:
- (1) **Multi-asset 검증**: pt OOS robust를 주식 ETF, FX 등에서 검증. 짧은-hold 메커니즘 일반화 정량.
- (2) **Meta-RL adaptive (α, λ)**: 시변 환경에서 (α, λ)를 online 적응하는 meta-RL.
- (3) **Domain Randomization 통합**: DR로 dsr OOS 실패가 회복되는지 — 같은 reward 양면성이 DR로 해소 가능한지 정량.

### Q55. 본 논문 발견을 한 줄로 요약하면?
**10초**: "Reward design은 RL 알파의 source가 아니라 risk profile trade-off의 차원이며, Kahneman의 prospect theory가 RL trading 정책에 OOS robust를 부여한다."

---

## L. 선행 연구 (Q56–58)

### Q56. 가장 직접적 선행 연구는 누구이며 어떻게 다른가요?
**1분**: "(1) **Liu 2021** — LSTM 가격 예측 + PPO 2단계 구조. 본 논문은 가격 예측 단계 *완전 제거* (변동성 적응 그리드만). (2) **Yasin 2024** — DQN/PPO/A2C + 20 기술지표, *이산 action*. 본 논문은 *연속 2D action*. (3) **Pham 2025** — DQN으로 5개 사전 전략 선택. 본 논문은 *전략 자체를 학습*. (4) **Bandarupalli 2025** — PPO + 변동성 패널티 + 거래비용 항. *단일 reward의 hyperparameter 조정*에 머묾, 본 논문은 *reward 형식 4가지 통제 비교*."

### Q57. 본 논문이 Gort 2022로부터 무엇을 받았나요?
**1분**: "Gort et al.(2022)은 암호화폐 DRL에서 *단일 train/test 분할 결과가 OOS 일반화를 보장하지 않음*을 실증하고 CPCV와 Deflated Sharpe를 권장했습니다. 본 논문은 이 권장사항을 채택하여 (1) Val (2021-2023) / CPCV 6-fold (Train 내부) / Test (2024-) 세 환경 동시 사용, (2) Test partition을 exp035 단계까지 봉인, (3) CPCV 결과에 Bonferroni + Deflated SR 검정 적용. **본 논문의 winner reversal 발견은 Gort et al.의 우려가 reward 변형 선택 차원에서도 동일하게 나타남을 정량 확인**한 것."

### Q58. Henderson 2018의 RL reproducibility 권고를 어떻게 따랐나요?
**1분**: "Henderson et al.은 RL이 동일 알고리즘이라도 시드 차이만으로 결과가 크게 변동함을 실증, **5-10 시드의 통계 비교, IQM, 부트스트랩 신뢰구간**을 표준 권장. 본 논문은 (1) variant당 10 시드(Val/exp033), 100 모델(Test), (2) 페어와이즈 Cohen's d, (3) 부트스트랩 P(A>B) 10⁴ resamples, (4) IQM(Interquartile Mean)과 5% CVaR을 보조 metric으로 보고."

---

## M. 적대적 / 곡구 질문 (Q59–63)

### Q59. 10 시드는 부족하지 않나요? 100 시드 정도는 해야?
**1분**: "Henderson 2018 권장이 5-10 시드입니다. (1) variant당 10 시드 + bootstrap CI로 통계 신뢰성 확보, (2) Test 단계에서는 100 모델 (4 var × 10 seed + 4 var × 15 path) 평가 — 사실상 100 시드. (3) Cohen's d로 효과 크기를 정량화하므로 시드 추가가 결정적 결론을 바꾸지 않을 것 (effect size가 |d|>0.79로 매우 큼). 시드 추가는 future work의 일부."

### Q60. PPO만 썼는데 SAC, TD3는 결과가 다를 수도 있지 않나요?
**1분**: "정확한 지적이며 한계로 인정합니다. 본 논문 한계 5번째와 future work에 명시. 단, PPO는 안정성과 reproducibility로 RL trading 표준 baseline이므로, **algorithm 효과를 통제하고 reward 효과를 격리**하는 본 연구 목적에는 적합. 다른 algorithm에서 reward 변형 효과가 어떻게 다른지는 별도 future work — 'algorithm × reward 2-way comparison'으로 확장 가능."

### Q61. ATR 비례 격자 공식의 계수 (A_b, B_b 등)도 hyperparameter인데 fix한 이유는?
**1분**: "(A_b, B_b, ...)는 Phase 2 Optuna Trial #34로 한 번 결정 후 모든 RL 학습에서 *환경 상수*로 fix. 이유: (1) 이 계수는 *grid 자체의 baseline 폭*을 결정 — RL action의 작용 범위를 정의. RL과 같이 최적화하면 search space가 폭발. (2) Bayesian으로 최적화된 ATR 계수 자체가 성능의 주요 원동력임이 Phase 1에서 드러남 (RL=Fixed ablation). (3) 계수가 fix되어야 reward 변형 효과를 격리할 수 있음 — *변수 하나 변경하라*는 통제 실험 원칙."

### Q62. Test에 한 번이라도 일찍 봤다면 결과가 달랐을 가능성은?
**1분**: "그게 백테스트 과적합의 핵심 우려입니다. 본 논문은 그것을 피하기 위해 exp035까지 봉인. 다만 정직하게 말하면, hyperparameter는 Train + Val에서 결정했으므로 Val에 과적합되었을 가능성은 있고, 그것이 Val→Test gap (모든 변형 1.30~1.93)의 일부 원인입니다. 본 논문은 이 gap을 시장 distribution shift (KS p<10⁻¹⁰)로 정량화하여 *정책 과적합과 시장 변화를 분리*했습니다. 만약 Test를 일찍 봤다면 이런 정량적 분리 자체가 불가능."

### Q63. 본 연구의 가장 큰 약점은?
**1분 — 자기 비판**: "솔직히 두 개. (1) **Test regime이 강세장 단일**입니다. pt의 '짧은 holding이 sell-side timing risk를 회피한다'는 메커니즘이 bear 시장에선 hold 자체가 손실 누적이라 다른 메커니즘이 작용할 수 있습니다. (2) **자산 단일** — BTC가 leverage cycle을 가지는 특수한 자산이라 주식이나 FX에서 같은 결론일지 미확인. 이 두 가지 한계가 Future work 우선순위입니다. *발표할 때 이 약점을 먼저 말하는 것*이 학술적으로 더 안전합니다."

---

## 부록: 30초 안에 막힌 답 회피용 phrase

- "그 부분은 본 논문 §X에 정확한 수치가 있는데 핵심만 말씀드리면 …"
- "정확히는 [모름]이지만 메커니즘은 [확실]입니다."
- "그 점은 본 논문이 인정한 한계 중 하나입니다."
- "Future work로 남겨두었습니다."
- "그건 사후 발견(H5)이라서 사전 가설로 등록하지 않았습니다."

---

**총 63문.** 모든 질문에 *10초 답*은 막힘 없이 — *1분 답*은 더듬어도 핵심은 빠뜨리지 않게.
