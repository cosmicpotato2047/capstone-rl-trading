# Pivot Timeline — 의사결정 narrative

> 본 논문의 **narrative 핵심**. "왜 그 시점에 그 결정을 했는가"를 시간 순으로.
> 발표의 첫 5분 hook이 여기서 나옴. 막힘없이 답할 수 있어야 함.

---

## 한 그림 요약

```
2026 1-3월        4월              5월 초          5월 중반         5월 후반        6월
─────────┬─────────┬────────────┬──────────────┬──────────────┬─────────┐
시작 가설  Phase 1   pivot         Phase 3 메인    Test 봉인 해제   본 논문
"동적 적응 negative  "reward를     "시나리오 D    "PT OOS robust  최종
 학습"     finding   비대칭/경로    (Pareto)"      (H5)" — 본
                    의존으로?"                     논문 진짜 main
```

---

## 단계 1 — 시작 가설 (2026년 1-3월)

### 그 시점 상황
- 캡스톤 디자인 1학기 시작
- RQ 초안: *"PPO 강화학습이 BTC 그리드 트레이딩에서 동적 변동성 적응을 학습하여 고정 그리드 베이스라인을 초과한다"*
- 가정: RL이 시장 상태를 보고 *자동으로* 변동성에 적응하는 정책을 학습할 것이다

### 무엇을 했나
- 환경 v2 구축 (이때는 favorable bias 인식 못함)
- State 7차원, action 2차원 (현 표준과 동일)
- sym reward, PPO 1M step

### 그때 의사결정 근거
- 직접 선행 연구 (Liu 2021 등) 가 PPO가 buy-and-hold를 초과한다고 보고
- "RL이 단순 ATR 규칙을 초과할 수 있다"가 합리적 가설

---

## 단계 2 — Phase 1 Negative Finding (2026년 4월)

### 결정적 발견
exp020, exp021, exp022 세 실험 모두에서 **학습된 정책이 사실상 상수**로 수렴.

| exp | aggressiveness 재정의 | 결과 |
|---|---|---|
| exp020 | budget_fraction (예산 사용 비율) | 모든 regime에서 a^(0) ≈ 1.0 |
| exp021 | entry_gate (진입 허용) | bear regime에서도 99.7% 진입 |
| exp022 | aggressiveness (원래) | a^(0) ≈ 0, a^(1) ≈ 0, raw output saturation |

### 결정적 ablation
exp020 결과로 RL을 Fixed [1.0, 0.0]으로 교체:

| 정책 | Val Sharpe |
|---|---|
| RL exp020 (1M step) | **45.390** |
| Fixed [1.0, 0.0] | **45.390** |

**소수점 셋째 자리까지 일치.** 1M step 학습의 결과가 사실상 상수 정책.

> ⚠️ Val Sharpe 45.39는 Env-v2 favorable bias artifact 포함. 절대값은 비교에 미사용, *RL = Fixed의 일치 자체*만 정성 결론.

### 메커니즘 진단
구조적 원인 두 가지:
1. **ATR 격자 공식이 변동성을 이미 흡수**
   - `g = ATR/price × (A + B·a)` 형태에서 ATR/price 항이 시장 변동성을 자동 반영
   - 하락장(ATR↑) → 격자 자동 확대 / 상승장(ATR↓) → 격자 자동 축소
   - RL에 남은 자유도가 매우 작음
2. **sym reward 하에서 최적해가 시점-독립**
   - 사이클 수 극대화 = 복리 누적 극대화가 유일 최적화 방향
   - 매수 가장 좁게 + 매도 가장 좁게 가 단일 최적해
   - 상태 의존 행동 학습 인센티브 사라짐

### 그때 의사결정 — 두 갈래
- **Option A**: "RL은 BTC 그리드 트레이딩에 의미 없다"고 결론 짓고 다른 도메인으로
- **Option B**: "이 negative는 *sym reward* 라는 특정 가정에 조건적"이라고 가설 세우고 reward 변형으로 pivot

→ **Option B 선택**. 근거:
- 직접 선행 연구 (Bandarupalli 2025) 가 reward 변형의 효과를 시사
- Moody 2001 DSR과 Kahneman 1979 prospect theory가 *경로 의존*과 *비대칭* reward를 정식화한 학술 기반 존재
- 졸업 논문에서 negative finding으로 끝내기보다 *negative → pivot → positive*의 narrative가 더 강함

---

## 단계 3 — Pivot의 정식화 (2026년 5월 초)

### 새 가설
> *대칭 reward + ATR 비례 공식의 조합이 RL 자유도를 흡수한다면,*
> *reward를 비대칭(asym, pt)으로 정식화하거나 경로 의존(dsr)으로 정식화하면*
> *RL이 추가 가치를 가질 수 있는가?*

### 새 RQ (현 RQ)
> BTC 그리드 트레이딩에서 reward 함수의 설계가 RL 정책의 행동 패턴과 일반화
> 성능에 어떤 영향을 미치는가? 특히, 어떤 reward 함수 하에서 RL이 ATR
> 규칙 기반을 초과하며, 그 메커니즘은 무엇인가?

### 환경 정상화 (Phase 2)
RQ 변경과 동시에 환경의 학술적 정합성 확보:

1. **체결 방식 수정** (favorable bias 제거)
   - 이전: 다음 봉 high/low를 *체결가*로 사용 → 수조% 가짜 수익
   - 이후: 다음 봉 high/low로 *체결 여부만 판정*, 가격은 호가 그대로

2. **학습 안정화 패키지 4종**:
   - LR linear decay (3e-4 → 1e-5)
   - Entropy coef annealing (0.01 → 0.001)
   - Target KL early stop (0.02)
   - Best checkpoint (50k step마다 Val Sharpe 평가)

3. **Optuna 하이퍼파라미터 탐색**:
   - asym β ∈ [1.0, 4.0] → best 3.420
   - dsr η ∈ [1/720, 1/24] → best 0.0352 (≈1/28h)
   - pt α ∈ [0.5, 1.0], λ ∈ [1.0, 4.0] → best α=0.683, λ=3.303
   - 각 30 trials × 200k step, TPE + MedianPruner

4. **사전 시나리오 등록 A/B/C** (사후 합리화 방지)

### 그때 의사결정 근거 (pivot의 학술적 정당화)
- Phase 1 결과가 *sym 특수성*에 종속이라는 가설은 합리적 추정 (메커니즘 분석)
- reward 변형이 정책에 *어떻게* 다른 영향을 주는지는 학술 미답 영역
- 4가지 변형이 4가지 이론적 계보 (PnL / 효용 1차 근사 / 위험 조정 / 행동경제학)를 대표 → 통제 비교 가능

---

## 단계 4 — Phase 3 메인 실험 (2026년 5월 중반)

### exp032b — 4 reward × 10 seed × 1M step
결과: 사전 시나리오 A/B/C 어디에도 안 맞음.

| 사전 시나리오 | 결과 매칭? |
|---|---|
| A (낙관: asym/dsr/pt 압도) | ❌ — dsr ≈ sym, asym/pt < sym in best Sharpe |
| B (중립: variant 차이는 있으나 ATR 못 미침) | ❌ — 4 variant 모두 ATR 초과 |
| C (비관: variant 차이 미미) | ❌ — within d<0.3, across d>0.79 두 클러스터 분리 |

### 사후 시나리오 D 정의
> *4 reward 변형은 단일 metric의 alpha source가 아니라 risk profile 차원의
> trade-off로 분리되며, Sharpe-MDD 평면에서 Pareto-유사 frontier를 형성한다.*

### 그때 의사결정 — 사후 시나리오 D를 정직 보고
- 데이터에 맞춘 사후 합리화가 아닌 *honest reporting* 으로 명시
- D를 *기존 가설의 증거*가 아닌 *새 가설*로 위치 — 학술 윤리
- 통계적 분리는 객관 측정 (Cohen's d 행렬, policy distance 2.22×) — 임의 분류 아님

### exp032c — 메커니즘 정량 (왜 두 클러스터?)
인과 사슬: **Reward 형식 → 거래 빈도 + Hold 시간 → Risk profile → trade-off**

- 손실 비대칭 (asym β=3.42, pt λ=3.30) → 매수 호가 멀리 → 거래 빈도 25-35% ↓
- DSR sliding-window memory → 긴 holding (hold rate 2-6배) ← *이게 본 논문의 단서*

### exp033 — Slippage 0.02% 강건성
모든 variant 일률 ~12% 감쇠, cluster preservation ratio 2.19× ≈ exp032b 2.22×

### exp034 — CPCV 6-fold 15 paths
**1차 winner reversal: Val sym(1.87) → CPCV dsr(1.41)**
- DSR의 긴 holding이 시기-robust 정책으로 작동
- Cluster preservation ratio 3.55× — multi-split에서 *더 또렷*

---

## 단계 5 — Test 봉인 해제와 H5 발견 (2026년 5월 후반)

### Test 봉인 원칙 유지
- exp035 단계까지 어떤 분석에도 Test partition 미사용
- 2026-05-16 — 사상 첫 1회 봉인 해제
- 평가 대상 100개 RL 모델 (exp032b 40개 + exp034 60개)

### 결과 — **2차 winner reversal: CPCV dsr → Test pt**

| Source | sym | asym | dsr | **pt** | ATR |
|---|---|---|---|---|---|
| exp032b (n=10) | +0.090 | +0.173 | **-0.122** | **+0.367** ★ | -0.055 |
| exp034 (n=15) | +0.001 | +0.175 | +0.070 | **+0.339** ★ | -0.055 |

- pt 양 source 일관 1위, p<0.0015 / p<0.0004
- dsr이 exp032b source에서 *음수 Sharpe로 ATR보다 더 나쁨* — CPCV 1위에서 Test 꼴찌로 완전 reversal

### 메커니즘 답변 (Phase 16d)
**Hold duration이 결정**:

| Variant | mean (h) | max (h) |
|---|---|---|
| sym | 2.15 | 98 |
| asym | 1.40 | 7 |
| dsr | **4.58** | **169 (7일!)** |
| pt | **1.39** | **6** |

- **pt OOS robust 메커니즘**: short-bounded hold → bull market sell-side timing risk 회피
- **dsr OOS 실패 메커니즘**: 긴 hold → bull market에서 7일간 가격 멀어짐 → sell 호가 fill 지연

### 정직 인정 사항
- **정책 안정성 vs 시장 shift**: 같은 model Val/Test 행동 Δ ≤ 5% vs 시장 KS p<10⁻¹⁰
  → OOS gap의 원인은 정책이 아닌 *시장 distribution shift*
- **B&H 정직 인정** (Phase 16b): B&H Sharpe 0.757 > pt 0.367이지만 Calmar는 pt 10배 우위
  → 본 논문 'pt OOS robust' 주장은 *risk-adjusted* 의미에서 성립

### H5 (사후 발견, 본 논문 진짜 main contribution)
> **Prospect-theoretic reward로 학습된 RL 정책은 unseen market regime에서**
> **다른 reward 변형보다 더 robust한 OOS 성능을 보인다,**
> **그 메커니즘은 짧은 holding을 통한 sell-side timing risk 회피이다.**

- 본 논문이 알기로는 prospect theory가 RL trading 정책에 부여하는 OOS 안전성의 **첫 정량 확인**
- 인간 표준값(α=0.88, λ=2.25)보다 더 극단적인 (α=0.683, λ=3.303)이 RL에 유리 → 응용 권고

---

## 단계 6 — 논문 작성 (2026년 6월)

### 본 논문의 9 chapter narrative
1. **Introduction**: RQ + Phase 1 motivation
2. **Related Work**: 9개 인용 + 직접 선행 4편
3. **Method**: MDP, 4 reward, PPO 안정화
4. **Phase 1 Negative**: motivation 재인용 (Env-v2 caveat)
5. **Pareto Frontier**: exp032b 시나리오 D
6. **Mechanism**: exp032c 인과 사슬
7. **Robustness**: exp033 slippage + exp034 CPCV + exp035 OOS
8. **Discussion**: Val-Test shift / 환경 의존성 / multi-env 권고 / PT 응용 / 한계
9. **Conclusion**: 메인 메시지 4가지 + future work

---

## 발표용 narrative 흐름 (15분)

> "1학기에 RL이 BTC 그리드 트레이딩에서 동적 변동성 적응을 학습할 거라는 가설로 시작했는데, **Phase 1 결과가 negative였습니다 — RL이 사실상 상수 정책으로 수렴**했어요. 그래서 'sym reward + ATR 공식 조합이 RL 자유도를 흡수한다면 다른 reward는?'이라는 새 가설로 pivot 해서, 4가지 reward 함수를 통제 비교했습니다.

> 결과가 사전 등록한 A/B/C 시나리오 어디에도 안 맞아서 **사후 시나리오 D — Pareto frontier**로 정직 보고. 4 variant가 두 클러스터로 갈리고, 각 metric별 1위가 다 다르다는 발견이 있었어요. 메커니즘은 reward의 손실 비대칭이 거래 빈도를, DSR sliding window memory가 hold 시간을 결정하는 인과 사슬.

> 강건성 검증에서 **세 환경 세 winner reversal**이 드러납니다 — Val에서 sym, CPCV에서 dsr, Test에서 pt. 그리고 본 논문의 진짜 핵심은 **Test 봉인 해제 후 발견한 H5**: Kahneman 손실 회피 reward가 미공개 강세장에서 양 source 일관 1위였고, 메커니즘은 평균 1.4시간 짧은 holding이 sell-side timing risk를 회피한다는 것. 본 논문이 알기로는 prospect theory가 RL trading에 부여하는 OOS 안전성의 첫 정량 확인입니다.

> 정직하게 인정할 부분 — B&H의 단순 Sharpe는 더 높지만 Calmar는 pt가 10배 우위, 그리고 한계 5가지 (자산 단일, Test 강세장 단일 등) 있습니다."

이게 15분 발표 한 줄 narrative.
