# Walk-Forward Analysis + Purged K-Fold + CPCV

> López de Prado, M. (2018). _Advances in Financial Machine Learning._ Ch. 7, 12.
> Bailey, D. H., & López de Prado, M. (2014). _The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality._ J. Portfolio Management.

## 요지

1. **시계열 데이터에서 일반 K-fold CV는 잘못**. 미래 정보가 과거로 누설(lookahead leak).
2. **Walk-Forward Analysis (WFA)** = 단일 시간순 분할 — 직관적이지만 단일 시나리오만 평가 → 우연성 통제 못 함.
3. **Purged K-Fold + Embargo** = 시간 인접 샘플을 학습/검증 사이에서 제거 → leak 방지.
4. **Combinatorial Purged CV (CPCV)** = 여러 fold 조합으로 다중 path 평가 → 가장 강력한 시계열 평가.

## 시계열 CV의 위계

```
나쁨:  Random K-Fold CV
       └ 미래 데이터로 학습, 과거로 검증 — 명백한 leak

보통:  Walk-Forward (단일 분할)
       └ Train [t0, t1] | Val [t1, t2] | Test [t2, t3]
       └ 단점: 단일 시나리오, 우연한 분할 의존

좋음:  Walk-Forward (Sliding window)
       └ [w1] [w2] ... [wN] 순차 평가
       └ 단점: 인접 window 간 label 누설 가능

더 좋음: Purged K-Fold + Embargo
       └ 각 fold 사이에 embargo gap, label 중첩 샘플 purge
       └ N개 fold 평가 → 분산 추정 가능

최선:  CPCV (Combinatorial Purged CV)
       └ N개 fold 중 k개를 test로 선택 (N choose k)
       └ Path 다수 생성 → 통계적 path-by-path 분석
```

## Purging과 Embargo 정식

```python
# 시계열 샘플의 label horizon = h (예: 미래 24봉 수익률)
# 각 샘플 i에 대해 t_start, t_end = t_i, t_i + h

# Purging:
def is_overlap(train_sample, test_sample, h):
    return train_sample.t_end >= test_sample.t_start and \
           train_sample.t_start <= test_sample.t_end

# Embargo (test 직후 일정 기간을 학습에서 제외):
embargo_period = h  # 보통 label horizon과 같게
```

## 우리 프로젝트와의 연결점

1. **현재 분할 방식 점검**:
   - Train: 2017-08-17 ~ 2022-12-31 (44,952봉)
   - Val:   2023-01-01 ~ 2023-12-31 (8,760봉)
   - Test:  2024-01-01 ~ (봉인)
   - → **단일 walk-forward.** N=1 fold. CPCV 미적용.

2. **이게 왜 문제인가**:
   - exp027 ATR+direction이 Val Sharpe +0.65 → Test -1.15로 reversal
   - **단일 분할의 우연성**으로 설명 가능. 다른 분할이었다면 결과 달랐을 것
   - 결국 "Val에 과적합"이지 "Test에 일반화 실패"가 아닐 수 있음

3. **즉시 적용 가능한 CPCV 변형**:
   ```
   Total 데이터: 2017-08 ~ 2026-04 (8.6년)
   N=6 fold (각 ~1.4년):
     F1: 2017-08~2018-12
     F2: 2019-01~2020-04
     F3: 2020-05~2021-08
     F4: 2021-09~2022-12
     F5: 2023-01~2024-04
     F6: 2024-05~2025-08

   각 trial에서:
     Train: F1, F2, F4, F5 (예시)
     Test:  F3, F6
   → 모든 (4 train, 2 test) 조합 평가 → 15 paths

   각 path별 Sharpe → 분포 → CVaR 또는 mean ± std
   → 단일 숫자 아닌 분포로 디펜스
   ```

4. **Embargo 구체값**:
   - State에 168봉(7일) lookback이 있음 → embargo = 168봉 = 7일
   - Train과 Test 사이에 최소 7일 gap

5. **Test set 봉인 원칙과의 양립**:
   - Test set (2024+)는 봉인 유지
   - CPCV는 Train+Val 안에서 적용 (2017-08~2023-12 = ~6.4년)
   - 최종 평가만 Test에서 1회

6. **Deflated Sharpe Ratio (DSR)**:
   - Optuna 100~150 trials를 돌렸음 → 다중검정 보정 필요
   ```
   DSR = (Sharpe_obs - E[max Sharpe under H0]) / σ[max Sharpe under H0]

   E[max Sharpe] ≈ √(2 ln(N_trials)) × σ(Sharpe)
   ```
   - 직관: "랜덤하게 N번 시도하면 어느 정도 Sharpe까지는 우연으로 나오나"
   - 우리 best Sharpe가 이걸 명확히 넘어야 진짜 알파.

## 즉시 액션 (구현 가능)

1. **CPCV 인프라 구축** (2~3일):
   - 6 fold 분할기 작성
   - Purge + embargo 구현
   - 각 path 평가 자동화
2. **exp029 best_model을 CPCV로 재평가** (1주):
   - 15 paths Sharpe 분포 측정
   - mean, std, 5% quantile (CVaR-style)
3. **DSR 계산** (당일):
   - Optuna log에서 trial 분포 추출
   - DSR 공식 적용
4. **Test set 평가는 한 번만** (최종 단계):
   - CPCV 결과로 충분히 robust 확인 후 봉인 해제

## 한계

- CPCV는 컴퓨팅 비용 큼 (15 paths × 1M steps = 15M steps)
- 작은 데이터셋에서는 fold가 너무 짧아질 수 있음 → 8.6년 데이터는 6 fold가 한계

## 백링크

- [[gort_2022_crypto_overfitting]] — CPCV를 DRL trading에 적용한 첫 사례
- [[bayesian_optimization_tpe]] — Optuna trial 수와 DSR 보정
- [[The Dangers of Backtesting]] (López de Prado Ch.11) — 단일 WF의 약점
- [[Backtest Statistics]] (Ch.14) — DSR 정식

## 출처

- [Purged Cross-Validation — Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [CPCV Method (Towards AI)](https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method)
- [Reasonable Deviations — Adv FinML notes](https://reasonabledeviations.com/notes/adv_fin_ml/)
- [KFold CV with Purging and Embargo](https://antonio-velazquez-bustamante.medium.com/kfold-cross-validation-with-purging-and-embargo-the-ultimate-cross-validation-technique-for-time-2d656ea6f476)
