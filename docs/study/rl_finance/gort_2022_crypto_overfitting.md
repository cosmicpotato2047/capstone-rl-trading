# DRL for Crypto: Practical Approach to Backtest Overfitting — Gort et al. (2022)

> Gort, B., Liu, X-Y., Sun, X., Gao, J., Chen, S., Wang, C. D. (2022). _Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting._ ICAIF '22 workshop, AAAI '23 showcase. arXiv: 2209.05559.
> 코드: github.com/berendgort/FinRL_Crypto

## 요지

1. **DRL 트레이딩 논문의 "수익률 부풀리기" 문제를 정면으로 다룬 첫 실용 논문.**
   대부분 DRL 트레이딩 논문이 in-sample/lucky-seed 수익만 보고하고 out-of-sample 검증이 부실하다는 비판에서 출발.
2. 해법: **백테스트 과적합 감지를 hypothesis test로 정식화** → 과적합 확률(PBO, Probability of Backtest Overfitting) 추정 → 임계값 초과 에이전트 reject.
3. 결과: 10개 코인, 2022 5월~6월(크래시 두 번 포함) 테스트셋에서 **less-overfitted 에이전트가 more-overfitted 대비 우수, equal-weight·S&P DBM 인덱스도 상회**. 과적합 46% 감소.

## 핵심 절차

```
1. 동일 환경에서 K개 DRL 에이전트 학습 (다른 seed/하이퍼파라미터)
2. Walk-forward로 N개 fold 생성
3. 각 에이전트 × 각 fold에서 Sharpe(또는 reward) 측정
4. CSCV (Combinatorially Symmetric Cross-Validation, López de Prado)로
   "in-sample 최고가 out-of-sample 중간 이하로 떨어질 확률" 계산 = PBO
5. PBO > 임계값(예: 0.5)이면 reject
6. 통과한 에이전트만 ensemble 또는 단일 선택
```

PBO 자체는 López de Prado가 만든 통계량. Gort는 이를 **DRL 트레이딩에 처음 체계적으로 적용**한 게 기여.

## 핵심 통찰

- DRL에서 흔한 함정: 한 seed로 우연히 좋은 결과 → 논문 발표 → 재현 실패
- DRL의 자유도(하이퍼파라미터 + seed + 환경 설정)가 일반 ML보다 큼 → 다중검정 보정이 더 절실
- 단순히 train/val/test split만으로는 부족, **multiple-testing-aware 검증**이 필요

## 우리 프로젝트와의 연결점

1. **CLAUDE.md의 test set 봉인 원칙의 출처가 이 논문 계열.**
   "Gort et al.(2022) 백테스트 과적합 방지 설계의 핵심"이라고 명시되어 있음.
   → 우리가 이미 인용 중이지만, **PBO를 실제로 계산해본 적은 없음**. 이것이 디펜스의 빈틈.

2. **현재 실험에 적용 가능한 즉각적 액션**:
   - exp027/exp029의 best_model을 단일로 평가하는 대신, **여러 seed + 여러 fold로 PBO 추정**
   - exp029 Val Sharpe 1.440이 PBO 0.5 이하인지 검증 → 통과하면 "과적합 아님" 객관적 디펜스 확보

3. **현재 자산별 단일 분할의 약점**:
   - Train 2017-2022 / Val 2023 / Test 2024+ 한 분할만 사용 중
   - Walk-forward로 여러 fold 생성하여 더 견고한 평가 필요

4. **exp026 체결가 버그 사례의 일반화된 교훈**:
   - DRL 결과가 코드/시뮬레이션 결함에 극도로 민감
   - Gort의 frame은 hypothesis test 형식이므로 결함 있어도 PBO가 그것을 catch

## 백링크

- [[bayesian_optimization_tpe]] — Optuna 다중 trial과 PBO 결합
- [[walk_forward_cv]] — Walk-forward + Purged K-Fold 구현
- [[finrl_framework]] — Gort의 코드 베이스가 FinRL fork
- López de Prado [[The Dangers of Backtesting]] — PBO의 원형

## 출처

- [arXiv 2209.05559](https://arxiv.org/abs/2209.05559)
- [GitHub FinRL_Crypto](https://github.com/berendgort/FinRL_Crypto)
- [OpenReview](https://openreview.net/forum?id=2U_AM7TcRQK)
