# Volatility Modeling — GARCH, Realized Vol, Jumps

> Engle (1982) — ARCH. Bollerslev (1986) — GARCH. Andersen & Bollerslev (1998) — Realized Variance.
> Bitcoin 특화: Catania et al. (2019), HAR-GARCH with Jumps for Bitcoin (Zahid 2022).

## 요지

1. **ATR만으로는 변동성을 모두 포착할 수 없음**. ATR은 단순 평균이라 비대칭성, 군집 효과, 점프를 못 잡음.
2. **GARCH 계열**은 변동성 군집(volatility clustering)을 모델링 → 어제 변동성이 크면 오늘도 클 확률.
3. **Realized Volatility**는 일중 데이터에서 계산한 직접 변동성 → 더 정확한 추정. **HAR(Heterogeneous AutoRegressive)**은 단/중/장기 변동성을 함께 모델링.
4. **점프(Jump) 모델**: BTC는 갑작스러운 가격 점프가 흔함 → ETF 승인, 해킹, 규제 발표 등에서 ATR이 사후적으로만 반응.

## ATR vs GARCH vs Realized Vol 비교

| 모델 | 정의 | 강점 | 약점 |
|---|---|---|---|
| **ATR(N)** | $\frac{1}{N}\sum_{i=1}^{N} TR_i$ | 단순, 직관적, 즉시 계산 | 평균이라 fat tail 무시, 점프에 늦게 반응 |
| **EWMA Vol** | $\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_t^2$ | 최근 정보 가중 | 비대칭성 미반영 |
| **GARCH(1,1)** | $\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$ | 군집 효과 모델링, 검증된 통계 | mean-reverting 가정 |
| **EGARCH** | log variance를 모델링 | 비대칭성(레버리지 효과) 포함 | 추정 복잡 |
| **Realized Vol (RV)** | $\sqrt{\sum r_{t,i}^2}$ (일중) | 사후적으로 정확 | 점프와 일반 변동성 구분 안 됨 |
| **HAR-RV** | $RV_t = \alpha_d RV^d + \alpha_w RV^w + \alpha_m RV^m$ | 단/중/장기 결합 | 사후적, forecasting에는 lag |
| **Jump-Diffusion** | $dP = \mu dt + \sigma dW + J dN$ | 점프 명시 | 추정 어려움, 계수 해석 까다로움 |

## GARCH(1,1) 수식

```
r_t = μ + ε_t,  ε_t ~ N(0, σ_t²)
σ_t² = ω + α ε_{t-1}² + β σ_{t-1}²

조건:
  ω > 0, α ≥ 0, β ≥ 0
  α + β < 1  (정상성 stationarity)
```

→ α + β가 1에 가까울수록 변동성 충격이 오래 지속. BTC에서 α+β ≈ 0.98 흔함.

## HAR-RV 모델 (실용성 최고)

```
RV_t = α_d × RV_{t-1}^{(1d)}  + α_w × RV_{t-1}^{(1w)}  + α_m × RV_{t-1}^{(1m)}  + ε_t

(d, w, m: daily, weekly, monthly)
```

→ 단순 선형회귀로 추정. GARCH보다 forecasting 성능 좋음 (Corsi 2009).
→ Bitcoin에서 검증됨 (Zahid 2022, "HAR-GARCH with Jumps"가 RGARCH보다 우수).

## Bitcoin 특화 발견

### Jump의 존재
- BTC의 약 30% 일간 큰 움직임이 점프 (불연속) 성분
- ATR/GARCH가 평활화로 잡지 못함

### Leverage effect 약함
- 주식: 가격 하락 → 변동성 증가 (강한 음의 상관)
- BTC: 양방향 모두 큰 변동성 (asymmetric 약함)

### Long memory
- BTC RV는 long-memory 특성 (HAR가 잘 맞는 이유)
- 짧은 ATR window는 정보 손실

## 우리 프로젝트와의 연결점

1. **현재 상태: ATR(168)만 사용**
   - State[4] = ATR(168) / price
   - Action 스케일링도 atr_ratio 기반
   - **단점**: ATR(168)은 168봉(7일) 평균 → 갑작스러운 변동성 변화에 늦게 반응

2. **즉시 적용 가능한 개선** (State 확장):
   ```yaml
   state_features:
     - atr_168 / price         # 기존 (long-term vol)
     - realized_vol_24h        # 신규: 단기 변동성
     - garch_h_predict         # 신규: GARCH forecast
     - jump_indicator          # 신규: 점프 dummy (BPV ratio)
   ```

3. **HAR-RV로 변동성 forecast** (중기):
   - state에 next-bar volatility forecast 포함
   - 그리드 폭을 forecast 기반으로 동적 조정 (현재는 lagging ATR만)
   - 학습 정책이 forecast를 활용해 더 빠르게 적응

4. **Jump-robust ATR 변형**:
   ```python
   # 일반 ATR: True Range 평균
   # Jump-robust: Bi-power variation (BPV) — 점프 제거
   BPV_t = (π/2) × (1/N) × Σ |r_{t-i}| × |r_{t-i-1}|
   ```

5. **점프 감지 dummy variable**:
   ```python
   # |return| > 3 × ATR(168) → jump = 1, 평소 = 0
   # state에 jump 신호 추가 → RL이 점프 후 행동 다르게
   ```

6. **2학기 자산 확장 고려사항**:
   - 주식 (SOXL, NVDA): 실적발표, 오버나이트 갭이 점프 → jump 모델 필수
   - 외환: 변동성 낮음, GARCH 표준
   - 원자재: 계절성 → seasonality + GARCH

## 비용 vs 효익

| 개선 | 구현 비용 | 예상 효과 |
|---|---|---|
| Realized Vol 24h 추가 | 1일 | 단기 변동성 인식 향상, marginal |
| HAR-RV forecast | 2~3일 | 의미 있는 향상 가능 |
| GARCH forecast | 3~5일 | 검증 필요, complexity 증가 |
| Jump dummy | 1일 | 단순하지만 효과 있을 수 있음 |
| EGARCH (asymmetric) | 4~5일 | BTC에서는 marginal |

→ **우선순위: Jump dummy → Realized Vol 24h → HAR-RV.**

## 백링크

- [[avellaneda_stoikov_2008]] — Market making이 변동성을 직접 사용
- [[distributional_rl]] — Fat tail 분포 학습은 변동성 모델과 자연 결합
- [[volatility_harvesting_grid]] — 그리드의 short-vol 본질

## 출처

- [Forecasting Bitcoin Volatility Using Hybrid GARCH Models (MDPI 2022)](https://www.mdpi.com/2227-9091/10/12/237)
- [HAR-GARCH with Jumps for Bitcoin (Zahid 2022)](https://www.researchgate.net/publication/359589383_Modeling_and_Forecasting_the_Realized_Volatility_of_Bitcoin_using_Realized_HAR-_GARCH-type_Models_with_Jumps_and_Inverse_Leverage_Effect)
- [Improving Realized GARCH for Bitcoin (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1062940820300620)
- [LSTM-GARCH Hybrid for Crypto (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10013303/)
