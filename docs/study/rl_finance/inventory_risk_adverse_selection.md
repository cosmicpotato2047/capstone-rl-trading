# Inventory Risk & Adverse Selection — Market Microstructure

> Glosten, L. R., & Milgrom, P. R. (1985). _Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders._
> Kyle, A. S. (1985). _Continuous Auctions and Insider Trading._
> Ho, T., & Stoll, H. (1981). _Optimal Dealer Pricing under Transactions and Return Uncertainty._

## 요지

1. **Bid-Ask spread는 두 가지 비용을 보상한다**:
   - **Inventory cost** — 마켓 메이커가 원치 않게 보유하게 된 재고의 가격 변동 위험
   - **Adverse selection cost** — 정보 우위 거래자 (informed trader) 와 거래할 때의 손실
2. **Glosten-Milgrom (GM)**: adverse selection만 모델링 (확률적 informed trader)
3. **Kyle**: market depth (lambda)를 통한 정보 흐름 모델 — batch auction
4. **Ho-Stoll**: inventory risk만 모델링 — Avellaneda-Stoikov의 직접 선조

## Glosten-Milgrom 모델

```
시장 참가자:
  - 마켓 메이커 (정보 없음, 위험 중립)
  - Informed trader (확률 α, 진짜 가치 V 알고 있음)
  - Uninformed (noise) trader (확률 1-α, 랜덤 거래)

마켓 메이커 전략:
  bid b = E[V | sell order arrived]
  ask a = E[V | buy order arrived]

→ Bayesian update: 매수 주문이 오면 V가 큰 쪽일 확률 ↑
→ ask > mid > bid
→ Spread = ask - bid = 2 × E[|V - mid|] × α / (1 + uninformed share)
```

### 핵심 통찰
- **Informed trader 비율 α 클수록 spread 넓어짐**
- **가치 불확실성 클수록 spread 넓어짐**
- 마켓 메이커는 평균적으로 break-even (informed 손실 = uninformed 이익으로 상쇄)

## Kyle 모델

```
연속 경매 시장:
  Informed trader가 hidden order x_t 제출
  Noise trader가 random u_t 제출
  마켓 메이커는 총 order flow y = x + u 만 관측 (개별 식별 불가)

균형:
  Price = μ + λ · y    (λ = "Kyle's lambda" = price impact)

  λ = √(σ_v² / σ_u²)    (가치 변동성 / noise 변동성)
```

→ λ가 클수록 시장 깊이 얕음 (가격 충격 큼)
→ Informed trader는 자기 정보를 천천히 흘려 발견 안 되도록 함

## Ho-Stoll (Inventory) 모델

```
마켓 메이커의 reservation price = mid - q · γ · σ² · T

q: 현재 inventory (양수=long)
γ: risk aversion
σ²: 가격 변동성
T: 시간 horizon
```

→ inventory 늘면 reservation price 내려감 → ask 낮추고 bid도 낮춤 → 매도 favorable
→ Avellaneda-Stoikov의 직접 선조.

## 두 위험의 분리

| 위험 | 원천 | 방어 방법 |
|---|---|---|
| **Adverse selection** | Informed trader 존재 | Spread 확대 (GM), trading volume 모니터링 |
| **Inventory risk** | 가격 변동성 σ² | Reservation price 조정 (Ho-Stoll, AS) |

GM은 inventory가 0이라 가정 (Stoll: 마켓 메이커는 항상 즉시 청산 가능).
Ho-Stoll은 informed trader 없다고 가정 (모든 거래가 noise).

→ **현실은 둘 다.** Lehalle & Laruelle (2018) "Market Microstructure in Practice"가 결합 모델.

## 우리 그리드 봇 관점에서의 두 위험

### Inventory Risk
우리 시스템에서 이미 부분적 처리:

```
사이클 시작 (cash → BTC) → BTC 보유 → 가격 변동 위험
sell_cost (평단가 +N×ATR) → 손익분기점 보장
threshold_btc → 1슬롯 이하면 전량 청산 → 사이클 강제 종료
```

→ **우리의 sell_cost와 threshold_btc는 inventory risk 관리 메커니즘.**
→ AS framework의 reservation price 개념을 단순화한 형태.

### Adverse Selection
**우리 시스템은 직접 다루지 않음.**
- 우리는 limit order만 → informed trader가 우리에게 체결시키면 손실
- 1h 봉 시뮬레이션에서는 가시화 어려움
- **이게 sim2real gap의 한 원천**:
  - 백테스트: informed/uninformed 구분 없음
  - 실거래: informed가 우리 limit를 hit하면 시장이 그 방향으로 움직임 → 우리 미실현 손실

→ 라이브 트레이딩에서 adverse selection 모니터링 지표 필요:
   - **체결 후 5분 가격 변화**: 음수 빈도 ↑ → adverse selection 의심
   - **체결 후 가격 mean reversion 시간**: 길어짐 → adverse selection 의심

## 우리 프로젝트 직결

1. **현재 시뮬레이터의 한계 명시**:
   - "백테스트는 inventory risk만 다룸, adverse selection 무시"
   - Paper trading + adverse selection 측정으로 sim2real gap 정량화

2. **Hierarchical RL과의 결합 가능성**:
   - 상위 정책: market regime + estimated informed trader 비율
   - 하위 정책: spread (gap) 조정
   - → Adverse selection이 높다고 판단되면 spread 확대

3. **Volume-based features 추가**:
   - 현재 state에 volume 정보 없음
   - **Volume imbalance** (= buy_volume / sell_volume) 추가하면 informed trader 신호
   - Kyle's lambda를 state로 — 가격 충격 추정
   - Order flow imbalance — market microstructure 표준 feature

4. **Adverse selection 방어 기법** (라이브 트레이딩 단계):
   - **Toxic flow detection** — VPIN, BVC
   - **Quote update timing** — 가격 변동 직후 quote 즉시 갱신
   - **Order size minimization** — 작은 주문이 informed 노출 적음

## 백링크

- [[avellaneda_stoikov_2008]] — Ho-Stoll의 직접 후계
- [[sim2real_finance]] — Adverse selection이 sim2real gap의 핵심 원천
- [[realistic_execution_simulation]] — Order flow / queue position
- [[volatility_modeling]] — σ² 추정 (inventory risk의 입력)

## 출처

- [Glosten & Milgrom (1985) PDF (Buffalo)](https://www.acsu.buffalo.edu/~keechung/MGF743/Readings/B3%20Glosten%20and%20Harris,%201988%20JFE.pdf)
- [Market Making with Asymmetric Information and Inventory Risk](http://apps.olin.wustl.edu/faculty/liuh/papers/lw.pdf)
- [Information in Securities Markets: Kyle Meets Glosten-Milgrom](https://www.researchgate.net/publication/4899142_Information_in_Securities_Markets_Kyle_Meets_Glosten_and_Milgrom)
- [Solving the Glosten-Milgrom Market Making Model](https://zerolag.club/p/lecture-5-solving-the-glosten-milgrom)
