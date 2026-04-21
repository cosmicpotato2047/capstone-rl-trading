# Project Roadmap
# ATR 규칙 기반 vs RL 그리드 트레이딩 — 다자산 비교 연구

**최종 목표:** 자산군별로 ATR 고정 vs RL 중 더 우월한 전략을 파악하고, 각 자산에 최적 전략으로 라이브 트레이딩 운용  
**핵심 질문:** "잘 설계된 규칙 기반 시스템(ATR)과 RL을 공정하게 비교하면 어느 쪽이 우월한가? 그 답이 자산군마다 다른가?"

---

## 현재 위치

```
Semester 1  ██████████ 완료 (BTC 설계 + 검증 + 핵심 발견)
Semester 2  ░░░░░░░░░░ 진행 예정 (ATR vs RL 체계적 비교 + 다자산 확장)
```

---

## Semester 1 — BTC 시스템 설계 + 핵심 발견 ✅ 완료 (2026-04)

### 완료된 것

- [x] ATR 비례 그리드 공식 설계 (buy/sell 4개 지정가, ATR 스케일링)
- [x] Bayesian 최적화 (Optuna) 로 공식 계수 확정
- [x] 데이터 확장: BTC 전체 이력 2017~현재, 균형 분할
- [x] RL(PPO) 실험: exp018 → exp020 → exp021 Ablation
- [x] Test set 최종 평가 완료 (Test Sharpe 42.090, MDD 1.26%)

### 핵심 발견 (→ Semester 2 동기)

**ATR 비례 공식이 RL의 역할을 흡수한다.**

- RL은 항상 고정값(profit_target≈0)으로 수렴
- Fixed policy와 성능 동일 (Val Sharpe 45.390 = 45.390)
- 원인: ATR/price 항이 이미 변동성 스케일링 → RL이 배울 것 없음

**그러나 시스템 자체는 강력:**
- Test Sharpe 42.090 vs 베이스라인 최고 1.550 (27배)
- MDD 1.26% (Buy&Hold 50.1% 대비)

### 성능 결과 (BTC, Test 2023H2~2026)

| 전략 | Sharpe | Return | MDD |
|---|---|---|---|
| Buy & Hold | 0.930 | 149.6% | 50.1% |
| Best Baseline (Fixed Grid 5%) | 1.550 | 8.9% | 1.6% |
| ATR 고정 (Fixed [1.0, 0.0]) | 41.769 | — | 1.26% |
| RL (exp020) | 42.090 | 3,045,713% | 1.26% |

---

## Semester 2 — ATR vs RL 체계적 비교 + 다자산 확장

### 연구 설계

두 시스템을 동일한 공식 구조에서 비교:

| 시스템 | 설명 | 계수 결정 방식 |
|---|---|---|
| **ATR 고정** | 규칙 기반, Bayesian 최적 계수 사용 | Optuna 1회, 고정 |
| **RL** | 학습 기반, 매 스텝 계수를 동적 결정 | PPO 학습 |

→ 공식 구조 동일, 계수 결정 방식만 다름 → 공정한 비교  
→ 상세 공식: `docs/FORMULAS.md` 참고

### Phase 3 — BTC: ATR vs RL 최종 비교 (exp022)

**목표:** RL이 ATR 고정을 이길 수 있는지 확인. Semester 1 발견 재현 or 반증.

**단계:**
- [ ] exp022 구현: RL이 계수를 직접 결정하는 버전 (ATR 구조 유지)
- [ ] Optuna 재실행 (새 action space에 맞게)
- [ ] exp022 학습 + 레짐 분석
- [ ] 비교: ATR 고정 vs RL vs 베이스라인 (Val + Test)
- [ ] paper trading: 더 우수한 전략으로 BTC 운용 시작

**완료 기준:** ATR 고정 vs RL 성능 차이 확인, BTC 최적 전략 결정

---

### Phase 4 — 주식 (SOXL or AAPL/NVDA)

**왜 주식이 RL에게 유리할 수 있나:**
- 실적 발표 전후: ATR이 포착 못하는 급등락
- 오버나이트 갭: 시장 외 시간 가격 변동
- 섹터/매크로 레짐 전환 (금리 인상기 등)
- RL이 trend/volatility state를 보고 이벤트 전 포지션 축소 가능

**단계:**
- [ ] 데이터 수집 (yfinance / Alpaca)
- [ ] 전처리: 장중 시간 필터, 오버나이트 갭 처리
- [ ] ATR 고정 버전 적용 + Bayesian 최적화
- [ ] RL 버전 학습
- [ ] 비교 + 더 나은 전략으로 paper trading

**선행 조건:** Phase 3 완료

---

### Phase 5 — 외환 (EUR/USD 또는 KRW/USD)

**왜 외환이 중간 난이도인가:**
- 24/5 (주말 갭 존재)
- 금리 결정, 고용지표 발표 시 급변동
- BTC보다 낮은 변동성 → ATR 스케일링 효과 다를 수 있음

**단계:**
- [ ] 데이터 수집 (OANDA / yfinance)
- [ ] 스프레드 비용 모델 반영
- [ ] ATR 고정 vs RL 비교
- [ ] paper trading

**선행 조건:** Phase 4 완료

---

### Phase 6 — 원자재 (금 GLD or 원유 USO)

**왜 원자재에서 RL이 유리할 수 있나:**
- 강한 계절성 (수요/공급 사이클)
- 지정학적 이벤트로 레짐 전환 급격
- ATR이 방향성 있는 레짐을 포착하기 어려움

**단계:**
- [ ] 데이터 수집
- [ ] ATR 고정 vs RL 비교
- [ ] paper trading

**선행 조건:** Phase 5 완료

---

### Phase 7 — 통합 라이브 트레이딩 + 대시보드

**목표:** 각 자산에 최적 전략을 실거래로 운용

**구성:**
- 자산별 최적 전략 (ATR 고정 or RL) 결정 후 실거래 배포
- 대시보드: 자산별 실시간 P&L, 포지션, MDD
- 리스크 관리: 자산별 자본 배분, 전체 포트폴리오 MDD 한도
- 서버: 24/7 무중단 운영

---

## 기술 스택

| 영역 | 현재 | Semester 2 |
|---|---|---|
| RL | PPO (stable-baselines3) | 동일 |
| 거래소 | Binance (ccxt) | + Alpaca (주식), OANDA (FX) |
| 데이터 | ccxt / yfinance | + broker API |
| 최적화 | Optuna (Bayesian) | 동일 |
| 대시보드 | — | Streamlit |
| 서버 | 로컬 | AWS EC2 / GCP |

---

## 의사결정 기록

| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-04 | ATR 비례 공식 채택 | 절대 간격 대비 변동성 적응 우수 |
| 2026-04 | Bayesian 최적화로 계수 확정 | A_b=0.285, C_b=5.223, A_s=0.05, C_s=2.5 |
| 2026-04 | 데이터 2017~현재로 확장 | Val 레짐 균형 확보 필요 |
| 2026-04 | **Pivot**: RL 단독 → ATR vs RL 비교 연구로 전환 | RL이 ATR 공식 안에서 기여 못함 발견 |
| 2026-04 | 다자산 확장 방향 결정 | 자산별 RL 가치 조건 탐구 |

---

## 관련 문서

| 문서 | 경로 | 내용 |
|---|---|---|
| 공식 정의 | `docs/FORMULAS.md` | ATR 고정 / RL 버전 공식 분리 정리 |
| 실험 기록 | `RESEARCH_LOG.md` | 날짜별 의사결정 + 실험 결과 |
| MDP 설계 | `docs/MDP.md` | State/Action/Reward 설계 근거 |
| 실험 설정 | `config/experiment_config.yaml` | 모든 수치 파라미터 |
