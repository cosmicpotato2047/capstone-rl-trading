# Project Roadmap
# PPO 기반 동적 그리드 트레이딩 → 멀티에셋 자동화 시스템

**최종 목표:** 멀티에셋 + 대시보드 + API 매매 자동화  
**전략:** ATR 비례 그리드 트레이딩 + PPO regime 적응

---

## 현재 위치

```
Phase 1 ██████████ 완료
Phase 2 ██████████ 완료 (Val Sharpe 38.330, regime 적응 확인)
Phase 3 ░░░░░░░░░░ 대기 (다음 단계)
Phase 4 ░░░░░░░░░░ 대기
Phase 5 ░░░░░░░░░░ 대기
```

---

## Phase 1 — BTC 연구 ✅ 완료 (2026-04)

**결과:**
- Bayesian 고정 공식 최적화 완료 (Val Sharpe 35.4 → Test Sharpe 43.0)
- 정책 포화 발견 → 재설계 필요성 확인
- 중간 보고서 제출 (reports/paper/main.tex)

**발견한 핵심 인사이트:**
- A_s=0.101이 탐색 하한 도달 → 빠른 매도가 최적
- Bayesian 역할(범위 결정) ≠ RL 역할(범위 안 선택) 분리 필요

---

## Phase 2 — RL 재설계 ✅ 완료 (2026-04)

**목표:** RL이 레짐(상승/하락/횡보)에 따라 실제로 다르게 행동하는가 증명

**진행 상황:**
- [x] State 5D → 7D (trend_1d, trend_1w 추가)
- [x] sell 공식 재설계 (A_s=0.05, RL 자율 범위 확장)
- [x] exp017 학습 완료 (Val Sharpe 38.186)
- [x] Optuna 하이퍼파라미터 튜닝 (39 trials, best: Trial #18 Sharpe 38.491)
- [x] exp018 최종 학습 (Val Sharpe 38.330, MDD 1.25%)
- [x] 레짐 적응 통계 검증 (Kruskal-Wallis p≈0.000, bull>sideways>bear)
- [ ] Test set 최종 평가 (1회, 봉인 해제) — Phase 3 직전 실시

**완료 기준:** Test Sharpe > 35 AND profit_target이 regime별로 통계적으로 다름

**다음 단계 조건:**
- 결과 좋음 → RL 모델로 라이브 트레이딩
- 결과 무난/나쁨 → 고정 공식으로 라이브 트레이딩, RL은 추후 개선

---

## Phase 3 — BTC 라이브 트레이딩

**목표:** 실거래에서 안정적으로 동작하는 봇 검증

**선행 조건:** Phase 2 완료

**단계:**
1. Testnet 2주 → 오류 없이 동작 확인
2. 실거래 소액 ($100) → 1개월 수익/손실 모니터링
3. 대시보드 v1 (Streamlit) — 포지션, 수익률, MDD 실시간 표시
4. 금액 단계적 증액

**완료 기준:** 1개월 실거래 Sharpe > 1, MDD < 15%, 무중단 운영

---

## Phase 4 — 멀티에셋 확장

**순서 (변동성·적합성 기준):**

### 4-A: SOXL (반도체 3배 레버리지 ETF)
- **선행 조건:** Phase 3 완료
- 변동성 높아 그리드 적합, BTC와 구조 유사
- 핵심 과제: 장중(9:30~16:00 ET)만 거래 → 장마감 주문 처리
- **레버리지 주의:** regime 감지 필수 (하락 추세에서 손실 3배)
- 데이터: yfinance / Alpaca API

### 4-B: 원달러 환율 (KRW/USD)
- **선행 조건:** 4-A 완료
- 한국 시장 특수성, 중간 변동성
- 핵심 과제: FX 브로커 API 연결, 스프레드 비용 모델 적용
- 데이터: OANDA / Interactive Brokers API

### 4-C: PDBC (원자재 분산 ETF)
- **선행 조건:** 4-B 완료
- 낮은 변동성 → 그리드 효율 낮음, 마지막 검증
- 핵심 과제: 변동성 작을 때 수수료 커버 가능한지 확인

---

## Phase 5 — 통합 대시보드 + 완전 자동화

**목표:** 전 자산 통합 관리 시스템

**구성요소:**
- 대시보드 v2: 전 자산 실시간 P&L, 포지션, MDD
- 리스크 관리: 자산별 자본 배분, 전체 포트폴리오 MDD 한도
- 알림: 비정상 동작, MDD 경고, 사이클 이상 감지
- 서버: 24/7 무중단 운영 (AWS / GCP / 자택 서버)

---

## 기술 스택

| 영역 | 현재 | 확장 시 |
|------|------|---------|
| RL | PPO (stable-baselines3) | 동일 |
| 거래소 | Binance (ccxt) | Alpaca (주식), OANDA (FX) |
| 데이터 | ccxt / yfinance | yfinance / broker API |
| 대시보드 | — | Streamlit → 필요 시 React |
| 서버 | 로컬 | AWS EC2 / GCP |
| 상태 저장 | sqlite | sqlite (확장 시 PostgreSQL) |

---

## 의사결정 기록

| 날짜 | 결정 | 근거 |
|------|------|------|
| 2026-04 | Bayesian 계수 고정, RL action 역할 분리 | 정책 포화 해결 |
| 2026-04 | A_s=0.05 (하한 0.1 이하 확장) | Bayesian 탐색 하한 도달 신호 |
| 2026-04 | PDBC 마지막 순서 | 낮은 변동성으로 그리드 비효율 |
| 2026-04 | SOXL 2순위 | BTC와 구조 유사, 높은 변동성 |

---

## 관련 문서

| 문서 | 경로 | 내용 |
|------|------|------|
| 실험 기록 | `RESEARCH_LOG.md` | 날짜별 의사결정 + 실험 결과 |
| 코딩 규칙 | `CLAUDE.md` | 개발 규칙, MDP 설계 |
| 라이브 봇 설정 | `live_trading/config.yaml` | 공식 계수, 거래 설정 |
| 중간 보고서 | `reports/paper/main.tex` | Phase 1 학술 정리 |
