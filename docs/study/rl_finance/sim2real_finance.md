# Sim-to-Real Gap in RL Trading

> 로봇틱스에서 정립된 Sim2Real 개념을 트레이딩에 적용. 시뮬레이터에서 학습한 정책이 실거래에서 성능 저하 (또는 완전 실패) 하는 문제.

## 요지

1. **시뮬레이터는 결코 실거래의 완벽한 복제가 아니다.** 차이가 크면 학습한 정책의 가정이 깨져 손실 발생.
2. 트레이딩의 sim2real gap 주요 원인: **체결 가정**, **슬리피지**, **레이턴시**, **부분체결**, **시장 충격**, **유동성 한계**, **거래소 장애**, **regime shift**.
3. 로봇 분야의 대응책 (domain randomization, time-in-state)이 트레이딩에 거의 그대로 적용 가능.

## 트레이딩에서의 sim2real gap 분류

### 1. 체결 모델 차이

| 시뮬레이터 가정 | 실거래 |
|---|---|
| 1h 봉 high/low로 체결 판단 | 호가창 큐 위치, 큐 진입 시점이 결정 |
| 지정가 즉시 체결 | 큐에서 대기 → 시장이 그 가격을 다시 안 찍으면 미체결 |
| Maker fee 고정 | Maker/Taker 동적 결정, VIP 등급별 차이 |

### 2. 레이턴시

```
시뮬레이터:  state observed at t  →  action executed at t  (zero delay)
실거래:      state observed at t  →  action executed at t + δ
            δ = local compute + network + exchange queue
```

δ는 보통 100~500ms (개인 환경 기준). 1h 봉에서는 무시 가능하지만,
짧은 타임프레임이나 변동성 급변 시에는 결정적.

### 3. 부분체결 (Partial Fill)

- 시뮬레이터: 0/1 binary (전부 체결 or 전혀 안 됨)
- 실거래: 큰 주문은 여러 호가를 거치며 일부만 체결
- → 평단가가 시뮬레이터보다 불리하게 형성

### 4. 시장 충격 (Market Impact)

- 시뮬레이터: 우리 주문이 시장에 영향 없음 가정 (price taker)
- 실거래: 큰 주문은 가격을 끌어올림/내림 (Almgren-Chriss 모델)
- 그리드 봇의 작은 사이즈에서는 무시 가능하나, 자본 증가 시 결정적

### 5. Regime shift (학습 분포와 실거래 분포 차이)

- 학습 기간 BTC: ATH+크래시+회복 모두 포함했어도
- 미래에는 본 적 없는 regime (e.g., ETF 승인 같은 구조 변화) 등장 가능
- **Non-stationarity는 sim2real gap의 본질적 원천**

## Domain Randomization (대응 1)

> 학습 중에 시뮬레이터 파라미터를 다양화 → 정책이 robust해짐

트레이딩 적용 가능 항목:
- **수수료 랜덤**: fee_rate ∈ [0.04%, 0.08%]
- **슬리피지 랜덤**: 체결가에 [-0.05%, +0.05%] noise
- **레이턴시 랜덤**: state 관측 시점을 1~3봉 지연
- **데이터 랜덤**: ATR window, log_price normalization window 변동

## Time-in-State RL (Sandha et al., CoRL 2020) (대응 2)

> "지연/sampling rate를 state에 추가해서 정책이 그것에 적응하게 만든다"

트레이딩 적용: 다음 정보를 state에 명시
- 직전 주문의 미체결 여부
- 시뮬레이터의 가정 위반 빈도 (e.g., 봉 내 가격 경로 추정 vs 실제)

## 우리 프로젝트와의 연결점

1. **현재 시뮬레이터의 sim2real gap 인벤토리**:

| Gap 항목 | 현재 가정 | 실거래 차이 | 위험도 |
|---|---|---|---|
| 체결 | 지정가 즉시 (지정가가 high/low 사이면) | 호가창 큐, 미체결 가능 | 중 |
| 슬리피지 | 0% | 0.01~0.05% | 중 |
| 레이턴시 | 0 | 200~500ms | 저 (1h봉) |
| 부분체결 | 전부/없음 | 가능 | 저 (작은 주문) |
| Maker fee | 0.05% 고정 | 0.02~0.075% (VIP) | 저 |
| 거래소 장애 | 없음 | 가끔 발생 | 저 |

→ 가장 큰 gap: **체결 가정**과 **슬리피지**.

2. **즉시 적용 가능한 대응책**:
   - **fee_rate를 [0.04%, 0.08%]로 randomize**해서 재학습 — 1주 정도 작업
   - **체결가에 ±0.02% noise** 추가 — 슬리피지 도메인 랜덤화
   - **state에 "직전 체결 성공률"** 추가 — Time-in-State 응용

3. **Paper trading이 결국 sim2real gap의 진짜 측정기.**
   ROADMAP의 Paper Trading 단계가 단순한 안전장치가 아니라 sim2real gap의 정량 측정 도구임을 명시할 것.
   - 1주일 paper trade → 실 Sharpe vs sim Sharpe 비교 → gap 수치화
   - gap > 50%이면 시뮬레이터 재설계 필요

4. **exp026 체결가 버그가 sim2real gap의 극단적 사례**:
   - 가정 깨지자 PnL이 수조% → 1자릿수 %로 붕괴
   - "현실에 가까운 시뮬레이터가 결국 더 학습 가능한 정책을 낳는다"는 교훈

## 백링크

- [[reward_hacking]] — sim2real gap이 hacking 채널의 원천
- [[domain_randomization_curriculum]] — Bundle D7
- [[realistic_execution_simulation]] — Bundle D4 (더 구체적 구현)

## 출처

- [Sandha et al. (2020) — Time-in-State RL (CoRL)](https://proceedings.mlr.press/v155/sandha21a/sandha21a.pdf)
- [Survey of Sim-to-Real Methods in RL (2025)](https://arxiv.org/abs/2502.13187)
- [CMU 10-703 — Sim2Real Lecture](https://www.andrew.cmu.edu/course/10-703/slides/Lecture_sim2realmaxentRL.pdf)
