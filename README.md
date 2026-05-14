# BTC 그리드 트레이딩 — Reward Design이 RL 알파에 미치는 영향

2026 컴퓨터공학과 캡스톤 디자인 / 졸업 논문

## 핵심 연구 질문 (RQ)

> **BTC 그리드 트레이딩에서 reward 설계가 RL 정책의 알파에 어떤 영향을 미치는가?
> 특히, 어떤 reward 함수 하에서 RL이 ATR 규칙 기반을 초과하며 (혹은 초과하지 못하며), 그 메커니즘은 무엇인가?**

가격 방향을 예측하지 않고, 변동성 자체에서 수익을 추구하는 **동적 그리드 트레이딩** 환경에서,
PPO 에이전트가 ATR 비례 규칙 기반 시스템 대비 보이는 성능 차이를 reward 함수 별로 비교 분석한다.

**자산 범위**: BTC/USDT 1시간봉 **단일 자산** (자산 확장은 본 논문 범위 외).

상세는 [docs/PROJECT_GOAL.md](docs/PROJECT_GOAL.md) 참조.

---

## 핵심 가설 및 사전 증거

| 가설 | 사전 증거 | 정식 검증 |
|---|---|---|
| **H1**: Symmetric reward + ATR 비례 공식에서 RL ≈ ATR | exp020~022 (Test Sharpe 42.0 vs 41.8) | RQ-1 (재현) |
| **H2**: Asymmetric / Prospect-theoretic reward로 RL > ATR | exp027_rl (Test Sharpe 1.955 vs 0.935) | **RQ-2 (메인, exp032)** |
| **H3**: 우수 reward는 "선택적 진입" 행동 | exp027_rl 거래수 214 (ATR 1591의 1/7) | RQ-3 (exp032 행동 분석) |
| **H4**: 알파가 CPCV + Slippage에서도 유지 | — | RQ-4 (exp033~035) |

---

## Phase 진행 상황

```
Phase 1  ██████████ 완료  환경 설계 + RL 학습 + 핵심 발견 (negative finding 확보)
Phase 2  ██████████ 완료  체결 정합성 + ATR 재최적화 + asymmetric reward 사전 증거
Phase 3  ██░░░░░░░░ 진행  Reward 변형 비교 (본 논문 메인) — exp030~035
Phase 4  ░░░░░░░░░░ 대기  논문 작성 + 디펜스 (Phase 3 후반과 병행)
```

상세는 [ROADMAP.md](ROADMAP.md) 참조.

---

## 다음 실험 시리즈

| Exp | 목적 | 논문 챕터 | 상태 |
|---|---|---|---|
| exp030 | PPO 학습 안정화 | §3.3 Method | 대기 |
| exp031 | BC warm-start | §3.4 Method | 대기 |
| exp031b | (조건부) CQL + mixed | §3.4 Method | 조건부 |
| **exp032** | **4가지 reward 비교 (메인)** | **§5 Positive finding** | 대기 |
| exp033 | Slippage + Domain Randomization | §7.1 | 대기 |
| exp034 | CPCV 6-fold + DSR | §5, §7.2 | 대기 |
| exp035 | Test set 봉인 해제 | §7.3 | 대기 |

상세 설계는 [docs/study/rl_finance/project_continuation_plan.md](docs/study/rl_finance/project_continuation_plan.md).

---

## 환경 설정

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements-dev.txt
```

> ⚠️ yfinance 1h 데이터는 최근 730일 제한. 학습 기간 전체는 Binance API(ccxt)로 다운로드.

## 데이터 다운로드 및 전처리

```bash
python scripts/download_data.py
python scripts/preprocess_data.py
```

## 베이스라인 실행

```bash
python scripts/run_baselines.py
```

## 학습 실행

```bash
python scripts/train_ppo.py
```

## 실험 결과 확인 (MLflow)

```bash
mlflow ui   # http://localhost:5000
```

---

## 데이터 분할

| 구분 | 기간 | 용도 |
|------|------|------|
| Train | 2017-08-17 ~ 2022-12-31 | 에이전트 학습 |
| Validation | 2023-01-01 ~ 2023-12-31 | 하이퍼파라미터 튜닝, CPCV |
| Test | 2024-01-01 ~ | 최종 평가 — **exp035까지 봉인** |

---

## 프로젝트 구조

```
capstone-rl-trading/
├── docs/
│   ├── PROJECT_GOAL.md         # ★ 단일 기준점 (RQ, 가설, scope)
│   ├── MDP.md                  # 환경 설계 근거
│   ├── FORMULAS.md             # ATR/RL 공식
│   ├── RELATED_WORK.md         # 선행 연구
│   └── study/
│       ├── rl_finance/         # ★ 학습 노트 24개 (이론 보강)
│       │   ├── 00_overview.md  #   허브
│       │   └── project_continuation_plan.md  # exp030~035 상세
│       ├── ernie chan/         # Quantitative Trading (Chan)
│       ├── lopez de prado/     # Advances in FinML
│       └── *.md                # 팩터, 위험지표 등
├── config/                     # 실험 설정 (exp별 yaml)
├── data/raw/                   # 다운로드 원본 (불변)
├── data/processed/             # 전처리 parquet (.gitignore)
├── src/
│   ├── data/                   # 다운로더, 전처리기
│   ├── env/                    # Gymnasium 트레이딩 환경
│   ├── agents/                 # PPO 래퍼, 베이스라인
│   ├── evaluation/             # 지표, 행동 분석
│   └── utils/                  # 설정 로더, 시각화
├── scripts/                    # CLI 진입점
├── notebooks/                  # 탐색/시각화 전용
├── experiments/                # 실험별 결과 (model + config + log)
├── reports/                    # 학기별 보고서
├── tests/                      # 단위 테스트 (46개)
├── papers/                     # 참고 논문 PDF
├── live_trading/               # (선택) Paper trading 인프라
├── ROADMAP.md                  # Phase별 진행 + 의사결정
├── CLAUDE.md                   # Claude Code 작업 규칙
└── RESEARCH_LOG.md             # 날짜별 실험 기록
```

---

## 주요 문서

| 문서 | 내용 |
|------|------|
| **[docs/PROJECT_GOAL.md](docs/PROJECT_GOAL.md)** | **★ 단일 기준점. RQ, 가설, 평가 방법, scope** |
| [ROADMAP.md](ROADMAP.md) | Phase별 진행 현황 + 의사결정 기록 |
| [docs/study/rl_finance/00_overview.md](docs/study/rl_finance/00_overview.md) | 학습 노트 24개 허브 |
| [docs/study/rl_finance/project_continuation_plan.md](docs/study/rl_finance/project_continuation_plan.md) | exp030~035 상세 설계 |
| [docs/MDP.md](docs/MDP.md) | 환경 설계 근거 (Phase 1~3 변천 포함) |
| [docs/FORMULAS.md](docs/FORMULAS.md) | ATR 고정 / RL / Phase 3 reward 변형 공식 |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | 선행 연구 + Phase 3 인용 매핑 |
| [RESEARCH_LOG.md](RESEARCH_LOG.md) | 날짜별 의사결정 및 실험 결과 |
| [CLAUDE.md](CLAUDE.md) | Claude Code 작업 규칙 |

---

## 이전 README

본 README는 2026-05-14 RQ 재정의 (Pivot 2) 에 맞춰 전면 개정됨.
Phase 1 시점의 RQ ("PPO 동적 그리드가 고정 그리드 대비 Sharpe Ratio 우위?") 와
자산 확장 계획(Phase 4~6 주식/FX/원자재) 은 **본 졸업 논문 범위에서 제외됨**.
이전 RQ의 결과(RL ≈ ATR 발견) 는 본 논문의 §4 Negative finding의 출발점으로 활용.

같은 날 2차 수정으로 RQ를 단정문에서 열린 질문 형태로 다듬음 — 학술 컨벤션 준수.
