Ernie Chan의 _Quantitative Trading: How to Build Your Own Algorithmic Trading Business_ (Wiley, 2009, 1판 기준 약 204페이지)의 전체 목차를 졸업 프로젝트(PPO 기반 BTC 그리드 봇, 보상함수 설계 단계) 관점에서 분류해드릴게요.

> **Edition 참고**: 1판(2009)은 8장 구성에 MATLAB 예제 중심, 2판(2021)은 Python/R 코드 추가 + ML 기법 일부 보강. 페이지 수는 1판 기준으로 표시했고, 2판도 챕터 구성은 거의 동일합니다. 페이지 수는 정확한 절단점이 공식 자료에 분 단위로 안 나와 있어서 **대략적인 분량(±2-3p)** 으로 보시면 됩니다.

## 챕터별 분류표

| 챕터     | 제목                                                 | 페이지            | 난이도 | 분류         | 핵심 이유                                                                                                                                                  |
| ------ | -------------------------------------------------- | -------------- | --- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1**  | The Whats, Whos, and Whys of Quantitative Trading  | ~14p (1-14)    | 하   | **C**      | 퀀트 산업 개론. Chan 본인 경력담 위주. 졸업 프로젝트 기여도 낮음                                                                                                               |
| **2**  | Fishing for Ideas                                  | ~22p (15-36)   | 하   | **B**      | 전략 발굴은 불필요(이미 그리드 확정), 단 §2.2 "[[Pitfalls]]"의 Sharpe/drawdown/거래비용/data-snooping bias 개념은 보상함수 설계에 직결                                                  |
| **3**  | [[Backtesting]]                                    | ~38p (37-74)   | 중   | **A**      | **반드시 학습.** Look-ahead bias, survivorship bias, transaction cost 모델링, Sharpe·drawdown 계산 — RL 환경(env) 정확성과 직결. 일중/일봉 해상도 이슈도 여기서 다룸                    |
| **4**  | Setting Up Your Business                           | ~16p (75-90)   | 하   | **C**      | 브로커 선택, 사업자 등록, 인프라 — 학술 프로젝트와 완전 무관                                                                                                                   |
| **5**  | Execution Systems                                  | ~14p (91-104)  | 중   | **B**      | 슬리피지·체결 지연 모델링. 백테스트 → 실거래 전환 시 필요하지만 졸업 발표 단계에선 후순위. "현실적 백테스트" 디펜스 방어용으로만                                                                            |
| **6**  | [[Money and Risk Management]]                      | ~28p (105-132) | 중   | **A**      | **반드시 학습.** Kelly Formula(§6.1) → `aggressiveness` 행동의 이론적 근거. Risk management(§6.2) → drawdown 패널티 보상함수 설계. holdings_ratio/cash_ratio 상태변수의 의미 부여에 핵심 |
| **7**  | [[Special Topics in Quantitative Trading]]         | ~50p (133-182) | 상   | **A** (선별) | **선별 학습 필수.** 그리드 트레이딩의 이론적 기반이 여기 있음. 단 일부 섹션은 C                                                                                                      |
| ├ §7.1 | Mean-Reverting vs. Momentum                        | ~6p            | 중   | **A**      | 그리드 = mean-reversion 베팅. divergence 상태변수의 정당화                                                                                                          |
| ├ §7.2 | Regime Switching                                   | ~8p            | 상   | **A**      | BTC는 명확한 regime change 자산. volatility 상태변수와 직결                                                                                                         |
| ├ §7.3 | Stationarity and Cointegration                     | ~10p           | 상   | **B**      | log_price 변환 근거. 단일 자산이라 cointegration은 후순위                                                                                                            |
| ├ §7.4 | Factor Models                                      | ~6p            | 상   | **C**      | 멀티 에셋 전략용. 단일 BTC 봇과 거리 있음                                                                                                                             |
| ├ §7.5 | What Is Your Exit Strategy?                        | ~5p            | 중   | **A**      | `profit_target` 행동 설계의 직접적 근거                                                                                                                          |
| ├ §7.6 | Seasonal Trading Strategies                        | ~5p            | 중   | **C**      | 에너지·농산물 위주. BTC와 무관                                                                                                                                    |
| ├ §7.7 | High-Frequency Trading Strategies                  | ~6p            | 상   | **C**      | 1시간봉 사용 중이므로 무관                                                                                                                                        |
| ├ §7.8 | High-Leverage vs. High-Beta Portfolio              | ~4p            | 중   | **C**      | 포트폴리오 구성 문제                                                                                                                                            |
| **8**  | Conclusion: Can Independent Traders Still Succeed? | ~12p (183-194) | 하   | **C**      | 산업 전망 에세이. 학습 가치 낮음                                                                                                                                    |

## 분류 기준 요약

**A (반드시 학습) — 약 90-100페이지** 보상함수 설계와 RL 환경 정확성에 **직접 영향**을 주는 챕터.

- Ch.3 → env가 잘못되면 PPO 학습은 의미 없음 (가장 흔한 졸업 프로젝트 실패 지점)
- Ch.6 → Kelly가 "왜 aggressiveness를 연속 행동으로 두는 게 합리적인가"의 이론적 디펜스
- §7.1, §7.2, §7.5 → 그리드 전략의 시장 가설(mean reversion + regime change)을 정당화. 논문/발표에서 "왜 RL이 필요한가"의 답변 근거

**B (시간 되면) — 약 35-40페이지** 직접적이진 않지만 **디펜스 방어력 강화** 및 향후 확장 시 필요.

- Ch.2 §2.2 (pitfalls), Ch.5, §7.3

**C (후순위) — 약 70-75페이지** 산업 개론, 사업 운영, 멀티 에셋 전략 — 졸업 프로젝트 1차 목표와 거리 있음.

## 추천 학습 순서

1. **Ch.3 (Backtesting)** — env 검증부터 (이전에 작업한 `grid_drift_engine.py`의 일중/일봉 해상도 갭 문제도 여기서 정리됨)
2. **Ch.7.1 + 7.2 + 7.5** — 그리드 + BTC + exit 정당화
3. **Ch.6** — 보상함수 설계 (drawdown 패널티, Kelly 기반 position sizing 보상)
4. (시간 여유 시) Ch.2 §2.2 → Ch.5 → §7.3

---

추가로 알려드릴 점: 이 책은 **RL이나 머신러닝 기반 트레이딩에 대한 직접적 내용은 거의 없습니다** (2판에서 ML 일부 추가됐지만 여전히 통계 모델 중심). 보상함수 설계 자체에 대한 구체적 가이드를 원하신다면 Marcos López de Prado의 _Advances in Financial Machine Learning_ (Ch.5 fractional differentiation, Ch.7 cross-validation, Ch.14-16 backtesting) 또는 Stefan Jansen의 _Machine Learning for Algorithmic Trading_ 2판 (Ch.22 RL 전용)이 더 직접적입니다. Chan의 책은 **시장 가설과 통계적 기반**을 다지는 용도로 쓰시는 게 가장 효율적입니다.