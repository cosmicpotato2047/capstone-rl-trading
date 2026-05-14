# López de Prado, _Advances in Financial Machine Learning_ (Wiley, 2018) 분류표

총 22장, 약 400페이지. 5개 파트로 구성. 졸업 프로젝트(PPO 기반 BTC 그리드 봇, 1시간 바, 현재 보상함수 설계 단계) 관점에서 분류함.

## 분류 기준

- **A (직접 필요)**: 보상함수 설계, 백테스트 신뢰성 확보, 평가 지표 정의에 즉시 사용. 누락 시 졸업 논문 심사에서 약점이 될 수 있음.
- **B (알아두면 좋음)**: 직접 필수는 아니지만 모델 품질, 상태변수 검증, PPO 튜닝, 논문 깊이에 기여. 1학기 여유분 학습.
- **C (후순위)**: 지도학습 중심 / 다자산 / 고빈도 마이크로구조 / HPC 등 현재 설계 범위와 거리가 있음. 졸업 후 또는 본 프로젝트가 RL 단일자산 단일타임프레임을 벗어날 때 의미 있음.

페이지 수는 Wiley 2018 1판 기준 근사값.

## 전체 분류표

### Part 1: Data Analysis (~90p)

| Ch  | 제목                                       | 페이지 | 난이도 | 분류    | 이유                                                                                                                                             |
| --- | ---------------------------------------- | --- | --- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Financial ML as a Distinct Subject       | ~12 | 하   | **B** | 분야 개요. 짧고 쉬워서 1시간이면 읽음. 논문 도입부 작성 시 인용 유용.                                                                                                     |
| 2   | [[Financial Data Structures]]            | ~20 | 중   | **A** | 시간 바 vs 틱/볼륨/달러 바 논의. 현재 yfinance 1시간 바 사용 중인 점이 졸업 심사에서 "왜 정보 기반 바를 안 썼나"로 지적될 수 있음. 최소한 한계와 대응을 논문에 적어둘 근거 필요.                               |
| 3   | Labeling (Triple-Barrier 등)              | ~18 | 중   | **C** | 지도학습용 라벨링. RL은 보상으로 대체되므로 직접 무관. 단, triple-barrier 아이디어는 profit_target 행동 해석에 약간 참고 가능 (B 경계).                                                 |
| 4   | Sample Weights                           | ~14 | 중   | **C** | 지도학습용 샘플 가중치. RL과 직접 관련 없음.                                                                                                                    |
| 5   | [[Fractionally Differentiated Features]] | ~24 | 상   | **A** | **현재 상태변수 `log_price`가 비정상(non-stationary)이라는 본질적 문제를 정면으로 다룸.** PPO 학습 안정성에 직결. log_price 대신 fracdiff 적용을 검토할 가치 있음. 보상함수 설계 다음으로 가장 우선순위 높음. |

### Part 2: Modelling (~64p)

|Ch|제목|페이지|난이도|분류|이유|
|---|---|---|---|---|---|
|6|Ensemble Methods|~14|중|**C**|Bagging/Boosting의 금융 적용. 지도학습 중심이라 PPO와 직접 무관.|
|7|Cross-Validation in Finance|~14|중|**B**|Purged K-Fold, embargo 개념. RL 백테스트에서 train/test 분할 시 시간 누설 방지에 응용 가능. 논문 평가 구간 설계에 도움.|
|8|Feature Importance|~22|중|**B**|현재 5개 상태변수(log_price, divergence, holdings_ratio, cash_ratio, volatility)의 기여도 사후 분석에 유용. SHAP/MDA 기법으로 "왜 이 5개인가"에 대한 정량적 근거 제시 가능 — 심사용으로 강력.|
|9|Hyper-Parameter Tuning with CV|~14|중|**B**|PPO 하이퍼파라미터(lr, clip ratio, GAE λ 등) 튜닝 절차에 일부 응용. RL 전용은 아니지만 시간 누설 회피 관점은 유효.|

### Part 3: Backtesting (~106p) — **이 파트가 졸업 프로젝트 핵심**

| Ch  | 제목                                          | 페이지 | 난이도 | 분류    | 이유                                                                                                                |
| --- | ------------------------------------------- | --- | --- | ----- | ----------------------------------------------------------------------------------------------------------------- |
| 10  | Bet Sizing                                  | ~14 | 중   | **B** | 베팅 사이즈 결정론. 현재 행동 공간 `aggressiveness`의 해석/검증 프레임워크로 유용. 행동 설계가 이미 끝났다면 우선순위 낮음.                                   |
| 11  | [[The Dangers of Backtesting]]              | ~12 | 중   | **A** | 백테스트 과적합, 다중검증 편향. RL은 환경 조정만으로도 cherry-pick이 쉬워 더 위험함. 졸업 논문 신뢰성 방어에 필수. 짧고 핵심적.                                 |
| 12  | Backtesting through Cross-Validation (CPCV) | ~14 | 중   | **B** | Combinatorial Purged CV. RL 백테스트에 그대로 쓰긴 어렵지만 "단일 walk-forward의 한계"를 인용할 때 유용.                                    |
| 13  | Backtesting on Synthetic Data               | ~14 | 상   | **B** | Monte Carlo / 합성 시계열로 백테스트. RL의 simulator robustness 검증에 직접 응용 가능. 시간 되면 강력 추천 (B 상위).                            |
| 14  | [[Backtest Statistics]]                     | ~16 | 중   | **A** | Sharpe, Probabilistic Sharpe, Deflated Sharpe, DSR. **보상함수 설계 단계에서 즉시 사용.** "왜 단순 수익률 보상이 아닌가"의 답이 여기 있음. 1순위 학습. |
| 15  | [[Understanding Strategy Risk]]             | ~14 | 중   | **A** | 드로다운, 손실 분포, 리스크 조정 수익. **현재 단계(보상함수 설계)와 가장 직결.** 14장과 묶어서 먼저 읽을 것.                                              |
| 16  | ML Asset Allocation (HRP)                   | ~16 | 중   | **C** | 다자산 포트폴리오 배분. 단일자산 BTC 봇에는 무관.                                                                                    |

### Part 4: Useful Financial Features (~60p)

|Ch|제목|페이지|난이도|분류|이유|
|---|---|---|---|---|---|
|17|Structural Breaks (CUSUM, SADF 등)|~16|상|**B**|레짐 변화 감지. 6번째 상태변수 후보 또는 평가 구간 분할 기준. BTC는 레짐 변화가 강해서 가치 있음.|
|18|Entropy Features|~16|상|**B**|정보 엔트로피 기반 특성. `divergence`와 `volatility`를 보강하는 추가 상태변수 후보. 이미 5개로 확정했다면 후순위.|
|19|Microstructural Features (VPIN 등)|~28|상|**C**|호가/체결 단위 분석. 1시간 바 환경에서는 적용 불가. tick 데이터로 확장 시 의미.|

### Part 5: High-Performance Computing Recipes (~62p)

|Ch|제목|페이지|난이도|분류|이유|
|---|---|---|---|---|---|
|20|Multiprocessing and Vectorization|~20|중|**C**|엔지니어링 기법. PPO 병렬 환경(`SubprocVecEnv`)은 stable-baselines3에서 이미 제공되므로 직접 학습 불요.|
|21|Brute Force and Quantum Computers|~14|상|**C**|Niche. 무관.|
|22|HPC Computational Intelligence and Forecasting|~28|상|**C**|HPC 워크플로우. 졸업 프로젝트 범위 밖.|

## 요약 통계

- **A 등급**: 5개 챕터 (Ch 2, 5, 11, 14, 15) — **약 86페이지**
- **B 등급**: 9개 챕터 (Ch 1, 7, 8, 9, 10, 12, 13, 17, 18) — 약 138페이지
- **C 등급**: 8개 챕터 — 약 176페이지

A 등급만 집중하면 100페이지 미만이라 1주~2주 내 학습 가능.

## 권장 학습 순서 (현재 단계 기준)

**1단계 — 보상함수 설계 직결 (이번 주):** Ch 15 → Ch 14 → Ch 11 순서. 위험 정의 → 평가 통계 → 백테스트 함정 흐름. 30~40페이지.

**2단계 — 상태변수 재검토 (다음 주):** Ch 5 (fractional differentiation). `log_price` 비정상성 문제는 PPO 수렴에 영향이 크므로 보상함수 다음으로 우선. 단독 24페이지지만 수식 밀도 높음.

**3단계 — 데이터 구조 점검 (여유 있을 때):** Ch 2. 1시간 바의 한계를 논문에 명시할 근거 확보용. 20페이지.

**4단계 — B 등급 중 우선순위:** Ch 8 (상태변수 사후 검증) → Ch 13 (합성 데이터 robustness) → Ch 7 (CV 시간 누설). 심사 방어력 강화용.

## 한 가지 짚어둘 점

이 책은 **지도학습 중심**으로 쓰여서 RL 프로젝트에 직접 매핑되는 부분이 생각보다 적음. A 등급 5개 챕터 외에는 "개념을 빌려와 RL에 응용"하는 방식이 됨. RL 전용 보완 자료로 Sutton & Barto 2판의 13장(Policy Gradient)과 Schulman의 PPO 논문은 별도로 챙기는 게 좋음. López de Prado는 **금융 도메인 지식**을, RL 자료는 **알고리즘**을 담당하는 분업 구도로 가는 걸 추천.