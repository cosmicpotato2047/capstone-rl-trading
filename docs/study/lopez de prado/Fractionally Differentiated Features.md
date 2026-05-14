# Chapter 5: Fractionally Differentiated Features — 세부 구성

## 구성

5.1 Motivation (p.75), 5.2 The Stationarity vs. Memory Dilemma (p.75), 5.3 Literature Review (p.76), 5.4 The Method (p.77) — 5.4.1 Long Memory, 5.4.2 Iterative Estimation, 5.4.3 Convergence (p.80), 5.5 Implementation (p.80) — 5.5.1 Expanding Window, 5.5.2 Fixed-Width Window Fracdiff (p.82), 5.6 Stationarity with Maximum Memory Preservation, 5.7 Conclusion (p.88). 총 약 14페이지. 수식 밀도 높음, 난이도 상.

## 챕터의 핵심 메시지

11/14/15장이 "평가와 위험"이었다면 **5장은 상태변수 설계의 근본 문제**를 다룸. 사용자 프로젝트와의 연결점:

> 현재 상태변수 `log_price`는 **비정상(non-stationary)** 시계열. PPO 신경망 입력으로 부적절. 그렇다고 차분(returns)해버리면 **시계열의 메모리가 완전히 사라짐**. 어떻게 둘 다 잡을 것인가?

이게 챕터 전체를 관통하는 단 하나의 질문이고, 답이 **fractional differentiation** (분수 차분).

## 절별 내용

### 5.1 + 5.2 The Stationarity vs. Memory Dilemma — **챕터의 골격**

금융 시계열은 차익거래의 영향으로 신호 대 잡음비(SNR)가 낮음. 표준적인 정상화 변환(정수 차분)은 메모리를 제거함으로써 이 신호를 더 약화시킴. 가격 시계열은 메모리를 가짐 — 모든 값이 이전 가격들의 긴 역사에 의존하기 때문. 반면 정수 차분된 시계열(예: 수익률)은 메모리 컷오프가 있어 유한 표본 윈도우 이후의 역사는 완전히 무시됨.

핵심 표:

|변환|정상성|메모리 보존|
|---|---|---|
|원본 가격 (d=0)|✗ 비정상|✓ 완전|
|log 가격 (d=0)|✗ 비정상 (단위근)|✓ 완전|
|수익률 = log차분 (d=1)|✓ 정상|✗ 1-lag 후 완전 망각|
|**fracdiff (0 < d < 1)**|✓ 정상|**✓ 부분 보존**|

ML 모델은 정상 입력을 가정. 비정상 입력을 넣으면 분포 변화로 학습 불안정. PPO도 예외 아님 — actor/critic 네트워크의 layer normalization이 비정상 입력을 못 따라감.

### 5.3 Literature Review

Hosking(1981)이 이 문제를 처음 다룬 학자로 등장. ARFIMA 계열 통계학 흐름 — 시계열 학자들에겐 익숙하지만 ML 진영엔 잘 안 알려진 도구. **읽기는 빠르게.**

### 5.4 The Method — 수학적 핵심

#### 5.4.1 Long Memory

이항급수 전개로 차분 연산자를 분수 차수로 일반화:

$$(1-B)^d = \sum_{k=0}^{\infty} \binom{d}{k}(-B)^k$$

여기서 B는 backshift 연산자. 가중치는 재귀적으로:

$$w_k = -w_{k-1} \cdot \frac{d - k + 1}{k}, \quad w_0 = 1$$

**중요한 직관**: d=1이면 w = [1, -1, 0, 0, ...] → 단순 차분. d가 1보다 작으면 가중치가 **느리게 0으로 감쇠** → 먼 과거가 약하게나마 살아남음 = long memory.

예시: d=0.4일 때 가중치 처음 몇 개 ≈ [1, -0.4, -0.12, -0.064, -0.04, ...] — 100 lag 뒤에도 가중치가 0이 아님.

#### 5.4.2 Iterative Estimation + 5.4.3 Convergence

가중치를 재귀로 계산하는 방법과 수렴 조건. 구현 시 직접 신경 쓸 부분은 아님 (라이브러리가 처리). **읽고 넘어가도 됨.**

### 5.5 Implementation — **실전에서 가장 중요한 절**

#### 5.5.1 Expanding Window

데이터 시작점부터 모든 과거를 누적해 가중치 적용. 문제점:

- 시간이 갈수록 윈도우가 커져 계산 비대칭
- 초기 관측치들은 적은 lag만 사용 → 분포가 시간에 따라 변함 (정상성 깨짐)

→ 실용성 낮음.

#### 5.5.2 Fixed-Width Window Fracdiff (FFD) — **실제로 쓸 방법**

가중치가 일정 임계값(thresh) 이하로 떨어지면 잘라내, 모든 시점에서 동일한 길이의 윈도우로 차분을 계산. 이렇게 하면:

- 모든 시점이 같은 분포에서 추출됨 (진짜 정상)
- 계산량 일정
- ML 입력으로 안정적

**이게 사용자 PPO에 들어가야 할 형태.**

### 5.6 Stationarity with Maximum Memory Preservation — **방법론적 핵심**

알고리즘:

1. d를 0부터 1까지 0.05 단위로 grid search
2. 각 d에 대해 fracdiff 시계열을 만들고 **ADF 검정** (Augmented Dickey-Fuller) 통과 여부 확인
3. ADF 통과하는 **최소 d**를 선택 (= 정상성 확보하는 가장 약한 차분)
4. 그 d에서 원본 시계열과의 상관계수를 보고 → 보통 0.95+ 유지 가능

López de Prado가 다양한 선물 자산에 적용한 결과 표가 책에 나옴. 대부분 d* ≈ 0.3 ~ 0.5 영역. **BTC도 비슷한 범위가 나올 가능성 큼** (BTC가 워낙 모멘텀이 강해 d* < 0.5 예상).

### 5.7 Conclusion

요약: 금융 ML이 수익률을 디폴트 입력으로 쓰는 관행은 **메모리를 통째로 버리는 손해**. fracdiff가 표준이 되어야 한다는 저자의 주장.

## 사용자 PPO 그리드 봇에 직접 적용하기

### 현재 상태변수 점검

|상태변수|정상성|fracdiff 필요?|
|---|---|---|
|**log_price**|**✗ 비정상**|**✓ 핵심 적용 대상**|
|divergence (price − MA 가정)|✓ 이미 차분 형태|✗ 불필요|
|holdings_ratio|✓ [0,1] bounded|✗ 불필요|
|cash_ratio|✓ [0,1] bounded|✗ 불필요|
|volatility (rolling std)|✓ 정상|✗ 불필요|

**fracdiff는 `log_price` 하나에만 적용.** 5개 변수 중 1개만 손대면 됨.

### 적용 레시피

```python
# 1. mlfinlab 또는 직접 구현 (FFD 알고리즘 약 30줄)
from mlfinlab.features.fracdiff import frac_diff_ffd
from statsmodels.tsa.stattools import adfuller

# 2. d 탐색
log_price = np.log(btc_close)
results = []
for d in np.arange(0.0, 1.05, 0.05):
    fd_series = frac_diff_ffd(log_price, d, thresh=1e-4).dropna()
    adf_stat, p_value = adfuller(fd_series)[:2]
    corr = fd_series.corr(log_price.loc[fd_series.index])
    results.append((d, adf_stat, p_value, corr))

# 3. p_value < 0.05 인 최소 d 선택
# 4. 그 d로 fracdiff한 시계열을 log_price 자리에 투입
```

BTC 1시간 봉 예상 결과: d* ≈ 0.35 ~ 0.45 영역. ADF p-value < 0.05이면서 원본과 상관 0.95 이상. **즉 거의 모든 가격 정보를 유지하면서 정상화 달성.**

### PPO 학습에 미치는 영향 (예상)

1. **학습 안정성**: critic의 value function이 비정상 입력에서 발산 경향. fracdiff 적용 후 수렴 속도 개선 기대.
2. **정규화 효율**: layer norm / observation normalization이 정상 분포에서 의도대로 작동.
3. **일반화**: train 구간 가격 레벨에 의존하는 정책이 test 구간에서 무너지는 문제 완화. **2022년 train → 2025년 test 같은 극단적 분포 차이에서 특히 효과적.**

### 졸업 논문 ablation 후보

**A 안**: `log_price` (원본 비정상) **B 안**: `returns` (1-lag 차분, 메모리 손실) **C 안**: `fracdiff(log_price, d*)` (제안)

세 가지 PPO 학습 결과를 같은 시드 풀에서 비교 → 논문에 표 한 장으로 들어감. **이게 논문의 contribution을 강화하는 가장 저렴한 실험.** 이론적 근거(López de Prado)와 실험적 근거가 모두 있는 ablation은 심사위원에게 인상적임.

## 11/14/15장 / 5장 — 역할 정리

|챕터|기여 영역|사용자 직접 적용|
|---|---|---|
|5 Fractional Diff|**상태변수 설계**|log_price 교체|
|11 Dangers|**방법론 정당성**|train/test 규율|
|14 Backtest Stats|**보상 signal 후보**|Sharpe/PSR/DSR|
|15 Strategy Risk|**보상 reward shaping**|Implied Sharpe, PSF|

**5장만 유일하게 "보상함수 외부"의 영역.** 나머지 셋이 reward 설계와 평가의 삼각형이라면, 5장은 **그 reward를 학습하는 PPO의 입력 품질**을 다룸. 둘은 독립적으로 동시에 진행 가능 — 이번 주에 보상함수(14, 15) 설계하면서, 데이터 전처리 파이프라인 한쪽에서 fracdiff 모듈을 따로 구현해두면 좋음.

## 한 가지 짚어둘 점

5장에는 함정이 하나 있음: __d_ 탐색을 train 데이터에서만 해야 함._* 전체 데이터(2017~2026)로 d*를 찾으면 test 구간 정보가 누설됨 — 11장이 경고한 look-ahead bias의 미묘한 형태. 졸업 논문에 다음과 같이 명시해야 함:

> "d_는 train 구간(2017~2022) BTC 1시간 봉 log-price에 대해 ADF 검정 기반 grid search로 결정되었으며 (d_ = X.XX), validation/test 구간 fracdiff 계산에도 동일한 d* 와 동일한 가중치 벡터를 사용했다."

이렇게 쓰면 5장 적용 + 11장 방어가 한 번에 해결됨. 두 챕터가 시너지를 내는 지점.