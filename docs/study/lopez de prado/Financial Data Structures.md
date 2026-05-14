# Chapter 2: Financial Data Structures — 세부 구성

## 구성

2.1 Motivation (p.23), 2.2 Essential Types of Financial Data (p.23) — 2.2.1 Fundamental Data, 2.2.2 Market Data, 2.2.3 Analytics, 2.2.4 Alternative Data, 2.3 Bars — 2.3.1 Standard Bars, 2.3.2 Information-Driven Bars, 2.4 Dealing with Multi-Product Series — 2.4.1 The ETF Trick, 2.4.2 PCA Weights, 2.4.3 Single Future Roll, 2.5 Sampling Features — 2.5.1 Sampling for Reduction, 2.5.2 Event-Based Sampling. 약 20페이지. 난이도 중.

## 챕터의 핵심 메시지

이 챕터는 **"입력 데이터를 어떻게 만들 것인가"**에 대한 책 전체의 출발점. 핵심 주장이 도발적임:

> 대부분의 금융 ML 프로젝트가 **시간 봉(time bar)**을 쓰지만, 이건 ML 관점에서 거의 최악의 선택이다.

5장이 "log_price를 어떻게 정상화할 것인가"였다면, 2장은 그 한 단계 더 앞 — **"애초에 가격을 어떤 단위로 샘플링할 것인가"**. 사용자 프로젝트는 yfinance로 BTC 1시간 봉을 사용 중이므로, 이 챕터는 **현재 데이터 파이프라인의 가장 근본적 가정을 흔드는 챕터**.

## 절별 내용

### 2.2 Essential Types of Financial Data — 4가지 데이터 유형

|유형|설명|BTC 그리드 봇 관련성|
|---|---|---|
|**Fundamental**|재무제표, 매출, 부채 등 회계 데이터|✗ 암호화폐엔 거의 무관|
|**Market**|OHLCV, 호가, 체결 데이터|✓ **사용자 데이터의 본진**|
|**Analytics**|가공된 신호 (애널리스트 추정치, 뉴스 점수 등)|△ 온체인 지표(BTC 고유)로 응용 가능|
|**Alternative**|뉴스, 소셜, 위성사진 등 비전통 데이터|△ 트위터 sentiment 등 확장 가능|

**사용자에게 중요한 점**: 암호화폐는 fundamental이 빈약한 대신 **on-chain analytics**가 풍부. 미실현 자본이득(MVRV), 해시레이트, 거래소 유출입 등. 이게 졸업 논문에서 BTC 봇의 차별화 포인트가 될 수 있음 — 현재 5개 상태변수가 모두 가격에서 파생된 것이라면, on-chain 변수 1~2개 추가가 유의미할 수 있음.

### 2.3 Bars — **이 챕터의 핵심, 그리고 책 전체에서 가장 자주 인용되는 절**

#### 2.3.1 Standard Bars — 4가지 봉 형태

|봉 종류|샘플링 기준|사용자 현재 상태|
|---|---|---|
|**Time bars**|정해진 시간 간격마다 (1분, 1시간, 1일)|✓ **현재 사용 중 (1h)**|
|Tick bars|매 N개 거래마다|✗|
|Volume bars|매 N개 코인 거래량마다|✗|
|**Dollar bars**|매 $N 거래액마다|**저자 추천**|

**López de Prado가 time bars를 비판하는 이유 (이 책에서 가장 유명한 주장 중 하나)**:

1. **시장 활동 강도와 무관하게 샘플링** — 활동이 적은 새벽 시간과 폭등하는 뉴스 직후가 같은 1개 봉으로 처리됨. 정보량 불균형.
2. **수익률 분포의 통계적 성질이 나쁨** — 시간 봉 수익률은 fat-tail이 심하고 비정규. 분산이 시간에 따라 크게 변동.
3. **자기상관 강함** — i.i.d. 가정을 깬다.

**Dollar bars의 장점**:

- 매 봉마다 "같은 양의 경제적 활동"을 담음
- 수익률 분포가 정규에 가까워짐 (i.i.d.에 근접)
- 가격 분할 / 통화 절하에 강함

**암호화폐 맥락에서 추가 고려사항**:

- BTC는 24/7 거래라 주식의 "주말 갭" 문제는 없음 → time bars의 단점이 _주식보다는_ 덜 심함
- 하지만 아시아/유럽/미국 세션 차이, 뉴스 이벤트(예: ETF 승인, FOMC) 직후 폭발적 거래량 → time bars는 여전히 정보량 불균형
- BTC의 변동성 클러스터링은 dollar bars에서 훨씬 잘 정규화됨

#### 2.3.2 Information-Driven Bars — 고급

Tick Imbalance Bars, Volume Imbalance Bars, Dollar Imbalance Bars, Runs Bars 등. **정보가 비대칭적으로 도착할 때(매수세 vs 매도세 불균형) 더 많이 샘플링**하는 적응형 봉. 졸업 프로젝트에서는 **읽고 넘어가도 무방**. 구현 비용 대비 한계 효용이 낮음.

### 2.4 Dealing with Multi-Product Series

ETF Trick, PCA Weights, Single Future Roll. **단일 자산(BTC) 그리드 봇이므로 무관.** 스킵 가능. 졸업 후 멀티 코인으로 확장할 때 참고.

### 2.5 Sampling Features

#### 2.5.1 Sampling for Reduction

단순 다운샘플링. 정보 손실 문제 지적.

#### 2.5.2 Event-Based Sampling — **CUSUM 필터**

"중요한 사건"이 발생한 시점만 샘플링. CUSUM 통계량이 임계값을 넘는 순간만 학습 데이터로 사용.

**그리드 봇 응용**: 학습 효율 측면에서 흥미로움. 그리드 봇은 횡보장에선 별로 학습할 게 없고, **레짐 전환 시점에 결정적 학습이 일어남**. CUSUM 기반 이벤트 샘플링으로 **학습 중요 구간을 가중**할 수 있음. 단, PPO의 episode 구조와 어떻게 결합할지는 비자명 — 졸업 논문 후속 연구 아이디어로 메모.

이 절은 17장(Structural Breaks)과 직접 연결됨.

## 사용자 PPO 그리드 봇에 직접 적용

### 현실 점검

|항목|이상 (López de Prado)|현실 (졸업 프로젝트)|
|---|---|---|
|데이터 소스|거래소 tick 데이터|yfinance OHLCV|
|봉 형태|Dollar bars|**Time bars (1h)**|
|샘플링|Event-based|균등 시간|

**유의사항**: yfinance는 시간 봉만 제공. Dollar bars를 만들려면 Binance API의 raw trade 데이터(또는 Kaiko, CryptoDataDownload 등)가 필요.

### 졸업 프로젝트 관점에서 3가지 선택지

**A. 그대로 가기 (가장 현실적)** 1시간 시간 봉 유지. 단, 졸업 논문 limitation 절에 명시:

> "본 연구는 yfinance에서 제공하는 1시간 시간 봉을 사용한다. López de Prado(2018, Ch.2)는 시간 봉이 정보 도착 강도와 무관하게 균등 샘플링되어 ML 입력으로 비최적임을 지적했다. Dollar bars 또는 volume bars로의 전환은 본 연구의 범위를 벗어나며 후속 연구로 남긴다."

이 한 문단으로 심사 방어 완료. 가장 깔끔.

**B. Dollar bars로 전환 (의욕적, 데이터 작업량 큼)** Binance의 free aggregated trades API로 BTC tick 데이터 수집(2~3년치도 수 GB) → dollar bars 생성. 작업량 1~2주. **얻는 것**:

- 졸업 논문 contribution이 한 단계 강화 ("BTC 그리드 봇에 dollar bar 입력 적용")
- 학습 안정성 개선 가능성 (수익률 분포가 정규에 가까워져 PPO critic 안정화)

**얻기 위해 감수할 것**:

- 데이터 엔지니어링 시간 소모
- 그리드 봇 실행 로직과의 결합 재설계 (그리드는 시간이 아니라 가격 기반)

**C. Ablation으로 비교 (최선)** 시간 봉과 dollar 봉 두 가지로 같은 PPO 모델을 학습 → 결과 비교. 논문에 표 한 장으로 들어감.

> Table X: PPO BTC 그리드 봇 성능 비교
> 
> - 1h Time bars: Sharpe 1.2, DD 18%
> - $5M Dollar bars: Sharpe X.X, DD Y%

5장 fracdiff ablation과 합쳐 **데이터/특성 설계 절에 2개의 ablation**이 들어가면 논문 깊이가 크게 상승. 졸업 심사 관점에서 ROI 최고.

### 그리드 봇 특수성: 봉과 그리드 실행의 미스매치

이 챕터에서 한 가지 더 짚을 점. 그리드 트레이딩은 **가격 격자**에서 실행됨 (예: 1% 간격 그리드). 그런데 학습 데이터는 **시간 봉**으로 들어옴. 이 미스매치가 미묘한 문제를 만듦:

- 1시간 안에 그리드 셀 여러 개를 통과해도 모델은 1개 관측치만 봄
- 1시간 안에 가격이 그리드를 안 건드려도 학습 스텝은 1개 발생 (낭비)

**Dollar bars 또는 가격 변화 기반 샘플링(±X% 이동마다 1봉)**이 그리드 봇의 실행 단위와 더 자연스럽게 맞음. 이걸 졸업 논문 "데이터 설계 동기" 절에 적으면 단순히 "López de Prado가 추천해서"보다 훨씬 설득력 있는 정당화가 됨.

## 11/14/15/5/2장 — 역할 정리

|챕터|영역|사용자 작업|
|---|---|---|
|**2 Data Structures**|**데이터 표현 (가장 앞단)**|봉 형태 검토 / limitation 명시|
|5 Fractional Diff|상태변수 정상성|log_price 교체|
|11 Dangers|방법론 정당성|train/test 규율|
|14 Backtest Stats|보상 signal 후보|Sharpe/PSR/DSR|
|15 Strategy Risk|reward shaping|Implied Sharpe, PSF|

2장과 5장은 **데이터 파이프라인의 두 단계**: 2장이 "어떻게 봉을 만들 것인가", 5장이 "그 봉에서 어떤 변환을 거칠 것인가". 둘 다 PPO 학습 안정성에 영향을 주는 입력 품질 이슈.

## 권장 학습/적용 순서

1. **이번 주에는 2.3 (Bars)만 정독.** 30분. 시간 봉의 한계를 정확히 이해하는 게 목적.
2. **2.4는 스킵, 2.5는 훑기.** 단일 BTC라 4번은 불필요. 5번은 17장과 같이 보면 더 의미.
3. **결정 포인트**: 위 A/B/C 중 선택. 시간 여유와 의욕에 따라 결정. **권장은 A 또는 C.** B(완전 전환)는 졸업 프로젝트 데드라인 리스크가 큼.
4. **선택과 무관하게 limitation 절은 명시.** 시간 봉을 쓰든 dollar bars를 쓰든, 그 선택의 이론적 근거를 1문단 적어두는 게 모든 케이스에서 가산점.

## 한 가지 짚어둘 점

López de Prado가 _Advances_에서 가장 강하게 주장하는 두 가지가 **Dollar Bars(2장)와 Triple-Barrier Labeling(3장)**인데, **이 둘은 지도학습 패러다임에서 도출된 것**. RL 프레임에서는 라벨이 없으므로 3장은 무관해지지만, **2장은 RL에도 그대로 유효** — 데이터 표현은 학습 패러다임과 독립적. 따라서 2장의 메시지를 RL 그리드 봇에 적용하는 것은 자연스러움. 사용자 프로젝트에서 5장보다 2장이 **개념적으로는 더 근본적이지만**, 데이터 엔지니어링 비용 때문에 실제 적용 우선순위는 5장이 더 높음 — 이게 앞서 분류표에서 5장을 1순위, 2장을 그 다음으로 둔 이유.