# 드릴 세션 순서 (15 세션)

> 발표 직전까지 매일 한 세션씩 (30–45분). 쉬운 것 → 어려운 것 순서.
> ★ = 가장 많이 반복할 세션.

## 세션 카탈로그

| # | 세션 | 핵심 학습목표 | 예상 시간 | 우선순위 |
|---|---|---|---|---|
| 1 | Big Picture 다지기 | 30초·5분·30분 답변 | 30분 | ★★★ |
| 2 | 사전 개념 (RL/PPO/그리드/ATR) | 비전문가 비유로 설명 | 30분 | ★★ |
| 3 | 통계 개념 (Sharpe/Cohen's d/CPCV) | 두 DSR 헷갈리지 않기 | 30분 | ★★ |
| 4 | 4가지 Reward 함수 | 수식·직관·하이퍼파라미터 | 45분 | ★★★ |
| 5 | MDP 환경 설계 | State·Action·격자 공식 | 30분 | ★★ |
| 6 | Phase 1 negative + pivot | RL=Fixed 메커니즘 + pivot 정당화 | 45분 | ★★★ |
| 7 | Phase 2 환경 정상화 | favorable bias + 안정화 4종 | 30분 | ★★ |
| 8 | exp032b 시나리오 D | 사전 시나리오 + 두 클러스터 | 45분 | ★★★ |
| 9 | exp032c 메커니즘 | DSR sliding window 메커니즘 | 45분 | ★★★ |
| 10 | exp033 slippage + exp034 CPCV | 1차 reversal + cluster 보존 | 30분 | ★★ |
| 11 | **exp035 + H5** (본 논문 main) | 2차 reversal + PT OOS 메커니즘 | 60분 | ★★★★ |
| 12 | 결론 frame + 한계 정직 인정 | B&H 정직 인정 대응 | 30분 | ★★★ |
| 13 | 선행 연구 좌표 | 9개 인용 논문 + 직접 선행 4편 | 30분 | ★ |
| 14 | 시각자료 발표 시연 | figure-by-figure 청중 설명 | 45분 | ★★ |
| 15 | **Mock 발표 + 곡구 질문** | end-to-end + 적대적 질문 | 60분 | ★★★★ |

총 ~9시간. 발표 전 2주에 걸쳐 진행하면 적당.

---

## 권장 일정 (발표 2주 전부터)

| Day | 세션 | 비고 |
|---|---|---|
| D-14 | 세션 1, 2 | warm-up |
| D-13 | 세션 3 | 통계 정리 |
| D-12 | 세션 4 | reward 함수 (오래 걸림) |
| D-11 | 세션 5 | MDP |
| D-10 | 세션 6 ★ | pivot 정당화 — 발표의 narrative 핵심 |
| D-9 | 세션 7 | 환경 정상화 |
| D-8 | 세션 8 ★ | exp032b — Pareto frontier |
| D-7 | 세션 9 ★ | 메커니즘 (DSR sliding window 가장 어려움) |
| D-6 | 세션 10 | slippage + CPCV |
| D-5 | 세션 11 ★★ | H5 본 논문 main — 가장 중요 |
| D-4 | 세션 11 다시 (반복) | 같은 주제 다른 각도로 |
| D-3 | 세션 12, 13 | 결론·한계·선행연구 |
| D-2 | 세션 14 | 그림 설명 시연 |
| D-1 | 세션 15 ★★ | end-to-end mock |
| D-0 | 발표 | — |

세션 6, 8, 9, 11은 발표 narrative의 핵심이므로 **여러 번 반복**.

---

## Web Claude에 던질 트리거 문장 (세션별)

### 세션 1 — Big Picture
> "오늘은 이 연구를 **30초, 5분, 30분 세 가지 길이**로 설명하는 연습이야.
> 내가 먼저 30초 버전 답할 테니, 부족한 점 + 추가 단어 + 청중에게 더 잘 통할 비유를
> 피드백해줘. 그 다음 5분, 30분 순서로 진행."

### 세션 2 — 사전 개념
> "오늘은 청중이 **RL/금융 비전문가 (가족, 친구)**라 가정하고, 다음 개념들을
> 1분 안에 비유로 설명하는 드릴이야:
>   강화학습 / PPO / 그리드 트레이딩 / ATR / Sharpe ratio / MDD
> 하나씩 차례로 던져줘. 내 답이 jargon-heavy하면 즉시 잘라줘."

### 세션 3 — 통계 개념
> "오늘은 통계·평가 지표. 특히 청중이 헷갈릴 만한 부분 점검:
>   CPCV(2018 ch.8) vs Deflated Sharpe(2014) — 둘 다 López de Prado, 다른 개념
>   Cohen's d 해석 (|d|<0.3 vs |d|>0.8)
>   IQM과 5% CVaR이 평균보다 좋은 이유
> 차례로 던져줘."

### 세션 4 — 4가지 Reward 함수
> "오늘은 4가지 reward (sym/asym/dsr/pt)에 깊이 들어가. 한 변형씩:
>   (1) 수식이 뭔지 + 직관
>   (2) 왜 그 변형을 선택했는지 (이론적 계보)
>   (3) Optuna best 값과 인간 표준값의 차이 (특히 pt!)
>   (4) 함수 모양 (sym은 직선, pt는 곡선 등)
> dsr 차례에는 **'왜 본질적으로 다른가'**를 깊이 물어줘 — sliding window memory."

### 세션 5 — MDP 환경 설계
> "오늘은 환경 설계. 다음을 차례로:
>   State 7차원 — 각 변수와 왜 그걸 골랐는지
>   Action 2차원 — 왜 연속, 왜 이산 안 쓰는지
>   격자 공식 4개 호가 — 어떻게 도출되는지
>   사이클 / fee / 봉인 원칙
> 특히 'Sell 우선 원칙' 이유를 물어줘."

### 세션 6 — Phase 1 negative + pivot ★
> "오늘은 **이 연구의 narrative 핵심**. Phase 1 negative finding이 발표 첫 5분의
> hook이야. 차례로:
>   (1) exp020/021/022 각각 무엇을 했고 무엇이 나왔는지
>   (2) **'RL = Fixed [1.0, 0.0]'의 의미** — 왜 그게 결정적인가
>   (3) **왜 그렇게 됐는가의 메커니즘** (ATR 흡수 + sym reward 최적해)
>   (4) **왜 pivot 했고 무엇으로** — 본 논문 RQ 정식화
>   (5) 왜 본 논문에서 Phase 1을 정직 인용했는가 + Env-v2 caveat
> 마지막에 '청중이 *왜 Phase 1을 굳이 보고하나*'라고 물으면 어떻게 답할지 시뮬레이션."

### 세션 7 — Phase 2 환경 정상화
> "오늘은 Phase 2. 차례로:
>   favorable bias 정의 + 왜 제거
>   학습 안정화 4종 (LR decay / entropy anneal / target_kl / best ckpt)
>   왜 best checkpoint (final 1.21 vs best 1.97)
>   Optuna TPE 작동 방식
>   Env 버전 (v2 → v3 → v4) 차이"

### 세션 8 — exp032b 시나리오 D ★
> "오늘은 메인 실험 exp032b. 차례로:
>   사전 등록 시나리오 A/B/C — **왜 사전 등록하는가** (사후 합리화 방지)
>   결과가 A/B/C에 안 맞아서 사후 시나리오 D 정의 — **이게 학술적으로 정당한가?**
>   두 클러스터의 통계적 분리 (within d<0.3, across d>0.79)
>   policy distance 2.22× 의미
>   H1~H3 verdict
> 특히 '시나리오 D 사후 정의가 사후 합리화 아닌가?'라는 적대 질문 대비."

### 세션 9 — exp032c 메커니즘 ★
> "오늘은 가장 어려운 부분. 차례로:
>   인과 사슬 (Reward → 행동 → Risk → Outcome)
>   거래빈도 차이의 원인 (asymmetric reward → 매수 호가 멀리)
>   **Hold rate 차이의 원인** — DSR sliding window memory가 정확히 어떻게 긴 hold를 학습시키는가
>   Counterfactual state-grid (같은 state에서 다른 action)
> DSR sliding window 메커니즘 답을 적어도 3번 다른 각도로 물어줘. 가장 잘 막힐 부분."

### 세션 10 — exp033 + exp034
> "오늘은 강건성. 차례로:
>   exp033 slippage 0.02% — 4 variant 일률 ~12% 감쇠
>   ATR baseline 공정 비교 (Phase 16a) — ATR-slip 0.835 baseline 사용
>   exp034 CPCV 6-fold 15 paths 설정 + purge ±168h
>   1차 winner reversal (Val sym → CPCV dsr) — 이게 왜 중요한가
>   Cluster preservation 비율 (2.22 → 2.19 → 3.55)"

### 세션 11 — **exp035 + H5 (본 논문 main)** ★★
> "오늘은 본 논문의 가장 중요한 chapter. 이 세션은 60분 이상.
>   (1) Test 봉인 이유와 봉인 해제까지의 timeline
>   (2) **세 환경 세 winner** (Val sym → CPCV dsr → Test pt) — 발표의 climax
>   (3) PT OOS 강건성 메커니즘 — hold mean 1.4h, max 6h, sell-side timing risk 회피
>   (4) DSR OOS 실패 메커니즘 — hold mean 4.58h, max 169h (7일!)
>   (5) 정책 안정성 vs 시장 distribution shift (Δ ≤ 5% vs KS p<10⁻¹⁰)
>   (6) **B&H 정직 인정** — Sharpe 0.757 > pt 0.367이지만 Calmar 10배 우위
>   (7) H5 학술 의의 — prospect theory의 RL trading 첫 정량 OOS 적용
>   (8) 인간 표준값 vs Optuna best (α 0.683 < 0.88, λ 3.30 > 2.25)
> 마지막 30분은 적대적 질문: 'B&H가 더 좋잖아?', '강세장 단일이라 일반화 안 되잖아?'"

### 세션 12 — 결론 + 한계
> "오늘은 발표 마지막 5분. 차례로:
>   메인 frame 4가지
>   세 환경 동시 사용 권고 (Henderson 2018 확장)
>   한계 5가지 정직 인정
>   exp027_rl 환경 의존성 정직 인정 (환경 효과 > reward 효과)
>   Future work 3개 우선순위
> '한계를 솔직하게 말하면 점수 깎이지 않을까?' 같은 발표 전략 대화도 포함."

### 세션 13 — 선행 연구
> "오늘은 9개 인용 논문 각각이 본 논문 어디에 기여하는지 + 직접 선행 4편 차별점.
>   Avellaneda, Wilder, Moody, Kahneman, López de Prado×2, Gort, Henderson, Ng et al
>   직접 선행: Liu 2021, Yasin 2024, Pham 2025, Bandarupalli 2025
> 청중이 '이런 논문 있던데?' 라고 다른 논문 던졌을 때 대응도 시뮬레이션."

### 세션 14 — 시각자료 발표 시연
> "오늘은 reading_guide.html의 figure-by-figure 절을 보면서 청중에게 한 장씩 설명하는 시뮬레이션.
>   1순위 3장: Pareto scatter / Hold duration / Three env reversal
>   각 그림 30초 안에 청중에게 설명
> 그림 보지 않고 머릿속으로 그릴 수 있는지 점검."

### 세션 15 — **Mock 발표 + 곡구** ★★
> "오늘은 처음부터 끝까지 15분 발표 흐름 시뮬레이션. 룰:
>   - 내가 한 chapter씩 발표 (큰 그림 → 배경 → 방법론 → 결과 → 한계 → 결론)
>   - 매 chapter 끝나면 1-2개 곡구 질문 던지기
>   - 적대적 질문 / 청중 수준이 갈리는 질문 / '이 부분 더 설명해주세요' 류 다 섞어줘
>   - 마지막에 발표 전체에서 *가장 약했던 답 3개*를 지적해줘"

---

## 세션 운영 팁

### 시작할 때
- 어느 세션인지 한 줄 알림: "오늘은 세션 X — 주제 Y"
- 본인 컨디션 알림 (피곤하면 짧게, 컨디션 좋으면 깊이)

### 진행 중
- 답이 막히면 **숨기지 말고 막혔다고 말하기** → AI가 단서를 줄 것
- jargon-heavy 답이 나왔다 싶으면 즉시 "비전문가 버전으로 다시"
- 1번에 답이 됐어도 **AI가 같은 개념을 다른 각도에서 다시 물도록 요청**

### 마칠 때
- "오늘 가장 약했던 부분 1-2개 정리해줘" 요청
- 그 부분을 `00_KNOWLEDGE_OUTLINE.md`의 체크박스에 메모 → 다음 세션에서 재방문

### 모바일 진행
- 출퇴근/통학 시 음성으로 답하기 → claude.ai 앱의 음성 기능 활용
- 답이 거칠어도 OK. *유창함*이 목표
