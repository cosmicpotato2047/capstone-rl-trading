# Hierarchical RL in Trading — Options Framework

> Sutton, Precup, Singh (1999). _Between MDPs and Semi-MDPs: A framework for temporal abstraction in reinforcement learning._ Artificial Intelligence, 112(1-2).
> 트레이딩 응용: Wang et al. (2023) "Select and Trade: Towards Unified Pair Trading with Hierarchical RL." KDD.

## 요지

1. **계층적 RL = 정책을 두 층으로 분리.** 상위 정책이 "전략적 결정", 하위 정책이 "전술적 실행".
2. **Options framework**: 하위 정책은 종료 조건 β를 가진 "옵션"이고, 상위 정책은 옵션을 선택. Semi-MDP로 정식화.
3. 트레이딩에 자연스러운 이유: 인간 트레이더가 "포지션 진입 결정"과 "구체적 호가 결정"을 분리하듯, 그리드 봇도 "사이클 시작" vs "호가 위치"가 다른 시간 스케일.

## Options Framework 정식

```
Option o = ⟨I_o, π_o, β_o⟩
  I_o: 진입 조건 (어떤 state에서 이 옵션을 시작 가능한가)
  π_o: 옵션 내부 정책 (옵션이 활성 중일 때의 행동)
  β_o: 종료 조건 (어떤 state에서 옵션이 끝나는가)

상위 정책: μ(o | s) — 옵션 자체를 action처럼 선택
```

→ Semi-MDP 변환: 옵션 하나의 실행이 여러 primitive 시간 step을 차지.
→ Q-learning, Policy gradient 모두 SMDP 버전으로 일반화 가능.

## 트레이딩 응용 패턴

### 패턴 1: 전략-전술 분리

| 층 | 결정 사항 | 시간 스케일 |
|---|---|---|
| 상위 | 진입 / 청산 / 홀드, 위험 노출 수준 | 시간~일 |
| 하위 | 구체적 호가, 주문 크기 분할 | 분~시간 |

### 패턴 2: Pair Trading (KDD 2023)
- 상위: 모든 가능한 자산 쌍 중 어떤 쌍을 거래할지 (discrete, K choose 2)
- 하위: 선택된 쌍에 대해 long/short/hold 행동
- 결과: 단일 정책 대비 명확한 Sharpe 향상

### 패턴 3: Regime-switching trading
- 상위: 현재 시장 regime 분류 (Bull/Bear/Sideways)
- 하위: regime별로 별도 학습된 sub-policy 활성
- 우리 프로젝트의 가설에 가장 가까움

## 우리 프로젝트와의 연결점

1. **현재 우리 시스템은 명시적 계층 구조 없음**:
   - 단일 PPO가 매 스텝 [aggressiveness, profit_target] 결정
   - exp029에서 "사이클 시작 시 1회 결정 (5D action)"으로 부분적 계층화 시도 → 이미 options framework의 한 변형

2. **계층화의 이론적 정식화로 exp029 정당화 가능**:
   - 상위: "지금 사이클 시작할지" (또는 새 그리드 설정 결정)
   - 하위: 사이클 내 자동 체결 (환경이 처리)
   - 종료 조건 β: holdings == 0 (전량 청산)
   - → **이게 정확히 options framework**. exp029의 의도를 학술적 언어로 표현한 것.

3. **확장 가능한 설계 (2학기)**:
   - **상위 정책**: regime 분류기 (LSTM/Transformer로 trend, volatility 입력)
   - **하위 정책**: regime별 그리드 파라미터 결정
   - → 단일 PPO가 못 풀던 "BTC 강세장 vs 횡보장" 일반화 문제 우회

4. **즉시 적용 가능한 implementation 패턴**:
   ```python
   # 의사 코드
   class HierarchicalGridAgent:
       high_policy: PPO          # 4-class output: enter/exit/hold/halt
       low_policies: dict[regime, PPO]  # regime별 grid 파라미터

       def step(self, state):
           regime = self.classify(state)
           if not self.in_cycle:
               decision = self.high_policy(state)
               if decision == ENTER:
                   self.params = self.low_policies[regime](state)
           # ATR + params로 호가 계산 (환경)
   ```

5. **단점/위험**:
   - 학습량 2~3배 증가 (정책 두 개)
   - High-level policy의 reward 신호가 sparse → exploration 어려움
   - Option discovery (어떤 옵션을 만들지) 자체가 풀리지 않은 문제

## 백링크

- [[reward_shaping_ng1999]] — Hierarchical reward 시그널 설계
- [[curriculum_learning]] — 계층 학습 시 점진적 난이도
- [[avellaneda_stoikov_2008]] — 시장조성의 inventory control이 hierarchy의 자연스런 예

## 출처

- [Sutton, Precup, Singh (1999) — Options framework](https://www.sciencedirect.com/science/article/pii/S0004370299000521)
- [Select and Trade (KDD 2023)](https://dl.acm.org/doi/10.1145/3580305.3599951)
- [Options Trading RL (MDPI)](https://www.mdpi.com/2076-3417/11/23/11208)
