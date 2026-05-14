# Live Trading — Paper Trading 인프라

> Binance 실거래 / Testnet 연결을 위한 모듈.
> **본 졸업 논문 범위 외 — 선택적 활용** (sim2real gap 측정 도구).
> 본 논문 RQ는 [`docs/PROJECT_GOAL.md`](../docs/PROJECT_GOAL.md) 참조.

## 구조

```
live_trading/
├── bot.py             # 메인 루프 (1시간봉 완성 시 실행)
├── config.yaml        # 거래소 + 전략 설정
├── exchange.py        # ccxt Binance 래퍼 (잔고/주문/체결)
├── formula.py         # ATR/RL 공식 적용 (src/env와 일치)
└── state_tracker.py   # 포지션·사이클 영속화 (sqlite)
```

## 단계

1. **Testnet 검증** (1주):
   - `.env`에 Binance Testnet API 키 입력
   - `config.yaml`에 `exchange.testnet: true`
   - ATR 고정 정책 (`formula.mode: atr`) 으로 운영
   - sim vs testnet Sharpe/MDD 차이 측정

2. **실계좌 소액** ($100, 1~2주):
   - `exchange.testnet: false`
   - ATR 고정 정책 유지
   - 슬리피지/체결률 정량 측정

3. **RL 정책 전환** (선택):
   - `config.yaml`: `formula.mode: rl`, `rl_model_path: experiments/expXXX/best_model.zip`
   - Phase 3 best variant (exp032 결과) 사용

## 본 논문에서의 위치

- **본 논문 메인 결과에 포함하지 않음** (§8 Discussion에서만 짧게 언급)
- 목적: sim2real gap의 정량 측정 → 시뮬레이터 충실도 검증 도구
- 시간 여유 있으면 Phase 3 후반에 진행

## 안전 수칙

- API 키는 `.env` 파일로 분리, git ignored
- Testnet 먼저, 실계좌는 소액 (`$100` 이하)
- 일일 손실 한도 (`config.yaml: risk.max_daily_loss`) 필수 설정
- 거래소 장애/네트워크 오류 시 자동 정지 로직 확인
