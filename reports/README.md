# Reports — 학기별 보고서 + 실험 분석

> 본 디렉토리는 학기 발표 자료(`semester1/`)와 실험별 분석 보고서(`exp032b/` ~ `phase16d/`)를 담는다.
> 최종 논문 LaTeX 소스는 root의 [`paper/`](../paper/)에 있다.
> 본 논문의 RQ는 [`docs/PROJECT_GOAL.md`](../docs/PROJECT_GOAL.md) 참조.

## 구조

```
reports/
├── semester1/                  # 1학기 (2026-03 ~ 2026-06) 발표 자료
│   ├── week02_연구주제.pptx ~ week11_knowledge_map.html
│   └── figures/                # 발표용 그래프 PNG
├── exp032b/                    # exp032b 분석 + figures
├── exp032c/                    # exp032c 분석 + figures
├── exp033/                     # 슬리피지 강건성
├── exp034/                     # CPCV 6-fold
├── exp035/                     # OOS Test 봉인 해제
├── phase15/                    # Phase 15 (B&H baseline, distribution shift)
└── phase16d/                   # Phase 16d (hold duration, OOS mechanism)
```

## 보고서 작성 규칙

- **실험 분석**: `expXXX/analysis.md` + `figures/` 표준
- **figures/**: PNG/SVG만. 원본 데이터는 `experiments/expXXX/`에
- **논문 LaTeX 소스**: [`paper/`](../paper/)에서 빌드 (root 레벨, reports/ 밖)

## 본 논문 작성 시 챕터별 출처

| 챕터 | 핵심 출처 |
|---|---|
| 1. Introduction | `docs/PROJECT_GOAL.md` (RQ, 가설) |
| 2. Background | `docs/study/rl_finance/{avellaneda_stoikov_2008, zhang_zohren_roberts_2020, gort_2022_crypto_overfitting}.md` |
| 3. Method | `docs/{MDP.md, FORMULAS.md}` + `docs/study/rl_finance/{ppo_schulman_2017, reward_shaping_ng1999}.md` |
| 4. Negative finding (RQ-1) | `RESEARCH_LOG.md` (exp020~022 결과) |
| **5. Positive finding (RQ-2, 메인)** | **`experiments/exp032_reward_variants/` + `RESEARCH_LOG`** |
| 6. Mechanism (RQ-3) | exp032 행동 분석 + `docs/study/rl_finance/prospect_theory.md` |
| 7. Robustness (RQ-4) | `experiments/exp033~035/` + `docs/study/rl_finance/walk_forward_cv.md` |
| 8. Discussion | `docs/study/rl_finance/00_overview.md` + `optimal_grid_spacing.md` |
| 9. Conclusion | — |

## 학기 일정

```
Semester 1 (2026-03 ~ 2026-06): Phase 1~2 완료 + 중간 보고서
Semester 2 (2026-08 ~ 2026-12): Phase 3 (exp030~035) + 논문 작성 + 디펜스
```
