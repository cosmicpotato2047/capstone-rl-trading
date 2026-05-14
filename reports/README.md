# Reports — 학기별 보고서

> 본 디렉토리는 주간/학기 보고서, 발표 자료, 논문 초고를 담는다.
> 본 논문의 RQ는 [`docs/PROJECT_GOAL.md`](../docs/PROJECT_GOAL.md) 참조.

## 구조

```
reports/
├── WEEKLY_TEMPLATE.md          # 주간 보고서 템플릿
├── project_journey.html        # 프로젝트 전체 흐름 인터랙티브 시각화
├── semester1/                  # 1학기 (2026-03 ~ 2026-06)
│   ├── week05_2026-04-10.md
│   ├── week07_presentation(midterm_report).md  # 중간 보고서
│   └── figures/                # 그래프 PNG
├── semester2/                  # 2학기 (2026-08 ~ )
│   └── figures/                # (작성 시 추가)
└── paper/                      # 최종 논문 초고 (LaTeX)
```

## 보고서 작성 규칙

- **주간 보고서**: `WEEKLY_TEMPLATE.md` 복사 후 `weekXX_YYYY-MM-DD.md` 명명
- **중간/기말 보고서**: 학기 디렉토리에 직접
- **figures/**: PNG/SVG만. 원본 데이터는 `experiments/expXXX/`에
- **paper/**: LaTeX 소스. 최종 PDF는 git ignored (`.gitignore`)

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
