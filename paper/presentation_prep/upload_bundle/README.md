# Presentation Prep Bundle — 발표 준비 자료 모음

> 1학기 캡스톤 보고서 발표 준비용 패키지.
> Web Claude (claude.ai) 의 **Projects** 기능과 함께 사용하도록 설계됨.

---

## 폴더 구조

```
presentation_prep/
├── README.md                       ← 본 파일 (사용법)
├── 00_KNOWLEDGE_OUTLINE.md         ← 알아야 할 것 항목화 (자가 점검)
├── 01_DRILL_SESSIONS.md            ← 드릴 세션 15개 (순서 + 트리거 문장)
├── 02_PRESENTATION_QA.md           ← 예상 질문 63문 + 모범 답안
├── 03_PIVOT_TIMELINE.md            ← Pivot 의사결정 narrative
├── 04_DASHBOARD_NOTES.md           ← 자산 대시보드 발표 노트 (template)
│
└── upload_bundle/                  ← 이 폴더 통째로 Web Claude에 업로드
    ├── main_ko.pdf                 (한글 논문)
    ├── READING_GUIDE.md            (비전문가용 가이드)
    ├── PROJECT_GOAL.md             (RQ, 가설, scope)
    ├── RESEARCH_LOG.md             (날짜별 의사결정)
    ├── ROADMAP.md                  (Phase별 진행)
    ├── MDP.md                      (환경 설계 근거)
    ├── FORMULAS.md                 (ATR / RL 공식)
    ├── RELATED_WORK.md             (선행 연구)
    ├── RESULTS_SUMMARY.md          (Phase별 수치)
    ├── 00_KNOWLEDGE_OUTLINE.md     (자가 점검)
    ├── 01_DRILL_SESSIONS.md        (세션 가이드)
    ├── 02_PRESENTATION_QA.md       (QA 모범 답안)
    ├── 03_PIVOT_TIMELINE.md        (pivot narrative)
    └── 04_DASHBOARD_NOTES.md       (대시보드 template)
```

총 14개 파일 (1개 PDF + 13개 markdown).

---

## Web Claude Projects 세팅 (5분)

1. **claude.ai 로그인** → 좌측 sidebar → "Projects" → "New Project"
2. **이름**: `캡스톤 발표 드릴`
3. **Project Knowledge**: `upload_bundle/` 폴더의 14개 파일 *전부 드래그 업로드*
   - 한 번에 안 되면 4-5개씩 나눠서. PDF 먼저 올리고 markdown은 나중에.
4. **Custom Instructions** 박스에 아래 프롬프트 붙여넣기:

```
나는 BTC 그리드 트레이딩 강화학습 캡스톤 논문을 1학기 보고서로 발표 예정이다.
발표 직전 드릴 파트너로서 다음을 수행하라:

[원칙]
1. 발표자가 익혀야 할 핵심 개념을 한 번에 하나씩 묻고, 답변의 정확성·간결성·
   비전문가 청중에 맞는 비유 사용 여부를 평가하라.
2. 답이 부정확하거나 자신 없이 들리면 즉시 핵심을 짚어 정정하고, 같은 개념을
   2~3 차례 다른 각도에서 다시 물어 내재화 여부를 확인하라.
3. "왜 그 reward를 골랐는가", "왜 그 시점에 pivot 했는가", "왜 dsr이 CPCV는
   1위, Test는 꼴찌인가" 류의 **이유 / 메커니즘 질문**을 우선 던지고, 단순
   수치 암기 질문은 마지막에 배치하라.
4. 발표 청중은 RL/금융 비전문가 (가족·친구·교수님 일부 포함) 라는 점을 항상
   염두에 두고, 답변이 과도하게 jargon-heavy 하면 즉시 지적하라.
5. 한 세션은 한 주제로 묶어 진행 (예: "오늘은 H5/Prospect Theory 위주").
   세션 시작 시 어느 주제를 다룰지 먼저 확인하라.

[자료 사용 가이드]
- `00_KNOWLEDGE_OUTLINE.md`: 발표자가 알아야 할 항목 트리. 드릴의 *체크리스트*.
- `01_DRILL_SESSIONS.md`: 15개 세션 가이드. 사용자가 어느 세션인지 알려주면
  해당 세션 가이드 따라 진행.
- `02_PRESENTATION_QA.md`: 63문 예상 질문 + 모범답안. 드릴 중 추가 질문이
  필요할 때 이 문서에서 가져오거나 직접 생성. 모범답안은 *참고용*이며 그대로
  암기하지 않게 (자기 말로 설명하도록).
- `03_PIVOT_TIMELINE.md`: 본 논문 narrative의 핵심. 의사결정 시점을 시간 순으로
  답할 수 있는지 우선 점검.
- `04_DASHBOARD_NOTES.md`: 별개 프로젝트 (자산 대시보드). 본 논문과 헷갈리지
  않게 시간 단위 차이 (1시간봉 vs 격주) 강조.
- `READING_GUIDE.md`: 비전문가용 종합 정리. 청중 눈높이 답변 어떻게 할지
  참고.
- `main_ko.pdf`: 정확한 수치, 그림 번호, table 번호 — ground truth.

[세션 운영 룰]
- 사용자가 세션 번호 (예: "오늘은 세션 11") 알리면 `01_DRILL_SESSIONS.md`의
  해당 세션 트리거를 참조해 시작.
- 세션 끝에 *오늘 가장 약했던 답 1-2개*를 정리해 사용자에게 전달.
- 매 세션 30~45분 안에 마무리 (60분 초과 세션은 11, 15만).
```

5. **Save** → 이제 새 대화 만들 때마다 위 14개 파일 + 위 instructions가 자동 적용됨.

---

## 첫 세션 시작하는 법

새 대화 생성 후 첫 메시지:

```
오늘은 세션 1 — Big Picture 다지기.
01_DRILL_SESSIONS.md 의 세션 1 트리거대로 진행해줘.
```

또는 더 짧게:
```
세션 1 시작.
```

→ AI가 자동으로 30초 → 5분 → 30분 답변 연습을 차례로 시킴.

---

## 자가 점검 체크리스트 사용법

`00_KNOWLEDGE_OUTLINE.md` 의 모든 항목에 `[ ]` 체크박스. 드릴 진행하며:
1. 막힘없이 답 가능 → `[x]` 로 변경
2. 막혔던 부분은 `[!]` 로 표시 + 짧은 메모
3. 매 세션 끝에 `[!]` 항목 다음 세션에서 재방문

GitHub Desktop이나 VS Code에서 직접 편집 추천 (Web Claude에서는 읽기만).

---

## 권장 일정 (D-14 ~ D-0)

`01_DRILL_SESSIONS.md` 의 권장 일정 참조.
- D-14 ~ D-10: 기초 다지기 (세션 1-6)
- D-9 ~ D-5: 메인 실험 (세션 7-11)
- D-4 ~ D-1: 결론·시각자료·mock 발표 (세션 12-15)
- D-0: 발표

발표 직전 D-1 의 **세션 15 (mock 발표 + 곡구)** 은 *반드시*.

---

## 자주 묻는 메타 질문

### Q. PDF를 어차피 봉인했는데 Web Claude에 올려도 되나요?
A. **Test partition 데이터**는 봉인 대상이지, **논문 PDF**는 아닙니다. 논문은 발표용이라 자유롭게 사용. 단, *test parquet 파일*은 절대 업로드 금지.

### Q. Custom Instructions를 다 안 쓰면 어떻게 되나요?
A. AI가 일반적 Q&A로 답하게 됩니다. 드릴 효과 떨어짐. 가능하면 전체 붙여넣기 권장.

### Q. Project 한도가 초과되면?
A. Free 플랜은 file 수 제한 있음 (5개?). 그런 경우 (1) main_ko.pdf + (2) READING_GUIDE.md + (3) 02_PRESENTATION_QA.md + (4) 03_PIVOT_TIMELINE.md 4개만 우선. 나머지는 필요 시 대화에 첨부.

### Q. 모바일에서 가능한가요?
A. 가능. Claude.ai 앱의 음성 입력 기능으로 답하면 *발표 톤 연습*까지 됨.

### Q. 본 가이드를 GitHub에 같이 공개할 건가요?
A. 본인 결정 사항. 공개 시 `presentation_prep/` 폴더 통째로 공개 가능. 04_DASHBOARD_NOTES.md 는 개인 자산 정보 들어가면 비공개 권장.

---

## Claude Code (지금 여기) vs Web Claude 역할 분담

| 작업 | Claude Code (지금) | Web Claude |
|---|---|---|
| 번들 / 가이드 자료 생성 | ✅ | ❌ |
| 발표 직전 일별 드릴 | ❌ | ✅ |
| 자료 업데이트 (수치 확인 등) | ✅ | ❌ |
| 모바일 음성 드릴 | ❌ | ✅ |
| 막판 디테일 다듬기 | ✅ | △ |

→ 자료 변경이 필요할 때만 여기 (Claude Code) 로 돌아오기.

---

**작성**: Claude Code (2026-05-21)
**대상 발표일**: ___ <!-- 본인 채워넣기 -->
**현재 진행**: 자료 패키지 완성, Web Claude Projects 세팅 대기
