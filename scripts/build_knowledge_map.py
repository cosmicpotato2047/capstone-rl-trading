"""
중간 보고서용 Knowledge Map HTML 빌드 스크립트.

자료 조사 ↔ 실험 설계 ↔ RQ 진화 의 매핑을 단일 self-contained HTML 로 시각화.

출력: reports/knowledge_map.html (단일 파일, vis-network + marked.js CDN)

사용:
    python scripts/build_knowledge_map.py
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


# ------------------------------------------------------------------
# 1. 메타데이터: Bundle 분류
# ------------------------------------------------------------------

BUNDLES = {
    "A": {
        "name": "A. RL × 금융 (직결도 최고)",
        "color": "#1976D2",
        "desc": "본 논문 RQ 및 메인 챕터의 직접 출처. Reward design, DRL trading 등.",
    },
    "B": {
        "name": "B. RL 이론 기초",
        "color": "#7B1FA2",
        "desc": "MDP, PPO, reward shaping 등 §3 Method 의 토대.",
    },
    "C": {
        "name": "C. 그리드의 학술적 뿌리",
        "color": "#388E3C",
        "desc": "Avellaneda-Stoikov 등 §2 Background 의 학술적 조상.",
    },
    "D": {
        "name": "D. 약점 보강 (실험 1:1)",
        "color": "#E65100",
        "desc": "exp030~034 각 실험과 1:1 매칭되는 방법론 노트.",
    },
    "E": {
        "name": "E. exp032 정합성 보강",
        "color": "#C62828",
        "desc": "공정 비교 + 메커니즘 + 통계 정직성 (2026-05-14 추가).",
    },
}

# ------------------------------------------------------------------
# 2. 노트 메타데이터 (00_overview.md 의 매핑을 코드로 인코딩)
# ------------------------------------------------------------------

# slug → (title 단축형, bundle, role, related_exp_list)
NOTES_META = {
    "differential_sharpe_moody2001": (
        "A1. Differential Sharpe (Moody 2001)", "A",
        "Sharpe 를 매 스텝 reward 로 직접 사용. exp032 DSR variant 이론적 출처.",
        ["exp032a", "exp032b", "exp032c"],
    ),
    "zhang_zohren_roberts_2020": (
        "A2. Zhang-Zohren-Roberts (2020)", "A",
        "Volatility scaling reward, DRL trading 표준. §2 Background 핵심 인용.",
        ["§2"],
    ),
    "gort_2022_crypto_overfitting": (
        "A3. Gort 2022 — Crypto Overfitting", "A",
        "PBO/CPCV 방법론. 백테스트 과적합 검증의 출처.",
        ["exp034", "exp035"],
    ),
    "finrl_framework": (
        "A4. FinRL Framework", "A",
        "DRL trading 표준 라이브러리. §2 Background 에서 대비 인용.",
        ["§2"],
    ),
    "reward_hacking": (
        "A5. Reward Hacking / Specification Gaming", "A",
        "exp026 체결가 버그가 reward hacking 사례. §3.1 안전 설계 근거.",
        ["exp026", "exp032"],
    ),
    "sim2real_finance": (
        "A6. Sim2Real Gap in RL Trading", "A",
        "시뮬레이터-실거래 갭. §7.1 Robustness + Phase 5 시사.",
        ["exp033"],
    ),
    "distributional_rl": (
        "A7. Distributional RL (C51, QR-DQN)", "A",
        "CVaR-aware policy, fat tail 대응. Discussion 시사.",
        ["§8"],
    ),
    "hierarchical_rl_trading": (
        "A8. Hierarchical RL in Trading", "A",
        "전략/전술 분리. exp029 사이클 단위 결정의 학술적 출처.",
        ["exp029", "§8"],
    ),
    "mdp_bellman_pomdp": (
        "B1. MDP / Bellman / POMDP", "B",
        "RL 의 수학적 framework. §2 Background.",
        ["§2"],
    ),
    "ppo_schulman_2017": (
        "B2. PPO (Schulman 2017)", "B",
        "본 논문 채택 알고리즘. §3.2 Method.",
        ["exp030"],
    ),
    "ddpg_continuous_control": (
        "B3. DDPG — Continuous Control", "B",
        "PPO 대안 비교. §3.2 알고리즘 선택 근거.",
        ["§3"],
    ),
    "reward_shaping_ng1999": (
        "B4. Reward Shaping (Ng 1999)", "B",
        "Potential-based shaping 의 정책 불변성. exp029 r_idle 안전성 근거.",
        ["exp029", "exp030", "exp032"],
    ),
    "prospect_theory": (
        "B5. Prospect Theory (Kahneman-Tversky 1979)", "B",
        "Asymmetric / PT reward 의 행동경제학적 출처. §3.4 Method.",
        ["exp027_rl", "exp032a", "exp032b"],
    ),
    "avellaneda_stoikov_2008": (
        "C1. Avellaneda-Stoikov (2008)", "C",
        "마켓 메이킹 정석. 그리드 봇의 학술적 조상. §2 Background / §1 Intro.",
        ["§1", "§2"],
    ),
    "inventory_risk_adverse_selection": (
        "C2. Inventory Risk & Adverse Selection", "C",
        "Glosten-Milgrom, Kyle, Ho-Stoll. §2 시장미시구조.",
        ["§2"],
    ),
    "optimal_grid_spacing": (
        "C3. Optimal Grid Spacing & Volatility Harvesting", "C",
        "그리드의 학술적 정당화 + Short-vol 위험. §2, §8.",
        ["§2", "§8"],
    ),
    "bayesian_optimization_tpe": (
        "D1. Bayesian Opt — TPE & Hyperband", "D",
        "Optuna 정당화 + DSR 보정. exp034 통계 검증의 토대.",
        ["exp032a", "exp034"],
    ),
    "walk_forward_cv": (
        "D2. Walk-Forward + Purged K-Fold + CPCV", "D",
        "exp027 Val 과적합 → CPCV 6-fold 동기. exp034 메인 도구.",
        ["exp034", "§7.2"],
    ),
    "policy_gradient_stabilization": (
        "D3. PPO 학습 안정화 — 실전 기법", "D",
        "exp028/029 학습 oscillation 해결. exp030 안정화 패키지.",
        ["exp030"],
    ),
    "realistic_execution_simulation": (
        "D4. 현실적 체결 시뮬레이션", "D",
        "Slippage, partial fill, queue. exp033 + §7.1.",
        ["exp033"],
    ),
    "volatility_modeling": (
        "D5. Volatility Modeling (GARCH, RV, Jumps)", "D",
        "ATR 너머 변동성 모델링. Discussion 시사.",
        ["§8"],
    ),
    "offline_rl_warm_start": (
        "D6. Offline RL + BC Warm Start", "D",
        "학습 초반 낭비 회피. exp031 동기.",
        ["exp031"],
    ),
    "cql_kumar_2020": (
        "D6+. CQL (Kumar 2020)", "D",
        "Mixed-policy offline RL. exp031b 조건부 사용.",
        ["exp031b"],
    ),
    "curriculum_learning": (
        "D7. Curriculum Learning + Domain Randomization", "D",
        "일반화 + DR. exp033 의 학습 방식.",
        ["exp033"],
    ),
    "effect_size_rliable": (
        "E1. Effect Size & rliable", "E",
        "Cohen's d, IQM, BEST, rliable. Variant 비교의 통계적 정직성.",
        ["exp032b", "§5", "§7.2"],
    ),
    "causal_counterfactual_rl": (
        "E2. Causal & Counterfactual Analysis", "E",
        "RQ-3 메커니즘 답변 강화. counterfactual, SHAP, mediation.",
        ["exp032c", "§6"],
    ),
    "hyperparameter_parity": (
        "E3. Hyperparameter Parity", "E",
        "Variant 별 reward hyperparameter 공정 튜닝의 정당성.",
        ["exp032a", "§3.5"],
    ),
}

# ------------------------------------------------------------------
# 3. 실험 데이터 (RESULTS_SUMMARY.md 발췌, 상태 + 핵심 결과)
# ------------------------------------------------------------------

EXPERIMENTS = [
    {
        "id": "exp030",
        "title": "PPO 학습 안정화",
        "chapter": "§3.3 Method",
        "status": "completed",
        "status_label": "완료 (2026-05-14)",
        "objective": "Best step 100k → 550k 늦추기 + final 붕괴 완화",
        "design": "LR linear decay 3e-4→1e-5, target_kl 0.02, ent annealing 0.01→0.001, patience 10",
        "result": "Val Sharpe best 1.974 (ATR baseline 1.505 대비 +31%). 1M 완주. 후반 붕괴 패턴 미해결 → exp031 동기.",
        "verdict": "부분 성공",
        "notes": ["policy_gradient_stabilization", "ppo_schulman_2017", "reward_shaping_ng1999"],
    },
    {
        "id": "exp031",
        "title": "BC Action Bias Init",
        "chapter": "§3.4 Method",
        "status": "abandoned",
        "status_label": "폐기 (2026-05-14)",
        "objective": "PPO action_net bias 를 ATR 행동에 매칭 → 학습 초반 낭비 회피",
        "design": "bias=[-10,-10], [-3,-3] 두 시도",
        "result": "두 시도 모두 학습 정체 (Sharpe 1.526 동일, early stop). SB3 PPO + Box[0,1] clipping 이슈.",
        "verdict": "Negative result, future work 로 보류",
        "notes": ["offline_rl_warm_start", "cql_kumar_2020"],
    },
    {
        "id": "exp032a",
        "title": "Reward Variant Hyperparameter Tuning",
        "chapter": "§3.5 Method",
        "status": "completed",
        "status_label": "완료 (2026-05-15)",
        "objective": "Variant 별 reward hyperparameter (β, η, α, λ) 공정 튜닝",
        "design": "Optuna TPE 30 trials × 200k steps × 3 variant (sym 제외)",
        "result": "asym β=3.42, dsr η=0.035 (≈ 1/28h EMA), pt α=0.68 λ=3.30. 200k single-seed Sharpe: dsr 1.89 / pt 1.80 / asym 1.52.",
        "verdict": "exp032b 입력으로 4 config 확정",
        "notes": ["hyperparameter_parity", "bayesian_optimization_tpe", "prospect_theory", "differential_sharpe_moody2001"],
    },
    {
        "id": "exp032b",
        "title": "4 Reward Variant × 10 Seeds 본 비교",
        "chapter": "§5 Positive finding (메인)",
        "status": "completed",
        "status_label": "완료 (2026-05-15) — §5 메인",
        "objective": "4 variant (sym/asym/dsr/pt) 의 reward design 효과 검증",
        "design": "40 runs × 1M = 40M steps. Effect size (Cohen's d) 본 분석.",
        "result": "sym 1.871 / dsr 1.809 / asym 1.681 / pt 1.667 (Best Sharpe ± std). 4 variant 모두 ATR(1.505) +11~24% 초과. 두 cluster {aggressive: sym,dsr} vs {conservative: asym,pt} 통계 분리.",
        "verdict": "시나리오 D (Pareto frontier) 확정. H2-weak 지지, H2-strong 부분 부정, H3 지지.",
        "notes": ["differential_sharpe_moody2001", "prospect_theory", "effect_size_rliable", "hyperparameter_parity", "reward_shaping_ng1999"],
    },
    {
        "id": "exp032c",
        "title": "Mechanism Analysis",
        "chapter": "§6 Mechanism (메인)",
        "status": "completed",
        "status_label": "완료 (2026-05-15) — §6 메인",
        "objective": "두 cluster 분리의 인과 메커니즘 정량화",
        "design": "5 메뉴 분석: Pareto scatter / SHAP / Counterfactual / Behavior per regime / Policy distance matrix",
        "result": "Policy distance within-cluster 0.129 vs across-cluster 0.286 (ratio 2.22×). DSR hold rate 2~6× (high_vol). H3 selective entry 강한 지지.",
        "verdict": "Reward 형식 → 행동 → 결과 의 인과 사슬 정량 확인.",
        "notes": ["causal_counterfactual_rl", "effect_size_rliable", "differential_sharpe_moody2001"],
    },
    {
        "id": "exp033",
        "title": "Slippage 0.02% Robustness",
        "chapter": "§7.1 Robustness",
        "status": "completed",
        "status_label": "완료 (2026-05-15) — §7.1",
        "objective": "Slippage 도입 후 cluster 구조 보존 여부 검증",
        "design": "40 runs × 1M with slippage_rate=0.0002",
        "result": "모든 variant ~12% Sharpe 감쇠. Cluster preservation ratio 2.19× ≈ exp032b 2.22×. asym/pt 가 ATR-no-slippage 아래로 marginal 하락.",
        "verdict": "Cluster 구조 robust. H4 (Slippage) 부분 지지.",
        "notes": ["realistic_execution_simulation", "sim2real_finance", "curriculum_learning"],
    },
    {
        "id": "exp034",
        "title": "CPCV 6-fold + DSR",
        "chapter": "§7.2 Robustness",
        "status": "pending",
        "status_label": "대기 (다음 실험)",
        "objective": "CPCV path Sharpe 분포 + DSR (Deflated Sharpe) 다중검정 보정",
        "design": "Train+Val 안에서 6-fold → 15 paths. Test 봉인 유지.",
        "result": "—",
        "verdict": "—",
        "notes": ["walk_forward_cv", "bayesian_optimization_tpe", "gort_2022_crypto_overfitting", "effect_size_rliable"],
    },
    {
        "id": "exp035",
        "title": "Test Set 봉인 해제 (최종)",
        "chapter": "§7.3 Robustness",
        "status": "pending",
        "status_label": "대기 (최종)",
        "objective": "CPCV 통과 후 Test 2024+ 1회 평가",
        "design": "봉인 해제 후 1회 평가, 재학습 금지",
        "result": "—",
        "verdict": "—",
        "notes": ["gort_2022_crypto_overfitting", "walk_forward_cv"],
    },
]


# ------------------------------------------------------------------
# 4. RQ 진화 타임라인
# ------------------------------------------------------------------

RQ_TIMELINE = [
    {
        "date": "2026-04 초",
        "phase": "Phase 1",
        "title": "초기 RQ: RL 단독 우위 검증",
        "rq": "PPO 가 BTC 그리드에서 baseline 대비 우위인가?",
        "evidence": "exp001~016. Phase 1 best Val Sharpe 35.4 (exp016).",
        "decision": "—",
        "type": "initial",
    },
    {
        "date": "2026-04-21",
        "phase": "Pivot 1",
        "title": "Decisive Ablation 발견",
        "rq": "RL 학습 결과가 Fixed [1.0, 0.0] 와 Val Sharpe 완전 동일 (45.390)",
        "evidence": "exp020. 5D state 학습 정책 = constant action.",
        "decision": "RQ 전환: 'RL 우위' → 'ATR vs RL 비교'. § Negative finding 의 출발.",
        "type": "pivot",
    },
    {
        "date": "2026-04-22",
        "phase": "정합성 수정",
        "title": "체결가 favorable bias 발견",
        "rq": "next_low/next_high 체결 → 수조% return artifact",
        "evidence": "exp026. 지정가 체결로 수정.",
        "decision": "이전 결과 (Sharpe 60+) 무효화. ATR Bayesian 재최적화 (Val Sharpe 1.978 로 현실화).",
        "type": "fix",
    },
    {
        "date": "2026-04-23",
        "phase": "Positive 단초",
        "title": "Asymmetric reward 발견",
        "rq": "exp027_rl: asym β=2.0 으로 Test Sharpe 1.955 (ATR 0.935 대비 +109%)",
        "evidence": "Env-v3 (4D 절대 gap). 거래 214 vs ATR 1591, MDD 0.39% vs 2.43%.",
        "decision": "Reward design 이 알파 채널이라는 가설. 본 논문의 메인 contribution 후보.",
        "type": "discovery",
    },
    {
        "date": "2026-05-14 (1차)",
        "phase": "Pivot 2",
        "title": "RQ를 reward design 중심으로 재정의 (단정문)",
        "rq": "'Reward design 이 RL 알파의 핵심 채널이다' (단정 thesis)",
        "evidence": "exp027_rl 사전 증거 + 학습 자료 27개 정리.",
        "decision": "자산 확장 제외 (BTC 단일 자산). exp032 4 variant 비교를 메인 챕터로 결정.",
        "type": "pivot",
    },
    {
        "date": "2026-05-14 (2차)",
        "phase": "RQ 정합성",
        "title": "단정문 → 열린 질문 형태로 수정",
        "rq": "'어떤 reward 함수 하에서 RL 이 ATR 을 초과하며 (혹은 초과하지 못하며), 그 메커니즘은 무엇인가?'",
        "evidence": "학술 컨벤션 + 확증 편향 회피.",
        "decision": "사전 증거는 가설 H1~H4 의 정당성으로 흡수, RQ 는 검증 결과에 따라 어느 쪽이든 살아남는 형태로.",
        "type": "refine",
    },
    {
        "date": "2026-05-14 (자료조사)",
        "phase": "이론 보강",
        "title": "Bundle E 추가 (exp032 정합성)",
        "rq": "—",
        "evidence": "공정 비교 + 메커니즘 + 통계 정직성 약점 발견.",
        "decision": "노트 3개 추가: hyperparameter_parity, effect_size_rliable, causal_counterfactual_rl. exp032 → 032a/b/c 3단계로 확장.",
        "type": "refine",
    },
    {
        "date": "2026-05-15",
        "phase": "본 검증",
        "title": "exp032b 결과 → 시나리오 D 확정",
        "rq": "4 variant 모두 ATR 초과, 단 4 metric 1위 다름. 두 cluster 분리.",
        "evidence": "40 runs × 1M. sym 1.87 / dsr 1.81 / asym 1.68 / pt 1.67.",
        "decision": "사전 시나리오 A/B/C 외에 시나리오 D (Pareto frontier in risk space) 추가 등재 + 확정. 본 논문 §5 thesis: 'single alpha source' 아닌 'risk profile trade-off'.",
        "type": "discovery",
    },
    {
        "date": "2026-05-15",
        "phase": "메커니즘",
        "title": "exp032c Mechanism + exp033 Slippage 완료",
        "rq": "두 cluster 분리는 손실 비대칭 + DSR 메모리 구조가 정책 거래 빈도를 직접 결정한 결과.",
        "evidence": "Policy distance ratio 2.22× (exp032c). Slippage 후 ratio 2.19× 유지 (exp033).",
        "decision": "§6 Mechanism + §7.1 Robustness 완료. 다음 exp034 (CPCV+DSR).",
        "type": "current",
    },
]


# ------------------------------------------------------------------
# 5. 가설 × 실험 매트릭스
# ------------------------------------------------------------------

HYPOTHESES = [
    {
        "id": "H1",
        "text": "Symmetric reward + ATR 비례 공식에서 RL ≈ ATR (RL 추가 알파 없음)",
        "results": {
            "exp020~022": "지지 (Env-v2, 단 favorable bias)",
            "exp030":     "부정 (Env-v4, sym best 1.974 > ATR 1.505)",
            "exp032b":    "부정 (sym best 1.871 > ATR 1.505)",
        },
    },
    {
        "id": "H2-weak",
        "text": "asym/dsr/pt > ATR",
        "results": {
            "exp027_rl": "지지 (Test 1.955 vs 0.935, Env-v3)",
            "exp032b":   "지지 (4 variant 모두 ATR +11~24%)",
            "exp033":    "부분 지지 (Slippage 후 asym/pt 가 ATR-no-slippage 아래)",
        },
    },
    {
        "id": "H2-strong",
        "text": "asym/dsr/pt > sym",
        "results": {
            "exp027_rl": "지지 (Env-v3, β=2.0)",
            "exp032b":   "부분 부정 (dsr ≈ sym, asym/pt < sym)",
        },
    },
    {
        "id": "H3",
        "text": "Reward variant 우위는 'selective entry' (낮은 빈도, 높은 승률) 로 나타남",
        "results": {
            "exp027_rl": "지지 (214 vs 1591 거래)",
            "exp032b":   "지지 (asym/pt 거래 ~75% of sym)",
            "exp032c":   "강한 지지 (Policy distance 2.22×, DSR hold rate 2~6×)",
        },
    },
    {
        "id": "H4",
        "text": "Slippage + CPCV 환경에서도 cluster 우위 유지",
        "results": {
            "exp033": "부분 지지 (Slippage 후 cluster ratio 2.19× 유지)",
            "exp034": "검증 예정",
            "exp035": "검증 예정",
        },
    },
]


# ------------------------------------------------------------------
# 6. 외부 참조 (배경 지식)
# ------------------------------------------------------------------

EXTERNAL_REFS = [
    ("learning list", "학습 로드맵 원본 (Sutton & Barto, PPO 논문, López de Prado 등 1순위 학습)"),
    ("퀀트투자", "큰 그림 — 퀀트 투자의 전체 맥락"),
    ("주요 방법론 상세", "방법론 7종"),
    ("factor / Fama-French 3-5 factor", "팩터 모델 배경 (Reward design 과 직접 관련 없음)"),
    ("Backtest Statistics (López de Prado)", "§7.2 검증과 직결"),
    ("The Dangers of Backtesting (López de Prado)", "§7.2 검증과 직결"),
    ("Backtesting / Pitfalls / Money & Risk Management (Ernie Chan)", "백테스트 함정 사례"),
    ("Special Topics in Quantitative Trading (Ernie Chan)", "실거래 특화 주제"),
    ("Fractionally Differentiated Features", "시계열 특성 추출 (현재 미사용)"),
    ("risk-adjusted return", "Sharpe, Sortino, Calmar 등 평가지표 정의"),
]


# ------------------------------------------------------------------
# 7. 노트 본문 로드
# ------------------------------------------------------------------

def load_note_bodies(notes_dir: Path) -> dict[str, str]:
    bodies = {}
    for slug in NOTES_META:
        p = notes_dir / f"{slug}.md"
        if p.exists():
            bodies[slug] = p.read_text(encoding="utf-8")
        else:
            bodies[slug] = f"# {slug}\n\n(노트 파일을 찾을 수 없습니다: {p})"
    return bodies


# ------------------------------------------------------------------
# 8. 그래프 노드/엣지 빌드
# ------------------------------------------------------------------

def build_graph_data() -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    edges: list[dict] = []

    # 실험 노드
    for exp in EXPERIMENTS:
        status_color = {
            "completed": "#388E3C",
            "abandoned": "#9E9E9E",
            "pending":   "#FBC02D",
        }[exp["status"]]
        nodes.append({
            "id": exp["id"],
            "label": exp["id"],
            "title": f'{exp["title"]} ({exp["chapter"]})',
            "group": "experiment",
            "color": {"background": status_color, "border": "#263238"},
            "shape": "box",
            "font": {"color": "white", "size": 16, "face": "Malgun Gothic"},
            "size": 30,
        })

    # 노트 노드
    for slug, (title, bundle, role, _related) in NOTES_META.items():
        nodes.append({
            "id": slug,
            "label": title.split(". ", 1)[-1] if ". " in title else title,
            "title": f"{title} — {role}",
            "group": f"bundle{bundle}",
            "color": {"background": BUNDLES[bundle]["color"], "border": "#263238"},
            "shape": "dot",
            "font": {"size": 13, "face": "Malgun Gothic"},
            "size": 16,
            "bundle": bundle,
        })

    # 엣지: 노트 → 실험
    for slug, (_t, _b, _r, related_exp) in NOTES_META.items():
        for exp_ref in related_exp:
            # exp_ref 가 실험 ID 인 경우만 엣지 생성 (§ 챕터 참조는 그래프 외)
            if exp_ref in {e["id"] for e in EXPERIMENTS}:
                edges.append({
                    "from": slug,
                    "to": exp_ref,
                    "arrows": "to",
                    "color": {"color": "#90A4AE", "opacity": 0.6},
                    "smooth": {"type": "curvedCW", "roundness": 0.1},
                })

    return nodes, edges


# ------------------------------------------------------------------
# 9. HTML 빌드
# ------------------------------------------------------------------

def build_html() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    notes_dir = repo_root / "docs" / "study" / "rl_finance"
    note_bodies = load_note_bodies(notes_dir)

    nodes, edges = build_graph_data()

    payload = {
        "bundles": BUNDLES,
        "notes_meta": {
            slug: {"title": t, "bundle": b, "role": r, "related_exp": e}
            for slug, (t, b, r, e) in NOTES_META.items()
        },
        "note_bodies": note_bodies,
        "experiments": EXPERIMENTS,
        "rq_timeline": RQ_TIMELINE,
        "hypotheses": HYPOTHESES,
        "external_refs": EXTERNAL_REFS,
        "graph": {"nodes": nodes, "edges": edges},
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    payload_json = json.dumps(payload, ensure_ascii=False)
    # </script> 충돌 방지
    payload_json = payload_json.replace("</", "<\\/")

    html = HTML_TEMPLATE.replace("__PAYLOAD__", payload_json)
    return html


# ------------------------------------------------------------------
# 10. HTML 템플릿
# ------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC RL 그리드 트레이딩 — 중간 보고서 (자료 조사 ↔ 실험 설계 지식 지도)</title>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@11.1.0/marked.min.js"></script>
<style>
  :root {
    --blue-dark:   #1565C0;
    --blue-mid:    #1976D2;
    --blue-light:  #E3F2FD;
    --orange-dark: #E65100;
    --orange-light:#FFF3E0;
    --green-dark:  #2E7D32;
    --green-light: #E8F5E9;
    --red-dark:    #C62828;
    --red-light:   #FCE4EC;
    --purple-dark: #6A1B9A;
    --purple-light:#F3E5F5;
    --yellow-dark: #F57F17;
    --gray-bg:     #F5F7FA;
    --gray-mid:    #90A4AE;
    --gray-text:   #37474F;
    --gray-soft:   #ECEFF1;
    --card-radius: 12px;
    --shadow:      0 2px 12px rgba(0,0,0,0.08);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
    background: var(--gray-bg);
    color: var(--gray-text);
    font-size: 15px;
    line-height: 1.6;
  }
  a { color: var(--blue-dark); text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* 헤더 */
  .site-header {
    background: linear-gradient(135deg, #0D47A1 0%, #1565C0 60%, #1976D2 100%);
    color: white;
    padding: 32px 48px;
  }
  .site-header h1 { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 6px; }
  .site-header .subtitle { font-size: 0.95rem; opacity: 0.9; margin-bottom: 12px; }
  .site-header .badges { display: flex; gap: 8px; flex-wrap: wrap; }
  .badge {
    background: rgba(255,255,255,0.18);
    padding: 4px 12px; border-radius: 14px;
    font-size: 0.78rem;
  }
  .badge-interim { background: #FFB300; color: #1A1A1A; font-weight: 600; }

  /* 컨테이너 */
  .container { max-width: 1400px; margin: 0 auto; padding: 32px 24px; }

  /* 섹션 */
  section { margin-bottom: 48px; }
  .section-title {
    font-size: 1.3rem; font-weight: 700; margin-bottom: 6px;
    border-left: 4px solid var(--blue-mid); padding-left: 12px;
  }
  .section-desc { color: #607D8B; font-size: 0.9rem; margin-bottom: 20px; padding-left: 16px; }

  /* RQ 타임라인 */
  .timeline { position: relative; padding-left: 30px; border-left: 3px solid var(--gray-mid); }
  .timeline-item { position: relative; margin-bottom: 22px; }
  .timeline-item::before {
    content: ''; position: absolute; left: -38px; top: 6px;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--gray-mid); border: 3px solid white; box-shadow: 0 0 0 2px var(--gray-mid);
  }
  .timeline-item.type-initial::before { background: var(--gray-mid); box-shadow: 0 0 0 2px var(--gray-mid); }
  .timeline-item.type-pivot::before { background: var(--red-dark); box-shadow: 0 0 0 2px var(--red-dark); }
  .timeline-item.type-fix::before { background: var(--yellow-dark); box-shadow: 0 0 0 2px var(--yellow-dark); }
  .timeline-item.type-discovery::before { background: var(--green-dark); box-shadow: 0 0 0 2px var(--green-dark); }
  .timeline-item.type-refine::before { background: var(--purple-dark); box-shadow: 0 0 0 2px var(--purple-dark); }
  .timeline-item.type-current::before { background: var(--blue-dark); box-shadow: 0 0 0 2px var(--blue-dark); animation: pulse 1.8s infinite; }
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.3); }
    100% { transform: scale(1); }
  }
  .timeline-date { font-size: 0.82rem; color: #607D8B; font-weight: 600; }
  .timeline-phase {
    display: inline-block; background: var(--gray-soft); padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; margin-left: 8px;
  }
  .timeline-title { font-size: 1.05rem; font-weight: 700; margin: 4px 0 6px 0; }
  .timeline-rq { background: var(--blue-light); padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 0.92rem; }
  .timeline-evidence, .timeline-decision { font-size: 0.88rem; margin-bottom: 3px; }
  .timeline-evidence::before { content: '근거: '; font-weight: 600; color: var(--green-dark); }
  .timeline-decision::before { content: '결정: '; font-weight: 600; color: var(--red-dark); }

  /* 그래프 컨테이너 */
  .graph-wrap {
    background: white; border-radius: var(--card-radius); box-shadow: var(--shadow);
    padding: 16px;
  }
  .graph-controls {
    display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;
    padding: 10px; background: var(--gray-soft); border-radius: 6px;
  }
  .filter-btn {
    background: white; border: 1.5px solid #B0BEC5; padding: 6px 12px;
    border-radius: 16px; cursor: pointer; font-size: 0.82rem;
    transition: all 0.15s;
  }
  .filter-btn:hover { background: var(--blue-light); }
  .filter-btn.active { background: var(--blue-mid); color: white; border-color: var(--blue-mid); }
  .filter-btn .swatch {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 5px; vertical-align: middle;
  }
  #network { height: 580px; border: 1px solid var(--gray-soft); border-radius: 6px; background: #FAFBFC; }
  .legend {
    margin-top: 10px; padding: 10px; background: var(--gray-soft); border-radius: 6px;
    font-size: 0.82rem; display: flex; gap: 16px; flex-wrap: wrap;
  }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-shape {
    width: 14px; height: 14px; border-radius: 50%; border: 2px solid #263238;
  }
  .legend-shape.box { border-radius: 3px; }

  /* 실험 카드 */
  .exp-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 16px;
  }
  .exp-card {
    background: white; border-radius: var(--card-radius); box-shadow: var(--shadow);
    padding: 16px; border-top: 4px solid var(--gray-mid);
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .exp-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
  .exp-card.status-completed { border-top-color: var(--green-dark); }
  .exp-card.status-abandoned { border-top-color: var(--gray-mid); opacity: 0.85; }
  .exp-card.status-pending { border-top-color: var(--yellow-dark); }
  .exp-card-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
  .exp-card-id { font-size: 1.05rem; font-weight: 700; color: var(--blue-dark); }
  .exp-card-status {
    font-size: 0.72rem; padding: 2px 8px; border-radius: 4px;
    background: var(--gray-soft); color: var(--gray-text);
  }
  .exp-card.status-completed .exp-card-status { background: var(--green-light); color: var(--green-dark); }
  .exp-card.status-pending .exp-card-status { background: #FFF8E1; color: var(--yellow-dark); }
  .exp-card-title { font-size: 0.98rem; font-weight: 600; margin-bottom: 4px; }
  .exp-card-chapter { font-size: 0.8rem; color: #607D8B; margin-bottom: 10px; }
  .exp-card dl { font-size: 0.85rem; margin-bottom: 8px; }
  .exp-card dt { font-weight: 600; color: var(--blue-dark); margin-top: 6px; }
  .exp-card dd { margin-left: 0; }
  .exp-card .note-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
  .note-chip {
    font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;
    background: var(--gray-soft); cursor: pointer;
    transition: background 0.15s;
  }
  .note-chip:hover { background: var(--blue-light); }

  /* 가설 매트릭스 */
  .hyp-table { width: 100%; border-collapse: collapse; background: white; border-radius: var(--card-radius); overflow: hidden; box-shadow: var(--shadow); }
  .hyp-table th, .hyp-table td { padding: 10px 12px; text-align: left; font-size: 0.88rem; border-bottom: 1px solid var(--gray-soft); vertical-align: top; }
  .hyp-table th { background: var(--gray-soft); font-weight: 600; }
  .hyp-table tr:last-child td { border-bottom: none; }
  .verdict-support { color: var(--green-dark); font-weight: 600; }
  .verdict-partial { color: var(--yellow-dark); font-weight: 600; }
  .verdict-reject { color: var(--red-dark); font-weight: 600; }
  .verdict-pending { color: var(--gray-mid); font-style: italic; }

  /* 외부 참조 */
  .ext-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 10px;
  }
  .ext-card {
    background: white; padding: 10px 14px; border-radius: 8px; box-shadow: var(--shadow);
    border-left: 3px solid var(--purple-dark);
  }
  .ext-title { font-weight: 600; font-size: 0.92rem; margin-bottom: 3px; }
  .ext-desc { font-size: 0.82rem; color: #607D8B; }

  /* 노트 모달 */
  .modal-overlay {
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5); z-index: 1000; padding: 40px;
  }
  .modal-overlay.active { display: flex; align-items: center; justify-content: center; }
  .modal {
    background: white; border-radius: var(--card-radius); width: 100%; max-width: 900px;
    max-height: 88vh; overflow: hidden; display: flex; flex-direction: column;
  }
  .modal-header {
    padding: 14px 20px; background: var(--blue-mid); color: white;
    display: flex; justify-content: space-between; align-items: center;
  }
  .modal-close {
    background: none; border: none; color: white; font-size: 1.4rem; cursor: pointer;
  }
  .modal-body { padding: 20px 26px; overflow-y: auto; font-size: 0.92rem; }
  .modal-body h1 { font-size: 1.4rem; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid var(--gray-soft); }
  .modal-body h2 { font-size: 1.15rem; margin-top: 18px; margin-bottom: 8px; color: var(--blue-dark); }
  .modal-body h3 { font-size: 1.0rem; margin-top: 14px; margin-bottom: 6px; }
  .modal-body p { margin-bottom: 8px; }
  .modal-body ul, .modal-body ol { margin-left: 24px; margin-bottom: 8px; }
  .modal-body li { margin-bottom: 3px; }
  .modal-body pre {
    background: #263238; color: #ECEFF1; padding: 12px; border-radius: 6px;
    overflow-x: auto; font-size: 0.85rem; margin: 8px 0;
  }
  .modal-body code {
    background: var(--gray-soft); padding: 1px 5px; border-radius: 3px; font-size: 0.88rem;
  }
  .modal-body pre code { background: none; padding: 0; color: inherit; }
  .modal-body table { border-collapse: collapse; margin: 8px 0; }
  .modal-body th, .modal-body td { border: 1px solid var(--gray-soft); padding: 6px 10px; font-size: 0.85rem; }
  .modal-body th { background: var(--gray-soft); }
  .modal-body blockquote {
    border-left: 4px solid var(--blue-mid); padding: 6px 14px; margin: 8px 0;
    background: var(--blue-light); border-radius: 0 4px 4px 0;
  }
  .modal-body a { color: var(--blue-dark); }

  /* 푸터 */
  .site-footer {
    text-align: center; padding: 24px; color: #607D8B; font-size: 0.82rem;
    background: var(--gray-soft);
  }

  /* 반응형 */
  @media (max-width: 768px) {
    .site-header { padding: 24px 20px; }
    .container { padding: 20px 14px; }
    #network { height: 420px; }
  }
</style>
</head>
<body>

<header class="site-header">
  <div class="badges">
    <span class="badge badge-interim">중간 보고서 (Interim)</span>
    <span class="badge">Phase 3 진행</span>
    <span class="badge">Build: __BUILT_AT__</span>
  </div>
  <h1 style="margin-top:14px;">BTC 그리드 트레이딩 RL — 자료 조사 ↔ 실험 설계 지식 지도</h1>
  <p class="subtitle">RQ 가 어떻게 진화했고, 27개 학습 노트가 exp030~035 실험에 어떻게 연결되는가</p>
</header>

<div class="container">

  <!-- 1. 한 줄 요약 / 어떻게 보는가 -->
  <section>
    <h2 class="section-title">이 보고서를 보는 방법</h2>
    <p class="section-desc">
      각 섹션은 독립적으로 읽을 수 있고, 노트 이름 (예: <code>differential_sharpe_moody2001</code>) 이나
      <span class="note-chip" style="display:inline-block;">노트 칩</span> 을 클릭하면 모달에서 풀텍스트가 열립니다.
      그래프 노드도 클릭 가능합니다.
    </p>
    <div style="background:white; padding:14px 18px; border-radius:8px; box-shadow:var(--shadow); font-size:0.9rem;">
      <strong>핵심 RQ (현재):</strong>
      <em>"BTC 그리드 트레이딩에서 reward 설계가 RL 정책의 알파에 어떤 영향을 미치는가?
      특히, 어떤 reward 함수 하에서 RL 이 ATR 규칙 기반을 초과하며 (혹은 초과하지 못하며), 그 메커니즘은 무엇인가?"</em>
      <br><br>
      <strong>현재 위치:</strong> Phase 3 진행 중. exp032b → 시나리오 D (Pareto frontier) 확정. exp032c (mechanism) + exp033 (slippage) 완료. 다음 exp034 (CPCV+DSR).
    </div>
  </section>

  <!-- 2. RQ 진화 타임라인 -->
  <section>
    <h2 class="section-title">1. RQ 진화 — 왜, 어떻게 바꿨나</h2>
    <p class="section-desc">
      자료 조사와 실험 결과가 어떻게 RQ 를 다시 정의하게 만들었는지의 시간순 기록.
      <span style="color:var(--red-dark); font-weight:600;">● Pivot</span> /
      <span style="color:var(--green-dark); font-weight:600;">● Discovery</span> /
      <span style="color:var(--yellow-dark); font-weight:600;">● Fix</span> /
      <span style="color:var(--purple-dark); font-weight:600;">● Refine</span> /
      <span style="color:var(--blue-dark); font-weight:600;">● Current</span>
    </p>
    <div class="timeline" id="rq-timeline"></div>
  </section>

  <!-- 3. 인터랙티브 그래프 -->
  <section>
    <h2 class="section-title">2. 인터랙티브 지식 지도</h2>
    <p class="section-desc">
      27개 학습 노트 (원형) ↔ 8개 실험 (사각형) 의 의존 관계. Bundle 필터로 카테고리별 강조 가능.
      노드를 클릭하면 노트 본문 또는 실험 상세가 열립니다.
    </p>
    <div class="graph-wrap">
      <div class="graph-controls" id="graph-filters"></div>
      <div id="network"></div>
      <div class="legend">
        <div class="legend-item"><div class="legend-shape box" style="background:#388E3C;"></div> 완료 실험</div>
        <div class="legend-item"><div class="legend-shape box" style="background:#FBC02D;"></div> 대기 실험</div>
        <div class="legend-item"><div class="legend-shape box" style="background:#9E9E9E;"></div> 폐기 실험</div>
        <div class="legend-item"><div class="legend-shape" style="background:#1976D2;"></div> A 직결</div>
        <div class="legend-item"><div class="legend-shape" style="background:#7B1FA2;"></div> B 이론</div>
        <div class="legend-item"><div class="legend-shape" style="background:#388E3C;"></div> C 그리드뿌리</div>
        <div class="legend-item"><div class="legend-shape" style="background:#E65100;"></div> D 보강</div>
        <div class="legend-item"><div class="legend-shape" style="background:#C62828;"></div> E 정합성</div>
      </div>
    </div>
  </section>

  <!-- 4. 실험 카드 -->
  <section>
    <h2 class="section-title">3. 실험 설계 카드 (exp030~035)</h2>
    <p class="section-desc">
      각 카드는 "무엇을 / 어떻게 / 무엇이 나왔는가 / 어떤 자료 기반인가" 의 4-tuple. 노트 칩 클릭 시 본문 열림.
    </p>
    <div class="exp-grid" id="exp-cards"></div>
  </section>

  <!-- 5. 가설 매트릭스 -->
  <section>
    <h2 class="section-title">4. 가설 × 실험 매트릭스</h2>
    <p class="section-desc">H1~H4 가설이 어느 실험에서 어떻게 (부분) 지지/부정되었는가.</p>
    <table class="hyp-table" id="hyp-table"></table>
  </section>

  <!-- 6. Bundle 분류 (자료 카테고리) -->
  <section>
    <h2 class="section-title">5. 자료 분류 — 무엇이 직결, 무엇이 배경인가</h2>
    <p class="section-desc">
      27개 노트의 5개 Bundle 분류. 노트 클릭 시 본문 열림.
    </p>
    <div id="bundle-sections"></div>
  </section>

  <!-- 7. 외부 참조 -->
  <section>
    <h2 class="section-title">6. 외부 참조 — 배경 지식 (실험 설계에 직접 미사용)</h2>
    <p class="section-desc">
      퀀트 투자 일반, López de Prado · Ernie Chan 등 책. 본 실험 설계에 직접 들어가진 않았으나 토대 지식.
    </p>
    <div class="ext-grid" id="ext-cards"></div>
  </section>

  <!-- 8. 남은 작업 (중간 보고서 특화) -->
  <section>
    <h2 class="section-title">7. 남은 작업 + 미해결 질문</h2>
    <p class="section-desc">중간 보고서 시점에서 명시적으로 짚어둔 오픈 이슈.</p>
    <div style="background:white; padding:16px 22px; border-radius:8px; box-shadow:var(--shadow); font-size:0.9rem;">
      <p style="font-weight:600; margin-bottom:8px; color:var(--blue-dark);">다음 실험 (시간순)</p>
      <ol style="margin-left: 22px; margin-bottom: 14px;">
        <li><strong>exp034 — CPCV 6-fold + DSR</strong>: Train+Val 안에서 path 분포로 다중검정 보정된 진짜 알파 검증. (§7.2)</li>
        <li><strong>exp035 — Test set 봉인 해제</strong>: CPCV 통과 후 1회 평가. (§7.3 최종)</li>
        <li>논문 본문 작성 병행.</li>
      </ol>
      <p style="font-weight:600; margin-bottom:8px; color:var(--red-dark);">미해결 / 오픈 질문</p>
      <ul style="margin-left: 22px;">
        <li>exp027_rl 의 Env-v3 사전 증거 (Test Sharpe 1.955, ATR +109%) 가 Env-v4 에서는 +12% 로 약화. <strong>환경 의존성 효과가 reward variant 효과보다 큰 것</strong> 의 §8 Discussion 정직 인정.</li>
        <li>H2-strong (variant &gt; sym) 의 부분 부정 — exp032b 에서 sym 이 Sharpe 1위. dsr 만 sym 과 동등. asym/pt 는 sym 보다 낮음. 단일 winner 없는 결과의 thesis statement 정교화 필요.</li>
        <li>exp031 BC warm-start 의 학습 정체 — 정석 BC pretrain 또는 SAC 전환은 future work.</li>
        <li>호가창 모델링 부재. 본 논문 §7 한계 명시.</li>
      </ul>
    </div>
  </section>

</div>

<footer class="site-footer">
  중간 보고서 자료 · 단일 HTML self-contained · 노트 본문 27개 인라인 임베드 · vis-network + marked.js (CDN)
</footer>

<!-- 노트 모달 -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal">
    <div class="modal-header">
      <span id="modal-title">노트 제목</span>
      <button class="modal-close" id="modal-close" aria-label="닫기">✕</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;

// ----- RQ 타임라인 -----
const tl = document.getElementById('rq-timeline');
DATA.rq_timeline.forEach(item => {
  const div = document.createElement('div');
  div.className = `timeline-item type-${item.type}`;
  div.innerHTML = `
    <span class="timeline-date">${item.date}</span>
    <span class="timeline-phase">${item.phase}</span>
    <div class="timeline-title">${item.title}</div>
    <div class="timeline-rq">${item.rq}</div>
    ${item.evidence !== '—' ? `<div class="timeline-evidence">${item.evidence}</div>` : ''}
    ${item.decision !== '—' ? `<div class="timeline-decision">${item.decision}</div>` : ''}
  `;
  tl.appendChild(div);
});

// ----- 실험 카드 -----
const expGrid = document.getElementById('exp-cards');
DATA.experiments.forEach(exp => {
  const noteChips = exp.notes.map(slug => {
    const meta = DATA.notes_meta[slug];
    const label = meta ? meta.title : slug;
    return `<span class="note-chip" data-note="${slug}">${label}</span>`;
  }).join('');
  const card = document.createElement('div');
  card.className = `exp-card status-${exp.status}`;
  card.innerHTML = `
    <div class="exp-card-header">
      <span class="exp-card-id">${exp.id}</span>
      <span class="exp-card-status">${exp.status_label}</span>
    </div>
    <div class="exp-card-title">${exp.title}</div>
    <div class="exp-card-chapter">${exp.chapter}</div>
    <dl>
      <dt>Objective</dt><dd>${exp.objective}</dd>
      <dt>Design</dt><dd>${exp.design}</dd>
      <dt>Result</dt><dd>${exp.result}</dd>
      <dt>Verdict</dt><dd>${exp.verdict}</dd>
    </dl>
    <div class="note-chips"><strong style="font-size:0.78rem; align-self:center; margin-right:4px;">참조 노트:</strong>${noteChips}</div>
  `;
  expGrid.appendChild(card);
});

// ----- 가설 매트릭스 -----
const hypTable = document.getElementById('hyp-table');
const allExpKeys = new Set();
DATA.hypotheses.forEach(h => Object.keys(h.results).forEach(k => allExpKeys.add(k)));
const expKeys = Array.from(allExpKeys);
let thead = '<tr><th style="width:80px;">가설</th><th>내용</th>';
expKeys.forEach(k => thead += `<th>${k}</th>`);
thead += '</tr>';
let tbody = '';
DATA.hypotheses.forEach(h => {
  let row = `<tr><td><strong>${h.id}</strong></td><td>${h.text}</td>`;
  expKeys.forEach(k => {
    const v = h.results[k] || '—';
    let cls = '';
    if (v.includes('지지') && !v.includes('부정')) cls = 'verdict-support';
    if (v.includes('부분')) cls = 'verdict-partial';
    if (v.includes('부정') && !v.includes('부분')) cls = 'verdict-reject';
    if (v.includes('검증 예정')) cls = 'verdict-pending';
    row += `<td class="${cls}">${v}</td>`;
  });
  row += '</tr>';
  tbody += row;
});
hypTable.innerHTML = thead + tbody;

// ----- Bundle 섹션 (자료 분류) -----
const bundleHost = document.getElementById('bundle-sections');
Object.entries(DATA.bundles).forEach(([key, b]) => {
  const slugs = Object.keys(DATA.notes_meta).filter(s => DATA.notes_meta[s].bundle === key);
  const block = document.createElement('div');
  block.style.marginBottom = '18px';
  let chips = '';
  slugs.forEach(slug => {
    const m = DATA.notes_meta[slug];
    chips += `
      <div class="note-chip" data-note="${slug}"
           style="display:inline-block; padding:6px 10px; margin:3px; background:${b.color}1A; border-left:3px solid ${b.color}; border-radius:4px; font-size:0.85rem;">
        <strong>${m.title}</strong> — <span style="color:#607D8B;">${m.role}</span>
      </div>`;
  });
  block.innerHTML = `
    <h3 style="font-size:1.05rem; margin-bottom:6px; color:${b.color};">${b.name} <span style="font-size:0.78rem; color:#607D8B; font-weight:400;">(${slugs.length}개)</span></h3>
    <p style="font-size:0.85rem; color:#607D8B; margin-bottom:8px;">${b.desc}</p>
    <div>${chips}</div>
  `;
  bundleHost.appendChild(block);
});

// ----- 외부 참조 -----
const extHost = document.getElementById('ext-cards');
DATA.external_refs.forEach(([title, desc]) => {
  const c = document.createElement('div');
  c.className = 'ext-card';
  c.innerHTML = `<div class="ext-title">${title}</div><div class="ext-desc">${desc}</div>`;
  extHost.appendChild(c);
});

// ----- 그래프 필터 버튼 -----
const filterHost = document.getElementById('graph-filters');
const filters = [
  {id: 'all', label: '전체', color: '#90A4AE'},
  {id: 'exp', label: '실험만', color: '#388E3C'},
  ...Object.entries(DATA.bundles).map(([k, b]) => ({id: `bundle${k}`, label: b.name.split('. ')[0] + ' ' + b.name.split('. ')[1].split(' ')[0], color: b.color})),
];
filters.forEach(f => {
  const btn = document.createElement('button');
  btn.className = 'filter-btn' + (f.id === 'all' ? ' active' : '');
  btn.dataset.filter = f.id;
  btn.innerHTML = `<span class="swatch" style="background:${f.color};"></span>${f.label}`;
  filterHost.appendChild(btn);
});

// ----- vis-network 그래프 -----
const nodes = new vis.DataSet(DATA.graph.nodes);
const edges = new vis.DataSet(DATA.graph.edges);
const network = new vis.Network(
  document.getElementById('network'),
  { nodes, edges },
  {
    physics: {
      enabled: true,
      barnesHut: { gravitationalConstant: -8000, springLength: 130, springConstant: 0.04 },
      stabilization: { iterations: 250 },
    },
    interaction: { hover: true, tooltipDelay: 150 },
    nodes: { borderWidth: 2 },
    edges: { width: 1.2 },
  }
);

network.on('click', (params) => {
  if (params.nodes.length === 0) return;
  const id = params.nodes[0];
  if (DATA.note_bodies[id]) openNote(id);
  else {
    const exp = DATA.experiments.find(e => e.id === id);
    if (exp) openExperiment(exp);
  }
});

// 필터
filterHost.addEventListener('click', (e) => {
  const btn = e.target.closest('.filter-btn');
  if (!btn) return;
  filterHost.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const f = btn.dataset.filter;
  const update = DATA.graph.nodes.map(n => {
    let hidden = false;
    if (f === 'all') hidden = false;
    else if (f === 'exp') hidden = n.group !== 'experiment';
    else hidden = n.group !== f && n.group !== 'experiment';
    return { id: n.id, hidden };
  });
  nodes.update(update);
});

// ----- 모달 -----
const overlay = document.getElementById('modal-overlay');
const modalTitle = document.getElementById('modal-title');
const modalBody = document.getElementById('modal-body');
document.getElementById('modal-close').onclick = () => overlay.classList.remove('active');
overlay.onclick = (e) => { if (e.target === overlay) overlay.classList.remove('active'); };

function openNote(slug) {
  const meta = DATA.notes_meta[slug];
  const body = DATA.note_bodies[slug] || '(노트 없음)';
  modalTitle.textContent = meta ? meta.title : slug;
  modalBody.innerHTML = marked.parse(body);
  overlay.classList.add('active');
  modalBody.scrollTop = 0;
}

function openExperiment(exp) {
  modalTitle.textContent = `${exp.id} — ${exp.title}`;
  const notesHtml = exp.notes.map(s => {
    const m = DATA.notes_meta[s];
    return m ? `<li><a href="#" data-note="${s}">${m.title}</a> — ${m.role}</li>` : `<li>${s}</li>`;
  }).join('');
  modalBody.innerHTML = `
    <h2>${exp.chapter}</h2>
    <p><strong>Status:</strong> ${exp.status_label}</p>
    <h3>Objective</h3><p>${exp.objective}</p>
    <h3>Design</h3><p>${exp.design}</p>
    <h3>Result</h3><p>${exp.result}</p>
    <h3>Verdict</h3><p>${exp.verdict}</p>
    <h3>참조 노트</h3><ul>${notesHtml}</ul>
  `;
  overlay.classList.add('active');
  modalBody.scrollTop = 0;
}

// 위임: note chip / 모달 안 link 클릭
document.addEventListener('click', (e) => {
  const t = e.target.closest('[data-note]');
  if (t) {
    e.preventDefault();
    openNote(t.dataset.note);
  }
});

// ESC 키로 모달 닫기
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') overlay.classList.remove('active');
});
</script>
</body>
</html>
"""


# ------------------------------------------------------------------
# 11. main
# ------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "reports" / "knowledge_map.html"

    html = build_html()
    html = html.replace("__BUILT_AT__", datetime.now().strftime("%Y-%m-%d %H:%M"))

    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"[ok] {out_path}  ({size_kb:.1f} KB)")
    print(f"     notes embedded: {len(NOTES_META)}")
    print(f"     experiments:    {len(EXPERIMENTS)}")
    print(f"     RQ timeline:    {len(RQ_TIMELINE)} events")


if __name__ == "__main__":
    main()
