"""
scripts/make_project_flow.py

프로젝트 전체 흐름 시각화 — 발표 준비용
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 색상 팔레트 ───────────────────────────────────────────────
C_PHASE   = "#1565C0"   # 진파랑 — 단계 제목
C_DID     = "#E3F2FD"   # 연파랑 — 한 일
C_DID_BD  = "#1976D2"   # 파랑 테두리
C_PROB    = "#FFF3E0"   # 연주황 — 문제
C_PROB_BD = "#E65100"   # 주황 테두리
C_FIX     = "#E8F5E9"   # 연초록 — 해결
C_FIX_BD  = "#2E7D32"   # 초록 테두리
C_RES     = "#FCE4EC"   # 연빨강 — 결과/수치
C_RES_BD  = "#C62828"   # 빨강 테두리
C_ARROW   = "#546E7A"   # 회색 화살표

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor("#FAFAFA")

# ── 레이아웃: 좌(80%) 타임라인  /  우(20%) Sharpe 그래프 ───────
ax_main  = fig.add_axes([0.01, 0.02, 0.72, 0.95])
ax_sharpe = fig.add_axes([0.76, 0.10, 0.22, 0.80])

ax_main.set_xlim(0, 10)
ax_main.set_ylim(0, 100)
ax_main.axis("off")
ax_main.set_facecolor("#FAFAFA")


# ── 헬퍼 함수 ────────────────────────────────────────────────
def box(ax, x, y, w, h, text, fc, ec, fs=9.5, bold=False, wrap=True):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center", fontsize=fs,
            fontweight=weight, color="#1A1A1A",
            wrap=True,
            multialignment="center",
            transform=ax.transData)

def phase_header(ax, y, label, sublabel=""):
    rect = FancyBboxPatch((0.15, y), 9.7, 1.1,
                          boxstyle="round,pad=0.2",
                          facecolor=C_PHASE, edgecolor=C_PHASE, linewidth=0)
    ax.add_patch(rect)
    ax.text(0.55, y + 0.7, label, ha="left", va="center",
            fontsize=13, fontweight="bold", color="white")
    if sublabel:
        ax.text(9.5, y + 0.7, sublabel, ha="right", va="center",
                fontsize=10, color="#B3E5FC")

def arrow_down(ax, x, y_top, y_bot):
    ax.annotate("", xy=(x, y_bot + 0.05), xytext=(x, y_top - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.8, mutation_scale=14))

def tag(ax, x, y, text, color):
    ax.text(x, y, text, ha="left", va="center", fontsize=8.5,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=color,
                      edgecolor="none"))

# ── 수직 타임라인 선 ──────────────────────────────────────────
ax_main.plot([0.85, 0.85], [1, 98], color="#B0BEC5", lw=2.5, zorder=0)

# ─────────────────────────────────────────────────────────────
# PHASE 1 — 프로젝트 설계 & 아이디어
# ─────────────────────────────────────────────────────────────
y = 93
ax_main.plot(0.85, y+0.55, "o", ms=14, color=C_PHASE, zorder=5)
ax_main.text(1.0, y+0.55, "Phase 1", fontsize=9, color=C_PHASE,
             va="center", fontweight="bold")
phase_header(ax_main, y - 0.4, "① 프로젝트 설계  —  비방향성 그리드 트레이딩", "Week 1~2")

box(ax_main, 1.0, y-3.5, 8.5, 2.8,
    "• 핵심 아이디어: 가격 방향 예측 없이 변동성 자체에서 수익\n"
    "• Action 공간: 연속 2D  (aggressiveness, profit_target) ∈ [0,1]²\n"
    "• State 5차원: log_price · divergence · holdings_ratio · cash_ratio · volatility\n"
    "• ATR 비례 주문 공식 설계: gap = ATR/price × (A + action × B)\n"
    "• Sell 우선 원칙 · n_splits 자본 슬롯 · 사이클 수익 추적",
    C_DID, C_DID_BD, fs=9.5)

# ─────────────────────────────────────────────────────────────
# PHASE 2 — 데이터 & 환경 구현
# ─────────────────────────────────────────────────────────────
y = 86
ax_main.plot(0.85, y+0.55, "o", ms=14, color=C_PHASE, zorder=5)
ax_main.text(1.0, y+0.55, "Phase 2", fontsize=9, color=C_PHASE,
             va="center", fontweight="bold")
phase_header(ax_main, y - 0.4, "② 데이터 수집 & 환경 구현", "Week 2~3")

box(ax_main, 1.0, y-2.6, 4.0, 2.2,
    "한 일\n"
    "• ccxt Binance API → BTC/USDT 1h\n"
    "  (54,933 캔들, 2020~2026)\n"
    "• ATR·log_price·rolling z-score 전처리\n"
    "• trading_env.py 전체 구현\n"
    "• gymnasium env_checker 통과",
    C_DID, C_DID_BD, fs=9)

box(ax_main, 5.2, y-2.6, 4.3, 2.2,
    "Train / Val / Test 분할\n\n"
    "Train  2020~2022  (25,916봉)\n"
    "Val    2023       ( 8,736봉)\n"
    "Test   2024~2026  (19,901봉)  ← 봉인",
    C_FIX, C_FIX_BD, fs=9.5)

# ─────────────────────────────────────────────────────────────
# PHASE 3 — 초기 학습 & 버그들
# ─────────────────────────────────────────────────────────────
y = 80
ax_main.plot(0.85, y+0.55, "o", ms=14, color="#E65100", zorder=5)
ax_main.text(1.0, y+0.55, "Phase 3", fontsize=9, color="#E65100",
             va="center", fontweight="bold")
phase_header(ax_main, y - 0.4, "③ 초기 학습 (exp001~006)  —  버그와의 전쟁", "Week 3~5")

box(ax_main, 1.0, y-2.4, 3.8, 2.0,
    "문제 ①  수수료 이중계산 버그\n\n"
    "reward에서 fee를 별도 차감했으나\n"
    "이미 cash 잔액에 반영되어 있었음\n"
    "→ 모든 거래 = 손해로 인식\n"
    "→ 에이전트: 거래 0건으로 수렴",
    C_PROB, C_PROB_BD, fs=9)

box(ax_main, 5.0, y-2.4, 4.5, 2.0,
    "문제 ② + ③  eval 파이프라인 버그\n\n"
    "• random_start=True가 eval 환경에 상속\n"
    "  → 평가 시작점 랜덤 → Sharpe 불안정\n"
    "• DummyVecEnv auto-reset으로\n"
    "  에피소드 종료 후 n_trades=0 소실",
    C_PROB, C_PROB_BD, fs=9)

box(ax_main, 1.0, y-4.2, 8.5, 1.5,
    "해결:  reward 수수료 항 제거  |  eval 환경 random_start=False 강제  |  DummyVecEnv → 단일 env 직접 실행",
    C_FIX, C_FIX_BD, fs=9.5)

# ─────────────────────────────────────────────────────────────
# PHASE 4 — 파라미터 탐색
# ─────────────────────────────────────────────────────────────
y = 72
ax_main.plot(0.85, y+0.55, "o", ms=14, color=C_PHASE, zorder=5)
ax_main.text(1.0, y+0.55, "Phase 4", fontsize=9, color=C_PHASE,
             va="center", fontweight="bold")
phase_header(ax_main, y - 0.4, "④ 체계적 파라미터 탐색 (exp007~015)", "Week 5~6")

box(ax_main, 1.0, y-1.8, 2.6, 1.4,
    "exp007\nVecNormalize 제거\n→ Sharpe 14.39",
    C_DID, C_DID_BD, fs=9)

box(ax_main, 3.8, y-1.8, 2.5, 1.4,
    "exp008\nent_coef 0.05\n→ Sharpe 16.38",
    C_DID, C_DID_BD, fs=9)

box(ax_main, 6.5, y-1.8, 3.0, 1.4,
    "exp009~015\nn_splits · threshold\n· n_buy_orders 탐색",
    C_DID, C_DID_BD, fs=9)

box(ax_main, 1.0, y-3.4, 8.5, 1.3,
    "최적 설정 확정  —  n_splits=2  |  threshold=avg_price  |  ent_coef=0.05   →   exp013b  Val Sharpe  17.58",
    C_FIX, C_FIX_BD, fs=10, bold=True)

# ─────────────────────────────────────────────────────────────
# PHASE 5 — Bayesian 계수 최적화
# ─────────────────────────────────────────────────────────────
y = 65
ax_main.plot(0.85, y+0.55, "o", ms=14, color=C_PHASE, zorder=5)
ax_main.text(1.0, y+0.55, "Phase 5", fontsize=9, color=C_PHASE,
             va="center", fontweight="bold")
phase_header(ax_main, y - 0.4, "⑤ Bayesian 계수 최적화 (Optuna TPE, 50 trials)", "Week 6")

box(ax_main, 1.0, y-2.5, 4.0, 2.1,
    "8개 계수 자동 탐색\n\n"
    "• 목적함수: Val Sharpe (1M step)\n"
    "• TPE sampler + MedianPruner\n"
    "• SQLite 저장 (중단/재개 가능)\n"
    "• 소요: 4.3시간 (CPU)",
    C_DID, C_DID_BD, fs=9)

box(ax_main, 5.2, y-2.5, 4.3, 2.1,
    "Trial #42  최적 계수 발견\n\n"
    "A_s=0.101  (sell-market 즉각 실행)\n"
    "C_b=5.22, D_b=18.68  (buy-lo 대폭 확장)\n"
    "C_s=6.91  (sell-cost 고기준 유지)\n"
    "→  Val Sharpe  42.997  (+145%)",
    C_RES, C_RES_BD, fs=9.5, bold=False)

box(ax_main, 1.0, y-4.0, 8.5, 1.2,
    "발견한 전략 패턴:  빠른 sell-market (0.1×ATR)  +  깊은 buy-lo (최대 23.9×ATR)  +  대형 랠리 전용 sell-cost",
    C_FIX, C_FIX_BD, fs=9.5)

# ─────────────────────────────────────────────────────────────
# PHASE 6 — Full Training exp016
# ─────────────────────────────────────────────────────────────
y = 57
ax_main.plot(0.85, y+0.55, "o", ms=14, color=C_PHASE, zorder=5)
ax_main.text(1.0, y+0.55, "Phase 6", fontsize=9, color=C_PHASE,
             va="center", fontweight="bold")
phase_header(ax_main, y - 0.4, "⑥ 최종 모델 훈련 (exp016)  —  3M 스텝 Full Training", "Week 6")

box(ax_main, 1.0, y-2.3, 4.0, 1.9,
    "설정\n\n"
    "• Trial #42 최적 계수 적용\n"
    "• 3,000,000 스텝  |  n_envs=4\n"
    "• Cosine LR: 3e-4 → 1e-5\n"
    "• eval_freq=50,000  |  seed=42",
    C_DID, C_DID_BD, fs=9)

box(ax_main, 5.2, y-2.3, 4.3, 1.9,
    "학습 수렴 패턴\n\n"
    "step  50K : Sharpe 31.9  (초기 수렴)\n"
    "step 350K~1.3M : 31.9 → 35.4  (단조 증가)\n"
    "step 1.3M~3.0M : Sharpe ≈ 35.39  (완전 수렴)\n"
    "best model @ step 2.1M : Sharpe  35.424",
    C_RES, C_RES_BD, fs=9)

# ─────────────────────────────────────────────────────────────
# PHASE 7 — Test Set 평가
# ─────────────────────────────────────────────────────────────
y = 49
ax_main.plot(0.85, y+0.55, "o", ms=14, color="#C62828", zorder=5)
ax_main.text(1.0, y+0.55, "Phase 7", fontsize=9, color="#C62828",
             va="center", fontweight="bold")
phase_header(ax_main, y - 0.4, "⑦ 봉인 해제  —  Test Set 최종 평가 (2024~2026)", "Week 7")

box(ax_main, 1.0, y-3.5, 4.0, 3.1,
    "Test Set 결과\n\n"
    "Val  Sharpe : 35.424   MDD : 2.46%\n"
    "Test Sharpe : 43.040   MDD : 3.12%\n\n"
    "완성 사이클 : 9,520개\n"
    "사이클 승률 : 96.9%\n"
    "사이클 평균 PnL : +0.182%",
    C_RES, C_RES_BD, fs=10, bold=False)

box(ax_main, 5.2, y-3.5, 4.3, 3.1,
    "베이스라인 비교\n\n"
    "Buy & Hold     :  Sharpe  0.681\n"
    "Fixed Grid 1%  :  Sharpe  0.724\n"
    "Fixed Grid 5%  :  Sharpe  1.472  ← 최강 베이스라인\n"
    "ATR Grid k=2.0 :  Sharpe  0.436\n\n"
    "PPO vs 최강 베이스라인 :  43.04 / 1.47  =  29.2×",
    C_FIX, C_FIX_BD, fs=9.5)

# ─────────────────────────────────────────────────────────────
# Sub-RQ 결론 박스
# ─────────────────────────────────────────────────────────────
y_res = 43
rect = FancyBboxPatch((0.5, y_res - 3.2), 9.3, 3.0,
                      boxstyle="round,pad=0.2",
                      facecolor="#E8EAF6", edgecolor="#3949AB", linewidth=2)
ax_main.add_patch(rect)
ax_main.text(5.15, y_res - 0.55,
             "핵심 연구 결과",
             ha="center", va="center", fontsize=13, fontweight="bold", color="#1A237E")
ax_main.text(5.15, y_res - 1.45,
             "RQ1  ─  PPO가 고정 그리드 베이스라인을 능가하는가?   →   YES  (29.2× on Test Set)",
             ha="center", va="center", fontsize=10.5, color="#1A237E")
ax_main.text(5.15, y_res - 2.20,
             "RQ2  ─  변동성 레짐에 따라 다른 간격을 선택하는가?   →   YES  (Mann-Whitney  p < 10⁻¹⁹)",
             ha="center", va="center", fontsize=10.5, color="#1A237E")
ax_main.text(5.15, y_res - 2.95,
             "과적합 없음  ─  Test Sharpe (43.04)  >  Val Sharpe (35.42)",
             ha="center", va="center", fontsize=10.5, color="#1A237E")

# ─────────────────────────────────────────────────────────────
# 화살표 (단계 연결)
# ─────────────────────────────────────────────────────────────
for y_arrow in [93.5, 87.0, 80.8, 72.8, 65.8, 57.8, 50.5]:
    arrow_down(ax_main, 0.85, y_arrow - 0.1, y_arrow - 1.5)

# 제목
ax_main.text(5.15, 99.0,
             "BTC Dynamic Grid Trading — Project Journey",
             ha="center", va="center", fontsize=16, fontweight="bold", color="#0D47A1")
ax_main.text(5.15, 98.0,
             "PPO + ATR-Scaled Orders + Bayesian Coefficient Optimization",
             ha="center", va="center", fontsize=11, color="#37474F")

# ─────────────────────────────────────────────────────────────
# 우측: Sharpe 성장 그래프
# ─────────────────────────────────────────────────────────────
ax_sharpe.set_facecolor("#F5F5F5")
ax_sharpe.spines[["top", "right"]].set_visible(False)

milestones = [
    ("exp007\n(no VecNorm)",  14.39,  "#90CAF9"),
    ("exp008\n(ent=0.05)",    16.38,  "#64B5F6"),
    ("exp013b\n(baseline)",   17.58,  "#42A5F5"),
    ("Optuna\n#42 (1M)",      42.997, "#FF8F00"),
    ("exp016\nVal (3M)",      35.424, "#1565C0"),
    ("exp016\nTest",          43.040, "#C62828"),
]

labels  = [m[0] for m in milestones]
sharpes = [m[1] for m in milestones]
colors  = [m[2] for m in milestones]
xs      = np.arange(len(labels))

bars = ax_sharpe.bar(xs, sharpes, color=colors, edgecolor="white",
                     linewidth=1.2, width=0.65, zorder=3)

for bar, val in zip(bars, sharpes):
    ax_sharpe.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.5,
                   f"{val:.1f}",
                   ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                   color="#1A1A1A")

# 베이스라인 수평선
ax_sharpe.axhline(17.58, color="#42A5F5", linestyle="--",
                  linewidth=1.2, alpha=0.6, label="exp013b baseline")
ax_sharpe.axhline(1.472, color="#E65100", linestyle=":",
                  linewidth=1.2, alpha=0.8, label="Best fixed baseline (1.47)")

ax_sharpe.set_xticks(xs)
ax_sharpe.set_xticklabels(labels, fontsize=8.5, rotation=0)
ax_sharpe.set_ylabel("Val / Test Sharpe Ratio", fontsize=10)
ax_sharpe.set_ylim(0, 52)
ax_sharpe.set_title("Sharpe 성장 추이", fontsize=11, fontweight="bold", pad=10)
ax_sharpe.legend(fontsize=8, loc="upper left")
ax_sharpe.grid(axis="y", alpha=0.3, zorder=0)

# 단계 레이블
ax_sharpe.text(1, 48.5, "파라미터\n탐색", ha="center", fontsize=8,
               color="#1565C0", style="italic")
ax_sharpe.text(3, 48.5, "Bayesian\n최적화", ha="center", fontsize=8,
               color="#FF8F00", style="italic")
ax_sharpe.text(5, 48.5, "최종\n평가", ha="center", fontsize=8,
               color="#C62828", style="italic")

# ─────────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────────
out_path = "reports/semester1/figures/project_journey.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"저장 완료: {out_path}")
plt.show()
