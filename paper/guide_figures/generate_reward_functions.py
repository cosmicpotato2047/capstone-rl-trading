"""
Reading guide 용 보상 함수 모양 비교 그림.

4가지 reward (sym, asym, dsr, pt) 의 함수 모양을 직관적으로 보여줌.
DSR 은 단일 step 의 함수가 아니므로 1-step PnL 가정 단순화 형태로 표시.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트
rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# x: 1-step PnL ratio  (= step reward 의 입력)
x = np.linspace(-0.02, 0.02, 1001)


def r_sym(x):
    return x


def r_asym(x, beta=3.42):
    return np.where(x >= 0, x, beta * x)


def r_pt(x, alpha=0.683, lam=3.303):
    sign = np.sign(x)
    out = np.where(
        x >= 0,
        sign * np.power(np.abs(x), alpha),
        -lam * np.power(np.abs(x), alpha),
    )
    return out


def r_dsr_approx(x, eta=0.0352, n_history=500, seed=1):
    """
    DSR step reward 의 simulated 모양 — 정상상태 EWMA 가정.
    A_prev = mean of past returns, B_prev = mean of past squared returns.
    """
    rng = np.random.default_rng(seed)
    past = rng.normal(0.0, 0.005, size=n_history)  # historical step returns
    A_prev = past.mean()
    B_prev = (past ** 2).mean()
    dA = x - A_prev
    dB = x ** 2 - B_prev
    denom = (B_prev - A_prev ** 2) ** 1.5
    return (B_prev * dA - 0.5 * A_prev * dB) / max(denom, 1e-9)


# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
fig.suptitle(
    "4 가지 보상 함수의 모양 비교\n"
    "가로축 = 1 시간 수익률 (PnL ratio),  세로축 = 정책이 받는 보상 신호",
    fontsize=14, fontweight="bold",
)

# (a) Symmetric
ax = axes[0, 0]
ax.plot(x * 100, r_sym(x) * 100, color="#2E86AB", lw=2.5)
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_title("(a) Symmetric (sym)\n— 베이스라인. 이익과 손실을 같은 무게로 처리", fontsize=11)
ax.set_xlabel("수익률 (%)")
ax.set_ylabel("보상")
ax.grid(alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(-2.5, 2.5)
ax.annotate("이익 = 그대로", xy=(1.2, 1.2), fontsize=9, color="#2E86AB")
ax.annotate("손실 = 그대로", xy=(-1.9, -1.0), fontsize=9, color="#2E86AB")

# (b) Asymmetric
ax = axes[0, 1]
ax.plot(x * 100, r_asym(x) * 100, color="#A23B72", lw=2.5)
ax.plot(x * 100, r_sym(x) * 100, color="gray", lw=1, ls="--", alpha=0.5, label="sym (비교)")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_title("(b) Asymmetric (asym, β=3.42)\n— 손실을 3.4배 무겁게 처벌", fontsize=11)
ax.set_xlabel("수익률 (%)")
ax.set_ylabel("보상")
ax.grid(alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(-7, 2.5)
ax.legend(loc="lower right", fontsize=9)
ax.annotate("이익 = 그대로", xy=(0.8, 1.5), fontsize=9, color="#A23B72")
ax.annotate("손실 = 3.4 × 처벌\n→ 위험 회피 학습", xy=(-1.95, -5.5), fontsize=9, color="#A23B72")

# (c) Prospect-theoretic
ax = axes[1, 0]
ax.plot(x * 100, r_pt(x) * 100, color="#F18F01", lw=2.5)
ax.plot(x * 100, r_sym(x) * 100, color="gray", lw=1, ls="--", alpha=0.5, label="sym (비교)")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_title(
    "(c) Prospect-theoretic (pt, α=0.68, λ=3.30)\n"
    "— Kahneman-Tversky 효용함수. 손실 회피 + 오목한 효용",
    fontsize=11,
)
ax.set_xlabel("수익률 (%)")
ax.set_ylabel("보상")
ax.grid(alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(-50, 30)
ax.legend(loc="lower right", fontsize=9)
ax.annotate("이익 영역:\n오목(concave)\n→ 큰 이익보다\n작은 이익 다발 선호",
            xy=(0.4, 7), fontsize=9, color="#F18F01")
ax.annotate("손실 영역:\n매우 가파른 처벌\n+ 오목한 곡선",
            xy=(-1.95, -35), fontsize=9, color="#F18F01")

# (d) DSR (simulated)
ax = axes[1, 1]
# DSR is path-dependent — show the 정상상태 approximation
y_dsr = np.array([r_dsr_approx(xi) for xi in x])
ax.plot(x * 100, y_dsr, color="#06A77D", lw=2.5)
ax.plot(x * 100, r_sym(x) * 1000, color="gray", lw=1, ls="--", alpha=0.5, label="sym ×1000 (참고)")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_title(
    "(d) DSR (η=1/28h)\n"
    "— 1-step 수익이 아니라 \"샤프비율의 변화량\"\n"
    "  (값 자체는 과거 1.2일 평균/분산에 의존)",
    fontsize=11,
)
ax.set_xlabel("수익률 (%)")
ax.set_ylabel("보상 (정규화 단위)")
ax.grid(alpha=0.3)
ax.set_xlim(-2, 2)
ax.legend(loc="lower right", fontsize=9)
ax.annotate(
    "★ 다른 3개와 본질 다름 ★\n"
    "값이 과거 윈도우의 평균/분산에\n"
    "의존 → 시간 의존성을 학습함\n"
    "(긴 holding 을 선호)",
    xy=(-1.95, max(y_dsr) * 0.4), fontsize=9, color="#06A77D",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#06A77D"),
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(OUT_DIR, "reward_functions_shape.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
