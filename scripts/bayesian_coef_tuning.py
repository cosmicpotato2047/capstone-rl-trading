"""
Phase 2-A.5: Bayesian Optimization — MDP 공식 계수 튜닝

탐색 대상 (8개 계수):
    buy_hi_gap      = atr_ratio × (A_b + aggressiveness × B_b)
    buy_lo_gap      = atr_ratio × (C_b + aggressiveness × D_b)
    sell_market_gap = atr_ratio × (A_s + profit_target  × B_s)
    sell_cost_gap   = atr_ratio × (C_s + profit_target  × D_s)

기본값 (설계값): A=0.5, B=1.5, C=2.5, D=7.5 (buy & sell 동일)

설정:
    - Sampler : Optuna TPE (Tree-structured Parzen Estimator)
    - Pruner  : MedianPruner (하위 50% 조기 중단, 10회 warm-up)
    - 목적함수: Val 세트 Sharpe Ratio (mean of n_eval_episodes)
    - Trial당 : total_timesteps = TRIAL_TIMESTEPS (기본 1M)
    - 총 시도 : N_TRIALS (기본 50)
    - 중단 재개: SQLite DB (experiments/optuna_coef_v1/study.db)

사용법:
    # 최초 실행
    python scripts/bayesian_coef_tuning.py

    # 추가 시도 (중단 후 재개)
    python scripts/bayesian_coef_tuning.py --n-trials 30

    # 결과만 출력 (학습 없음)
    python scripts/bayesian_coef_tuning.py --show-results
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit

from src.env.trading_env import BTCGridTradingEnv
from src.utils.config import load_config
from src.evaluation.metrics import sharpe_ratio as calc_sharpe


# ── 전역 설정 ─────────────────────────────────────────────────────────────────
BASE_CONFIG   = ROOT / "config" / "exp013b_config.yaml"
TRAIN_DATA    = ROOT / "data" / "processed" / "btc_train.parquet"
VAL_DATA      = ROOT / "data" / "processed" / "btc_val.parquet"
RESULTS_DIR   = ROOT / "experiments" / "optuna_coef_v1"
DB_PATH       = RESULTS_DIR / "study.db"
STUDY_NAME    = "coef_tuning_v1"

N_TRIALS         = 50
TRIAL_TIMESTEPS  = 1_000_000   # 빠른 평가용 (기본 설계 3M의 1/3)
N_ENVS           = 4
SEED             = 42

# 탐색 범위 — 현재 설계값(0.5/1.5/2.5/7.5)이 범위 내 포함됨
SEARCH_SPACE = {
    "A_b": (0.1, 3.0),    # buy_hi_gap base  (기본 0.5)
    "B_b": (0.0, 8.0),    # buy_hi_gap range (기본 1.5)
    "C_b": (1.0, 12.0),   # buy_lo_gap base  (기본 2.5) — C_b > A_b 권장
    "D_b": (2.0, 25.0),   # buy_lo_gap range (기본 7.5)
    "A_s": (0.1, 3.0),    # sell_market_gap base  (기본 0.5)
    "B_s": (0.0, 8.0),    # sell_market_gap range (기본 1.5)
    "C_s": (1.0, 12.0),   # sell_cost_gap base    (기본 2.5)
    "D_s": (2.0, 25.0),   # sell_cost_gap range   (기본 7.5)
}


# ── LR 스케줄 헬퍼 ────────────────────────────────────────────────────────────

def _build_lr_schedule(agent_cfg: dict) -> float | Callable:
    lr_start = float(agent_cfg["learning_rate"])
    schedule = agent_cfg.get("lr_schedule", "constant")
    if schedule == "constant":
        return lr_start
    lr_end = float(agent_cfg.get("lr_end", lr_start * 0.1))
    if schedule == "linear":
        def _lr(p): return lr_end + (lr_start - lr_end) * p
        return _lr
    if schedule == "cosine":
        def _lr(p):
            prog = 1.0 - p
            return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + math.cos(math.pi * prog))
        return _lr
    return lr_start


# ── Val Sharpe 계산 ───────────────────────────────────────────────────────────

def evaluate_val_sharpe(
    model: PPO,
    val_df: pd.DataFrame,
    config: dict,
    n_episodes: int = 5,
) -> float:
    """Val 세트에서 n_episodes 에피소드 실행 → 평균 Sharpe 반환."""
    ep_len = config["training"].get("max_episode_steps", 2016)
    sharpes = []

    for _ in range(n_episodes):
        env = BTCGridTradingEnv(val_df, config)
        # random_start=True이므로 에피소드마다 다른 시작점 사용
        obs, _ = env.reset()
        equity_curve = [env._equity(float(val_df.iloc[env.current_step]["close"]))]
        done = False
        steps = 0

        while not done and steps < ep_len:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            equity_curve.append(info["equity"])
            done = terminated or truncated
            steps += 1

        if len(equity_curve) > 10:
            sharpes.append(calc_sharpe(pd.Series(equity_curve)))

    return float(np.mean(sharpes)) if sharpes else -999.0


# ── Pruning 콜백 ──────────────────────────────────────────────────────────────

class PruningCallback(BaseCallback):
    """
    학습 중간에 Val Sharpe를 평가해 Optuna Pruner에 보고.
    하위 50% 시도는 조기 중단 → 계산 자원 절약.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        val_df: pd.DataFrame,
        config: dict,
        eval_freq: int,
        n_eval_episodes: int = 3,
    ):
        super().__init__(verbose=0)
        self.trial           = trial
        self.val_df          = val_df
        self.config          = config
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._last_eval_ts   = 0
        self._report_num     = 0

    def _on_step(self) -> bool:
        ts = self.model.num_timesteps
        if ts - self._last_eval_ts >= self.eval_freq:
            self._last_eval_ts = ts
            sharpe = evaluate_val_sharpe(
                self.model, self.val_df, self.config, self.n_eval_episodes
            )
            self.trial.report(sharpe, self._report_num)
            self._report_num += 1
            if self.trial.should_prune():
                return False   # SB3 학습 중단 신호
        return True


# ── 목적함수 ──────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """
    8개 계수 샘플링 → PPO 학습 → Val Sharpe 반환.

    제약 조건 (소프트):
        C_b > A_b  (buy_lo_gap base > buy_hi_gap base, agg=0 에서도 깊은 레벨 유지)
        D_b > B_b  (buy_lo의 action 감도 ≥ buy_hi)
        C_s > A_s  / D_s > B_s  (sell_cost > sell_market)
    조건 미충족 시 −999 반환 (sampler가 해당 영역 회피 학습).
    """
    # ── 계수 샘플링 ────────────────────────────────────────────────────────────
    params = {k: trial.suggest_float(k, lo, hi)
              for k, (lo, hi) in SEARCH_SPACE.items()}

    # 소프트 제약 검사
    if params["C_b"] <= params["A_b"] or params["D_b"] <= params["B_b"]:
        return -999.0
    if params["C_s"] <= params["A_s"] or params["D_s"] <= params["B_s"]:
        return -999.0

    # ── Config 구성 ────────────────────────────────────────────────────────────
    config = load_config(str(BASE_CONFIG))
    config["environment"]["formula_coefs"] = params
    config["agent"]["total_timesteps"]     = TRIAL_TIMESTEPS

    # ── 학습 환경 구성 ─────────────────────────────────────────────────────────
    ep_len = config["training"].get("max_episode_steps", 2016)

    def _make_train():
        env = BTCGridTradingEnv(train_df, config)
        return TimeLimit(env, max_episode_steps=ep_len)

    train_vec = DummyVecEnv([_make_train for _ in range(N_ENVS)])

    # ── PPO 구성 ───────────────────────────────────────────────────────────────
    cfg_a = config["agent"]
    lr    = _build_lr_schedule(cfg_a)
    model = PPO(
        "MlpPolicy",
        train_vec,
        learning_rate    = lr,
        n_steps          = cfg_a["n_steps"],
        batch_size       = cfg_a["batch_size"],
        n_epochs         = cfg_a["n_epochs"],
        gamma            = cfg_a["gamma"],
        gae_lambda       = cfg_a["gae_lambda"],
        clip_range       = cfg_a["clip_range"],
        ent_coef         = cfg_a["ent_coef"],
        vf_coef          = cfg_a["vf_coef"],
        max_grad_norm    = cfg_a["max_grad_norm"],
        seed             = SEED,
        verbose          = 0,
        device           = "cpu",
    )

    # 중간 평가: 3회 (0%, 33%, 66%, 100% 지점)
    eval_freq    = TRIAL_TIMESTEPS // 3
    pruning_cb   = PruningCallback(trial, val_df, config, eval_freq, n_eval_episodes=3)

    pruned = False
    try:
        model.learn(
            total_timesteps    = TRIAL_TIMESTEPS,
            callback           = pruning_cb,
            reset_num_timesteps= True,
        )
    except (KeyboardInterrupt, SystemExit):
        train_vec.close()
        raise
    finally:
        if not pruned:
            train_vec.close()

    # ── 최종 Val Sharpe ────────────────────────────────────────────────────────
    sharpe = evaluate_val_sharpe(model, val_df, config, n_episodes=5)
    return sharpe


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def print_results(study: optuna.Study) -> None:
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.FAIL]

    print(f"\n{'='*65}")
    print(f"  Optuna 결과 요약 -- {STUDY_NAME}")
    print(f"{'='*65}")
    print(f"  완료: {len(completed)}  |  조기중단: {len(pruned)}  |  실패: {len(failed)}")

    if not completed:
        print("  완료된 Trial 없음.")
        return

    best = study.best_trial
    baseline = 17.579   # exp013b Val Sharpe
    delta    = best.value - baseline

    print(f"\n  최적 Trial #{best.number}:")
    print(f"    Val Sharpe = {best.value:.3f}  "
          f"({'↑' if delta >= 0 else '↓'}{abs(delta):.3f} vs. exp013b {baseline})")
    print(f"\n  계수:")
    for k, v in best.params.items():
        default = {"A_b":0.5,"B_b":1.5,"C_b":2.5,"D_b":7.5,
                   "A_s":0.5,"B_s":1.5,"C_s":2.5,"D_s":7.5}[k]
        arrow = "→" if abs(v - default) < 0.01 else ("↑" if v > default else "↓")
        print(f"    {k}: {v:.4f}  (기본값 {default}  {arrow})")

    # Top-5
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n  Top-5 Trial:")
    print(f"  {'#':>4}  {'Sharpe':>8}  {'A_b':>6}  {'B_b':>6}  "
          f"{'C_b':>6}  {'D_b':>6}  {'A_s':>6}  {'B_s':>6}  {'C_s':>6}  {'D_s':>6}")
    print("  " + "-" * 63)
    for t in top5:
        p = t.params
        print(f"  {t.number:>4}  {t.value:>8.3f}  "
              f"{p.get('A_b',0):>6.3f}  {p.get('B_b',0):>6.3f}  "
              f"{p.get('C_b',0):>6.3f}  {p.get('D_b',0):>6.3f}  "
              f"{p.get('A_s',0):>6.3f}  {p.get('B_s',0):>6.3f}  "
              f"{p.get('C_s',0):>6.3f}  {p.get('D_s',0):>6.3f}")
    print()


# ── 최적 파라미터 저장 ─────────────────────────────────────────────────────────

def save_best(study: optuna.Study) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best = study.best_trial

    out = {
        "study_name"    : STUDY_NAME,
        "best_trial"    : best.number,
        "best_sharpe"   : round(best.value, 4),
        "baseline_sharpe": 17.579,
        "formula_coefs" : {k: round(v, 6) for k, v in best.params.items()},
    }

    yaml_path = RESULTS_DIR / "best_coefs.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True)
    print(f"  Best 파라미터 저장: {yaml_path}")

    # 전체 trials CSV
    csv_path = RESULTS_DIR / "all_trials.csv"
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"  전체 결과 저장 : {csv_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Bayesian 계수 튜닝 (Optuna TPE)")
    p.add_argument("--n-trials",      type=int, default=N_TRIALS,
                   help=f"시도 횟수 (기본값: {N_TRIALS})")
    p.add_argument("--trial-steps",   type=int, default=TRIAL_TIMESTEPS,
                   help=f"Trial당 학습 스텝 (기본값: {TRIAL_TIMESTEPS:,})")
    p.add_argument("--show-results",  action="store_true",
                   help="학습 없이 저장된 결과만 출력")
    p.add_argument("--n-jobs",        type=int, default=1,
                   help="병렬 Trial 수 (기본값: 1, 순차 실행)")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Study 생성 / 로드 ──────────────────────────────────────────────────────
    storage  = f"sqlite:///{DB_PATH}"
    sampler  = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10)
    pruner   = optuna.pruners.MedianPruner(
        n_startup_trials=10,   # 첫 10회는 pruning 없음 (warm-up)
        n_warmup_steps=1,      # 첫 번째 중간 체크 이후부터 pruning 판단
    )

    study = optuna.create_study(
        study_name    = STUDY_NAME,
        storage       = storage,
        direction     = "maximize",
        sampler       = sampler,
        pruner        = pruner,
        load_if_exists= True,
    )

    existing = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nStudy '{STUDY_NAME}' 로드 완료 (기존 완료: {existing}회)")

    if args.show_results:
        print_results(study)
        if existing > 0:
            save_best(study)
        return

    # ── 데이터 로드 ────────────────────────────────────────────────────────────
    print(f"데이터 로드 중…")
    train_df = pd.read_parquet(TRAIN_DATA)
    val_df   = pd.read_parquet(VAL_DATA)
    print(f"  Train: {len(train_df):,}봉  |  Val: {len(val_df):,}봉")

    # ── 최적화 실행 ───────────────────────────────────────────────────────────
    n_trials      = args.n_trials
    trial_steps   = args.trial_steps
    est_min_trial = trial_steps / 1_000_000 * 5   # 대략 1M steps ≈ 5분 (n_envs=4)
    est_total     = n_trials * est_min_trial / 60  # 시간 추정

    print(f"\nBayesian 계수 튜닝 시작")
    print(f"  시도 횟수 : {n_trials}회 추가 ({existing + n_trials}회 누적)")
    print(f"  Trial당   : {trial_steps/1e6:.1f}M 스텝")
    print(f"  예상 시간 : {est_total:.1f}시간 (pruning으로 단축 가능)")
    print(f"  SQLite DB : {DB_PATH}")
    print(f"  Baseline  : exp013b Sharpe = 17.579\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    total_start = time.time()

    study.optimize(
        lambda trial: objective(trial, train_df, val_df),
        n_trials    = n_trials,
        n_jobs      = args.n_jobs,
        show_progress_bar = (args.n_jobs == 1),
        gc_after_trial    = True,
    )

    elapsed = (time.time() - total_start) / 3600
    print(f"\n완료: {elapsed:.2f}시간 소요")

    # ── 결과 출력 및 저장 ──────────────────────────────────────────────────────
    print_results(study)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        save_best(study)


if __name__ == "__main__":
    main()
