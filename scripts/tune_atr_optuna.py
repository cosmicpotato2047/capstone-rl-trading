"""
scripts/tune_atr_optuna.py

ATR 고정 공식 계수 Bayesian 최적화.
PPO 학습 없이 고정 정책([1.0, 0.0])으로 Val 전체를 평가하므로 trial당 수초.

탐색 대상:
    A_b   : buy_hi_gap  = atr × A_b               현재 0.285
    C_b   : buy_lo_gap  = atr × C_b               현재 5.223
    A_s   : sell_market_gap = atr × A_s           현재 0.05  (미최적화 가능성)
    C_s   : sell_cost_gap   = atr × C_s           현재 2.5   (한 번도 최적화 안 됨)
    n_splits : 현금 분할 수                        현재 4     (CLAUDE.md에 튜닝 예정)

고정 정책:
    action = [1.0, 0.0]
    → entry_gate = 1.0 (항상 진입 허가)
    → profit_target = 0.0 (A_s, C_s만 작동)

사용법:
    python scripts/tune_atr_optuna.py
    python scripts/tune_atr_optuna.py --trials 100 --exp-name exp023_atr_optuna
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.env.trading_env import BTCGridTradingEnv
from src.evaluation.metrics import compute_all

optuna.logging.set_verbosity(optuna.logging.WARNING)

FIXED_ACTION = np.array([1.0, 0.0], dtype=np.float32)


def evaluate_fixed(df: pd.DataFrame, cfg: dict) -> float:
    """고정 정책 [1.0, 0.0]으로 Val 전체 실행 → Sharpe 반환."""
    eval_cfg = deepcopy(cfg)
    eval_cfg["environment"]["random_start"] = False

    env = BTCGridTradingEnv(df, eval_cfg)
    obs, _ = env.reset()
    equity_list = [cfg["environment"]["initial_cash"]]
    done = False

    while not done:
        obs, _, terminated, truncated, _ = env.step(FIXED_ACTION)
        done = terminated or truncated
        price = float(env.df.loc[env.current_step - 1, "close"])
        equity_list.append(env.cash + env.holdings * price)

    equity = pd.Series(equity_list, dtype=float)
    m = compute_all(equity, cfg["environment"]["initial_cash"],
                    env.n_trades, env.completed_cycles)
    return float(m["sharpe_ratio"])


def objective(trial: optuna.Trial, df_val: pd.DataFrame, base_cfg: dict) -> float:
    cfg = deepcopy(base_cfg)

    # ── 탐색 공간 ──────────────────────────────────────────────
    A_b      = trial.suggest_float("A_b",      0.05,  1.0)
    C_b      = trial.suggest_float("C_b",      1.0,  15.0)
    A_s      = trial.suggest_float("A_s",      0.01,  0.20)
    C_s      = trial.suggest_float("C_s",      0.5,   6.0)
    n_splits = trial.suggest_int(  "n_splits", 2,     8)

    cfg["environment"]["formula_coefs"]["A_b"] = A_b
    cfg["environment"]["formula_coefs"]["C_b"] = C_b
    cfg["environment"]["formula_coefs"]["A_s"] = A_s
    cfg["environment"]["formula_coefs"]["C_s"] = C_s
    cfg["environment"]["n_splits"]             = n_splits

    try:
        sharpe = evaluate_fixed(df_val, cfg)
    except Exception as e:
        print(f"  [trial {trial.number}] 오류: {e}")
        return float("-inf")

    print(f"  [trial {trial.number:3d}] Sharpe={sharpe:+.3f} | "
          f"A_b={A_b:.3f} C_b={C_b:.3f} A_s={A_s:.3f} C_s={C_s:.3f} n_splits={n_splits}")
    return sharpe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials",   type=int, default=50)
    p.add_argument("--exp-name", type=str, default="exp023_atr_optuna")
    args = p.parse_args()

    cfg    = load_config()
    df_val = pd.read_parquet("data/processed/btc_val.parquet")

    log_dir = Path("experiments") / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # 현재 계수로 baseline Sharpe 계산
    baseline_sharpe = evaluate_fixed(df_val, cfg)
    print(f"ATR 공식 계수 Bayesian 최적화 (고정 정책)")
    print(f"  trials={args.trials}  Val: {len(df_val):,}봉")
    print(f"  현재 계수: A_b=0.285 C_b=5.223 A_s=0.05 C_s=2.5 n_splits=4")
    print(f"  현재 Val Sharpe: {baseline_sharpe:.3f}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="atr_coef_v1",
        storage=f"sqlite:///{log_dir}/optuna.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # 현재 계수를 첫 번째 trial로 시드
    study.enqueue_trial({
        "A_b": 0.285, "C_b": 5.223,
        "A_s": 0.05,  "C_s": 2.5,
        "n_splits": 4,
    })

    study.optimize(
        lambda trial: objective(trial, df_val, cfg),
        n_trials=args.trials,
    )

    # ── 결과 ───────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"최적 Trial #{best.number}  Val Sharpe={best.value:.3f}  "
          f"(기존 {baseline_sharpe:.3f}, 개선 {best.value - baseline_sharpe:+.3f})")
    print(f"{'='*60}")
    for k, v in best.params.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    result = {
        "best_trial":       best.number,
        "best_sharpe":      round(best.value, 4),
        "baseline_sharpe":  round(baseline_sharpe, 4),
        "improvement":      round(best.value - baseline_sharpe, 4),
        "best_params":      best.params,
    }
    out_path = log_dir / "best_params.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, allow_unicode=True, default_flow_style=False)

    print(f"\n상위 5 trials:")
    top5 = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True
    )[:5]
    for t in top5:
        p = t.params
        print(f"  #{t.number:3d} Sharpe={t.value:.3f} | "
              f"A_b={p['A_b']:.3f} C_b={p['C_b']:.3f} "
              f"A_s={p['A_s']:.3f} C_s={p['C_s']:.3f} n_splits={p['n_splits']}")

    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
