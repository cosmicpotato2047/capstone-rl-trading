"""
scripts/tune_ppo_optuna.py

PPO 하이퍼파라미터 Optuna 탐색.

탐색 대상: ent_coef, learning_rate, n_steps, gamma, gae_lambda, clip_range
탐색 제외: 공식 계수 (A_s/B_s 등) — RL action 역할 충돌 방지

사용법:
    python scripts/tune_ppo_optuna.py
    python scripts/tune_ppo_optuna.py --trials 30 --exp-name exp018_optuna
"""

import argparse
import sys
import os
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import optuna
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.agents.ppo_agent import PPOAgent

optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trials",   type=int, default=30)
    p.add_argument("--exp-name", type=str, default="exp018_optuna")
    p.add_argument("--timesteps", type=int, default=500000,
                   help="trial당 학습 스텝 (기본 500k — 빠른 탐색)")
    return p.parse_args()


def objective(trial: optuna.Trial, df_train, df_val, base_cfg: dict, timesteps: int) -> float:
    cfg = deepcopy(base_cfg)

    # ── 탐색 공간 ──────────────────────────────────────────────
    ent_coef      = trial.suggest_float("ent_coef",      0.001, 0.2,   log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5,  1e-3,  log=True)
    n_steps       = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    gamma         = trial.suggest_float("gamma",         0.95,  0.999)
    gae_lambda    = trial.suggest_float("gae_lambda",    0.9,   0.99)
    clip_range    = trial.suggest_float("clip_range",    0.1,   0.4)

    cfg["agent"]["ent_coef"]      = ent_coef
    cfg["agent"]["learning_rate"] = learning_rate
    cfg["agent"]["n_steps"]       = n_steps
    cfg["agent"]["gamma"]         = gamma
    cfg["agent"]["gae_lambda"]    = gae_lambda
    cfg["agent"]["clip_range"]    = clip_range
    cfg["agent"]["total_timesteps"] = timesteps

    # trial별 log_dir (덮어쓰기 방지)
    cfg["training"]["log_dir"] = f"experiments/_optuna_trial_{trial.number}"
    cfg["training"]["experiment_name"] = f"trial_{trial.number}"

    try:
        agent = PPOAgent(cfg, df_train, df_val)
        agent.train(total_timesteps=timesteps)
        metrics = agent.evaluate(df_val)
        sharpe = float(metrics["sharpe_ratio"])
    except Exception as e:
        print(f"  [trial {trial.number}] 오류: {e}")
        return float("-inf")

    print(f"  [trial {trial.number:3d}] Sharpe={sharpe:+.3f} | "
          f"ent={ent_coef:.4f} lr={learning_rate:.2e} "
          f"n_steps={n_steps} gamma={gamma:.3f}")
    return sharpe


def main():
    args = parse_args()

    cfg      = load_config()
    df_train = pd.read_parquet("data/processed/btc_train.parquet")
    df_val   = pd.read_parquet("data/processed/btc_val.parquet")

    log_dir = Path("experiments") / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Optuna PPO 하이퍼파라미터 탐색")
    print(f"  trials={args.trials}  timesteps/trial={args.timesteps:,}")
    print(f"  결과 저장: {log_dir}")
    print(f"  baseline (exp017): Val Sharpe 38.186\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="ppo_hparam_v1",
        storage=f"sqlite:///{log_dir}/optuna.db",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, df_train, df_val, cfg, args.timesteps),
        n_trials=args.trials,
        show_progress_bar=False,
    )

    # ── 결과 저장 ──────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"최적 Trial #{best.number}  Val Sharpe={best.value:.3f}")
    print(f"{'='*60}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    result = {
        "best_trial":  best.number,
        "best_sharpe": best.value,
        "best_params": best.params,
        "baseline_sharpe": 38.186,
    }
    with open(log_dir / "best_params.yaml", "w") as f:
        yaml.dump(result, f, default_flow_style=False)

    # 상위 5개 출력
    print(f"\n상위 5 trials:")
    top5 = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True
    )[:5]
    for t in top5:
        print(f"  #{t.number:3d}  Sharpe={t.value:.3f}  {t.params}")

    print(f"\n결과 저장: {log_dir}/best_params.yaml")
    print(f"다음 단계: python scripts/train_ppo.py --exp-name {args.exp_name.replace('optuna','final')}")


if __name__ == "__main__":
    main()
