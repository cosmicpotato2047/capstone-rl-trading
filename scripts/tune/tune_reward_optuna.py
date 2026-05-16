"""
scripts/tune_reward_optuna.py

Reward variant 별 hyperparameter Optuna 튜닝 (exp032a).

각 variant 의 hyperparameter:
    asym: reward_loss_beta ∈ [1.0, 4.0]
    dsr:  dsr_eta ∈ [1/720, 1/24]  (월~일 EMA)
    pt:   pt_alpha ∈ [0.5, 1.0], pt_lambda ∈ [1.0, 4.0]

각 trial: 200k steps × Env-v4 + exp030 안정화 패키지.
trial 평가: Val Sharpe (n_eval_episodes=3 평균).

사용법:
    python scripts/tune_reward_optuna.py --variant asym --trials 30
    python scripts/tune_reward_optuna.py --variant dsr  --trials 30
    python scripts/tune_reward_optuna.py --variant pt   --trials 30
"""
import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import optuna
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.agents.ppo_agent import PPOAgent

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _train_and_eval(cfg: dict, total_timesteps: int = 200_000) -> float:
    """단일 학습 + Val 평가 → best Sharpe 반환 (callback 의 best 시점)."""
    df_train = pd.read_parquet("data/processed/btc_train.parquet")
    df_val   = pd.read_parquet("data/processed/btc_val.parquet")

    agent = PPOAgent(cfg, df_train, df_val)
    log_dir = cfg["training"]["log_dir"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    best_model_path = os.path.join(log_dir, "best_model")
    agent.train(total_timesteps=total_timesteps, best_model_path=best_model_path)

    # ValMetricsCallback 의 best_sharpe 가 학습 중 best
    # callback 가 self.callbacks 에 없으므로 model.learn 후 best 알기 어려움
    # → final 평가
    metrics = agent.evaluate(df_val)
    return float(metrics["sharpe_ratio"])


def objective(trial: optuna.Trial, variant: str, base_cfg: dict, total_timesteps: int) -> float:
    cfg = deepcopy(base_cfg)

    # variant 별 hyperparameter 탐색
    cfg["environment"]["reward_type"] = variant
    if variant == "asym":
        beta = trial.suggest_float("reward_loss_beta", 1.0, 4.0)
        cfg["environment"]["reward_loss_beta"] = beta
        msg = f"β={beta:.3f}"
    elif variant == "dsr":
        eta = trial.suggest_float("dsr_eta", 1.0/720.0, 1.0/24.0, log=True)
        cfg["environment"]["dsr_eta"] = eta
        msg = f"η={eta:.5f}"
    elif variant == "pt":
        alpha = trial.suggest_float("pt_alpha", 0.5, 1.0)
        lam   = trial.suggest_float("pt_lambda", 1.0, 4.0)
        cfg["environment"]["pt_alpha"]  = alpha
        cfg["environment"]["pt_lambda"] = lam
        msg = f"α={alpha:.3f} λ={lam:.3f}"
    else:
        raise ValueError(f"variant 는 'asym'|'dsr'|'pt' 중 선택: {variant!r}")

    # trial 별 log_dir
    cfg["training"]["experiment_name"] = f"exp032a_{variant}_trial_{trial.number}"
    cfg["training"]["log_dir"] = f"experiments/exp032a_{variant}/trial_{trial.number}"

    try:
        sharpe = _train_and_eval(cfg, total_timesteps=total_timesteps)
    except Exception as e:
        print(f"  [trial {trial.number}] error: {e}")
        return float("-inf")

    print(f"  [trial {trial.number:2d}] {variant} {msg} → Val Sharpe {sharpe:+.3f}")
    return sharpe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["asym", "dsr", "pt"], required=True)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--timesteps", type=int, default=200_000,
                   help="trial 당 학습 step (default 200k)")
    p.add_argument("--config", default="config/exp030_stabilization_config.yaml",
                   help="base config (exp030 안정화 패키지 사용)")
    args = p.parse_args()

    cfg = load_config(args.config)

    log_dir = Path(f"experiments/exp032a_{args.variant}")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reward Variant '{args.variant}' Bayesian 튜닝")
    print(f"  trials={args.trials}, timesteps={args.timesteps:,}, base={args.config}")
    print(f"  log_dir={log_dir}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"exp032a_{args.variant}",
        storage=f"sqlite:///{log_dir}/optuna.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    study.optimize(
        lambda trial: objective(trial, args.variant, cfg, args.timesteps),
        n_trials=args.trials,
    )

    best = study.best_trial
    print(f"\n{'='*60}\n최적 Trial #{best.number}  Val Sharpe={best.value:.3f}")
    print(f"{'='*60}")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}")
        else:
            print(f"  {k}: {v}")

    result = {
        "variant":     args.variant,
        "best_trial":  best.number,
        "best_sharpe": round(best.value, 4),
        "best_params": best.params,
        "n_trials":    len(study.trials),
    }
    out_path = log_dir / "best_params.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, allow_unicode=True, default_flow_style=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
