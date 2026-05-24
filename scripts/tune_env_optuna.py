"""
scripts/tune_env_optuna.py

환경 파라미터 Optuna 탐색 (exp029 2단계).

탐색 대상: coef_b1_max, coef_b2_max, coef_s1_max, coef_s2_max,
           w_cycle, idle_rate, grace_period
PPO 파라미터: exp029 Optuna 결과값으로 고정

사용법:
    python scripts/tune_env_optuna.py
    python scripts/tune_env_optuna.py --trials 50 --exp-name exp029_env_optuna
"""

import argparse
import sys
from pathlib import Path
from copy import deepcopy

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
    p.add_argument("--trials",    type=int, default=50)
    p.add_argument("--exp-name",  type=str, default="exp029_env_optuna")
    p.add_argument("--timesteps", type=int, default=200000)
    return p.parse_args()


def objective(trial: optuna.Trial, df_train, df_val, base_cfg: dict, timesteps: int) -> float:
    cfg = deepcopy(base_cfg)

    # ── 탐색 공간 ──────────────────────────────────────────────
    coef_b1_max  = trial.suggest_float("coef_b1_max",  1.0,  10.0)
    coef_b2_max  = trial.suggest_float("coef_b2_max",  3.0,  25.0)
    coef_s1_max  = trial.suggest_float("coef_s1_max",  1.0,  10.0)
    coef_s2_max  = trial.suggest_float("coef_s2_max",  3.0,  25.0)
    w_cycle      = trial.suggest_float("w_cycle",      1.0,  10.0)
    idle_rate    = trial.suggest_float("idle_rate",    1e-6, 1e-4, log=True)
    grace_period = trial.suggest_int(  "grace_period", 1,    72)

    cfg["environment"]["coef_b1_max"]  = coef_b1_max
    cfg["environment"]["coef_b2_max"]  = coef_b2_max
    cfg["environment"]["coef_s1_max"]  = coef_s1_max
    cfg["environment"]["coef_s2_max"]  = coef_s2_max
    cfg["environment"]["w_cycle"]      = w_cycle
    cfg["environment"]["idle_rate"]    = idle_rate
    cfg["environment"]["grace_period"] = grace_period

    cfg["agent"]["total_timesteps"] = timesteps
    cfg["training"]["log_dir"]          = f"experiments/_optuna_trial_{trial.number}"
    cfg["training"]["experiment_name"]  = f"trial_{trial.number}"

    try:
        agent = PPOAgent(cfg, df_train, df_val)
        agent.train(total_timesteps=timesteps)
        metrics = agent.evaluate(df_val)
        sharpe = float(metrics["sharpe_ratio"])
    except Exception as e:
        print(f"  [trial {trial.number}] 오류: {e}")
        return float("-inf")

    print(f"  [trial {trial.number:3d}] Sharpe={sharpe:+.3f} | "
          f"b1={coef_b1_max:.1f} b2={coef_b2_max:.1f} "
          f"s1={coef_s1_max:.1f} s2={coef_s2_max:.1f} "
          f"wc={w_cycle:.1f} ir={idle_rate:.1e} gp={grace_period}")
    return sharpe


def main():
    args = parse_args()

    cfg      = load_config()
    df_train = pd.read_parquet("data/processed/btc_train.parquet")
    df_val   = pd.read_parquet("data/processed/btc_val.parquet")

    log_dir = Path("experiments") / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Optuna 환경 파라미터 탐색 (2단계)")
    print(f"  trials={args.trials}  timesteps/trial={args.timesteps:,}")
    print(f"  결과 저장: {log_dir}")
    print(f"  baseline (fixed_grid_5pct): Val Sharpe 1.051\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="env_param_v1",
        storage=f"sqlite:///{log_dir}/optuna.db",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, df_train, df_val, cfg, args.timesteps),
        n_trials=args.trials,
        show_progress_bar=False,
    )

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
        "baseline_sharpe": 1.051,
    }
    with open(log_dir / "best_params.yaml", "w") as f:
        yaml.dump(result, f, default_flow_style=False)

    print(f"\n상위 5 trials:")
    top5 = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True
    )[:5]
    for t in top5:
        print(f"  #{t.number:3d}  Sharpe={t.value:.3f}  {t.params}")

    print(f"\n결과 저장: {log_dir}/best_params.yaml")


if __name__ == "__main__":
    main()
