"""
scripts/train_ppo.py

PPO 학습 실행 스크립트.

사용법:
    python scripts/train_ppo.py
    python scripts/train_ppo.py --config config/experiment_config.yaml
    python scripts/train_ppo.py --timesteps 500000 --exp-name exp002_test

실행 흐름:
    1. Train / Val 데이터 로드
    2. PPOAgent 생성 (config에서 하이퍼파라미터 읽음)
    3. MLflow 실험 시작
    4. 학습 (eval_freq마다 Val Sharpe 평가 + 최고 모델 저장)
    5. 최종 Val 평가 및 베이스라인 비교 출력
    6. 모델 저장 + MLflow 종료
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd

# 프로젝트 루트를 sys.path에 추가 (editable install 없이도 실행 가능하도록)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.agents.ppo_agent import PPOAgent
from src.agents.baselines import run_all_baselines
from src.evaluation.metrics import compute_all, print_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI 인자 파싱
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO 그리드 트레이딩 에이전트 학습")
    parser.add_argument(
        "--config", default="config/experiment_config.yaml",
        help="실험 설정 파일 경로 (기본값: config/experiment_config.yaml)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="학습 스텝 수 (기본값: config의 total_timesteps)"
    )
    parser.add_argument(
        "--exp-name", type=str, default=None,
        help="MLflow 실험 이름 (기본값: config의 experiment_name)"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="MLflow 로깅 비활성화"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. 설정 로드 ──────────────────────────────────────────
    config = load_config(args.config)
    train_cfg = config["training"]

    exp_name = args.exp_name or train_cfg["experiment_name"]
    log_dir  = train_cfg["log_dir"]
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # config 스냅샷 저장 (재현성)
    import shutil
    shutil.copy(args.config, os.path.join(log_dir, "config_snapshot.yaml"))

    # ── 2. 데이터 로드 ────────────────────────────────────────
    print("데이터 로드 중...")
    df_train = pd.read_parquet("data/processed/btc_train.parquet")
    df_val   = pd.read_parquet("data/processed/btc_val.parquet")
    print(f"  Train: {len(df_train):,}행  |  Val: {len(df_val):,}행")

    # ── 3. MLflow 설정 ────────────────────────────────────────
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        mlflow.set_tracking_uri(os.path.join(log_dir, "mlruns"))
        mlflow.set_experiment(exp_name)
        run = mlflow.start_run(run_name="ppo_train")
        # 하이퍼파라미터 기록
        agent_cfg   = config["agent"]
        vec_cfg     = config.get("vec_normalize", {})
        env_cfg = config["environment"]
        params = {
            "algorithm":       agent_cfg["algorithm"],
            "learning_rate":   agent_cfg["learning_rate"],
            "lr_schedule":     agent_cfg.get("lr_schedule", "constant"),
            "lr_end":          agent_cfg.get("lr_end", agent_cfg["learning_rate"]),
            "n_steps":         agent_cfg["n_steps"],
            "batch_size":      agent_cfg["batch_size"],
            "n_epochs":        agent_cfg["n_epochs"],
            "gamma":           agent_cfg["gamma"],
            "ent_coef":        agent_cfg["ent_coef"],
            "total_timesteps": agent_cfg["total_timesteps"],
            # Env-v4 environment params
            "n_buy_orders":     env_cfg.get("n_buy_orders", 2),
            "n_splits":         env_cfg.get("n_splits", 2),
            "threshold_basis":  env_cfg.get("threshold_basis", "price"),
            "reward_loss_beta": env_cfg.get("reward_loss_beta", 1.0),
            "vec_normalize":   vec_cfg.get("enabled", False),
            "norm_obs":        vec_cfg.get("norm_obs", False),
            "norm_reward":     vec_cfg.get("norm_reward", False),
        }
        mlflow.log_params(params)
        print(f"MLflow 실험: {exp_name}  (run_id: {run.info.run_id})")

    # ── 4. PPOAgent 생성 + 학습 ───────────────────────────────
    best_model_path = os.path.join(log_dir, "best_model")

    agent = PPOAgent(config, df_train, df_val)
    agent.train(
        total_timesteps=args.timesteps,
        best_model_path=best_model_path,
    )

    # 최종 모델도 별도 저장
    final_model_path = os.path.join(log_dir, "final_model")
    agent.save(final_model_path)

    # ── 5. 최종 Val 평가 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("최종 Val 평가 (final_model)")
    print("=" * 60)

    val_metrics = agent.evaluate(df_val)
    print_metrics(val_metrics, label="PPO (final)")

    # 베이스라인 비교
    print("\n" + "=" * 60)
    print("베이스라인 비교")
    print("=" * 60)
    bl_results = run_all_baselines(df_val, config)
    initial_cash = config["environment"]["initial_cash"]

    print(f"\n{'전략':<22} {'수익률(%)':>9} {'Sharpe':>7} {'MDD(%)':>8} {'거래':>6} {'사이클':>6}")
    print("-" * 64)
    for name, r in bl_results.items():
        m = compute_all(r["equity_curve"], initial_cash, r["n_trades"], r["completed_cycles"])
        marker = ""
        print(f"{name:<22} {m['total_return_pct']:>9.2f} {m['sharpe_ratio']:>7.3f} "
              f"{m['max_drawdown_pct']:>8.2f} {m['n_trades']:>6} {m['n_cycles']:>6}")

    ppo_sharpe = val_metrics["sharpe_ratio"]
    best_bl_sharpe = max(
        compute_all(r["equity_curve"], initial_cash, r["n_trades"], r["completed_cycles"])["sharpe_ratio"]
        for r in bl_results.values()
    )
    print(f"\n{'PPO (final)':<22} {val_metrics['total_return_pct']:>9.2f} "
          f"{ppo_sharpe:>7.3f} {val_metrics['max_drawdown_pct']:>8.2f} "
          f"{val_metrics['n_trades']:>6} {val_metrics['n_cycles']:>6}")
    print(f"\n→ PPO Sharpe {ppo_sharpe:.3f} vs 베이스라인 최고 {best_bl_sharpe:.3f} "
          f"({'[우위]' if ppo_sharpe > best_bl_sharpe else '[미달]'})")

    # ── 6. MLflow 최종 지표 기록 + 종료 ──────────────────────
    if use_mlflow:
        mlflow.log_metrics({
            "final_val_sharpe":     val_metrics["sharpe_ratio"],
            "final_val_return_pct": val_metrics["total_return_pct"],
            "final_val_mdd_pct":    val_metrics["max_drawdown_pct"],
            "final_val_n_trades":   val_metrics["n_trades"],
            "final_val_n_cycles":   val_metrics["n_cycles"],
            "best_baseline_sharpe": best_bl_sharpe,
        })
        mlflow.log_artifact(os.path.join(log_dir, "config_snapshot.yaml"))
        mlflow.end_run()
        print(f"\nMLflow 기록 완료: {os.path.join(log_dir, 'mlruns')}")

    print(f"\n모델 저장 위치:")
    print(f"  best  : {best_model_path}.zip")
    print(f"  final : {final_model_path}.zip")


if __name__ == "__main__":
    main()
