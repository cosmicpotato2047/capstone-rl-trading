"""
scripts/eval_test.py

Phase 2-B: Test set 최종 평가 스크립트.
봉인된 data/processed/btc_test.parquet를 개봉하여 최종 out-of-sample 성능을 측정한다.

사용법:
    python scripts/eval_test.py
    python scripts/eval_test.py --config config/exp016_final_config.yaml
    python scripts/eval_test.py --model experiments/exp016_final/best_model
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.env.trading_env import BTCGridTradingEnv
from src.agents.baselines import run_all_baselines
from src.evaluation.metrics import compute_all, print_metrics
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test set 최종 평가")
    parser.add_argument(
        "--config", default="config/exp016_final_config.yaml",
        help="실험 설정 파일 (기본: config/exp016_final_config.yaml)"
    )
    parser.add_argument(
        "--model", default="experiments/exp016_final/best_model",
        help="모델 경로 (.zip 제외, 기본: experiments/exp016_final/best_model)"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=1,
        help="평가 에피소드 수 (기본 1 = 전체 test 단일 실행)"
    )
    return parser.parse_args()


def evaluate_full(model: PPO, df: pd.DataFrame, config: dict) -> dict:
    """Test set 전체를 단일 에피소드로 실행 (random_start=False, ep_steps=None)."""
    initial_cash = config["environment"]["initial_cash"]

    eval_cfg = deepcopy(config)
    eval_cfg["environment"]["random_start"] = False

    raw_env = BTCGridTradingEnv(df, eval_cfg)
    obs, _ = raw_env.reset()
    equity_list = [initial_cash]
    done = False

    while not done:
        action, _ = model.predict(obs[None, :], deterministic=True)
        obs, _reward, terminated, truncated, _info = raw_env.step(action[0])
        done = terminated or truncated

        price = float(raw_env.df.loc[raw_env.current_step - 1, "close"])
        equity_list.append(raw_env.cash + raw_env.holdings * price)

    equity_curve = pd.Series(equity_list, dtype=float)
    metrics = compute_all(
        equity_curve=equity_curve,
        initial_cash=initial_cash,
        n_trades=raw_env.n_trades,
        completed_cycles=list(raw_env.completed_cycles),
    )
    metrics["equity_curve"] = equity_curve
    metrics["completed_cycles"] = list(raw_env.completed_cycles)
    return metrics


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Phase 2-B: Test Set 최종 평가 (2024.01~)")
    print("=" * 60)

    # ── 1. 설정 + 데이터 로드 ──────────────────────────────────
    config = load_config(args.config)

    print("\n[1] 데이터 로드 중...")
    df_train = pd.read_parquet("data/processed/btc_train.parquet")
    df_test  = pd.read_parquet("data/processed/btc_test.parquet")
    print(f"  Train : {len(df_train):,}행 (VecNormalize 통계 없음 → 미사용)")
    print(f"  Test  : {len(df_test):,}행  ({df_test.index[0]} ~ {df_test.index[-1]})")

    # ── 2. 모델 로드 ───────────────────────────────────────────
    model_path = args.model
    print(f"\n[2] 모델 로드: {model_path}.zip")
    model = PPO.load(model_path)
    print("  OK")

    # ── 3. PPO Test 평가 ───────────────────────────────────────
    print(f"\n[3] PPO Test 평가 (전체 단일 에피소드)...")
    ppo_metrics = evaluate_full(model, df_test, config)
    print_metrics(ppo_metrics, label="PPO best_model -- TEST SET")

    # ── 4. 베이스라인 Test 평가 ────────────────────────────────
    print("\n[4] 베이스라인 Test 평가...")
    # config의 날짜 범위를 test 기간으로 고정 (baselines는 df만 사용하므로 무관)
    bl_results = run_all_baselines(df_test, config)
    initial_cash = config["environment"]["initial_cash"]

    # ── 5. 비교 출력 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("베이스라인 비교 -- TEST SET (2024.01~)")
    print("=" * 60)
    print(f"\n{'전략':<22} {'수익률(%)':>9} {'Sharpe':>7} {'MDD(%)':>8} {'거래':>6} {'사이클':>6}")
    print("-" * 64)

    bl_sharpes = {}
    for name, r in bl_results.items():
        m = compute_all(r["equity_curve"], initial_cash, r["n_trades"], r["completed_cycles"])
        bl_sharpes[name] = m["sharpe_ratio"]
        print(f"{name:<22} {m['total_return_pct']:>9.2f} {m['sharpe_ratio']:>7.3f} "
              f"{m['max_drawdown_pct']:>8.2f} {m['n_trades']:>6} {m['n_cycles']:>6}")

    print(f"\n{'PPO (best_model)':<22} {ppo_metrics['total_return_pct']:>9.2f} "
          f"{ppo_metrics['sharpe_ratio']:>7.3f} {ppo_metrics['max_drawdown_pct']:>8.2f} "
          f"{ppo_metrics['n_trades']:>6} {ppo_metrics['n_cycles']:>6}")

    best_bl_name   = max(bl_sharpes, key=bl_sharpes.get)
    best_bl_sharpe = bl_sharpes[best_bl_name]
    ppo_sharpe     = ppo_metrics["sharpe_ratio"]

    print(f"\n→ PPO Sharpe {ppo_sharpe:.3f} vs 베이스라인 최고 {best_bl_sharpe:.3f} "
          f"({best_bl_name})  "
          f"({'[우위]' if ppo_sharpe > best_bl_sharpe else '[미달]'})")

    # ── 6. 결과 저장 ───────────────────────────────────────────
    out_dir = Path("experiments/exp016_final")
    result = {
        "dataset": "test",
        "period": f"{df_test.index[0]} ~ {df_test.index[-1]}",
        "ppo_sharpe":     round(ppo_sharpe, 4),
        "ppo_return_pct": round(ppo_metrics["total_return_pct"], 2),
        "ppo_mdd_pct":    round(ppo_metrics["max_drawdown_pct"], 2),
        "ppo_n_trades":   ppo_metrics["n_trades"],
        "ppo_n_cycles":   ppo_metrics["n_cycles"],
        "best_baseline":  best_bl_name,
        "best_bl_sharpe": round(best_bl_sharpe, 4),
    }

    import yaml
    result_path = out_dir / "test_eval_results.yaml"
    with open(result_path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, allow_unicode=True, default_flow_style=False)
    print(f"\n결과 저장: {result_path}")

    # equity curve 저장 (노트북 시각화용)
    eq_path = out_dir / "test_equity_curve.csv"
    ppo_metrics["equity_curve"].to_csv(eq_path, index=False, header=["equity"])
    print(f"Equity curve 저장: {eq_path}")

    print("\n" + "=" * 60)
    print("Phase 2-B 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
