"""
scripts/run_exp035_test.py

exp035 — Test 봉인 해제 (1회만). Final out-of-sample evaluation.

평가 대상:
    1. exp032b 40 models (4 variants x 10 seeds, Val 2021-2023 best)
    2. exp034 60 models (4 variants x 15 CPCV paths)
    3. ATR baseline (Bayesian best Trial #34, fixed action [0,0])

각 모델 -> df_test (2024-01 ~ , 20,189 rows) evaluate -> Test Sharpe / Return / MDD / Trades / Cycles.

산출:
    experiments/exp035_summary.csv
    columns: source (exp032b/exp034/atr_baseline), variant, seed_or_path,
             test_sharpe, test_return_pct, test_mdd_pct, test_n_trades, test_n_cycles
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.env.trading_env import BTCGridTradingEnv
from src.agents.ppo_agent import PPOAgent
from src.evaluation.metrics import compute_all

VARIANTS = ["sym", "asym", "dsr", "pt"]
SEEDS_EXP032B = list(range(42, 52))    # 10 seeds
PATHS_EXP034  = list(range(15))         # 15 paths

CSV_HEADER = [
    "source", "variant", "seed_or_path",
    "test_sharpe", "test_return_pct", "test_mdd_pct",
    "test_n_trades", "test_n_cycles",
]


def _append(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if is_new:
            w.writeheader()
        w.writerow(row)


def eval_ppo_model(model_path: str, cfg_path: str, df_test: pd.DataFrame,
                   df_train: pd.DataFrame) -> dict:
    """exp032b/exp034 best_model 로드 + df_test 평가."""
    cfg = load_config(cfg_path)
    cfg["environment"]["random_start"] = False
    cfg["training"]["max_episode_steps"] = None
    agent = PPOAgent.load(model_path, cfg, df_train, df_test)
    metrics = agent.evaluate(df_test, n_episodes=1)
    return metrics


def eval_atr_baseline(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """ATR Bayesian best baseline (formula_coefs Trial #34) 을 Test 위에서 평가.

    Trading env 에서 action=[0, 0] 으로 고정 시:
        buy_hi_gap   = atr_ratio * (A_b + B_b * 0) = A_b * atr_ratio
        buy_lo_gap   = atr_ratio * (C_b + D_b * 0) = C_b * atr_ratio
        sell_mkt_gap = atr_ratio * (A_s + B_s * 0) = A_s * atr_ratio
        sell_cost_gap= atr_ratio * (C_s + D_s * 0) = C_s * atr_ratio
    → ATR Bayesian best 의 동작과 동일.
    """
    cfg = load_config("config/exp032b_sym_config.yaml")
    cfg["environment"]["random_start"] = False
    cfg["environment"]["slippage_rate"] = 0.0
    cfg["environment"]["reward_type"] = "sym"
    # max_episode_steps 제거하여 전체 episode
    cfg["training"]["max_episode_steps"] = None

    env = BTCGridTradingEnv(df_test, cfg)
    obs, _ = env.reset()
    equity_list = [cfg["environment"]["initial_cash"]]
    done = False
    while not done:
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        equity_list.append(info["equity"])
        done = terminated or truncated

    metrics = compute_all(
        equity_curve=pd.Series(equity_list),
        initial_cash=cfg["environment"]["initial_cash"],
        n_trades=env.n_trades,
        completed_cycles=env.completed_cycles,
    )
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="experiments/exp035_summary.csv")
    p.add_argument("--sources", nargs="+", default=["atr_baseline", "exp032b", "exp034"],
                   choices=["atr_baseline", "exp032b", "exp034"])
    args = p.parse_args()

    csv_path = Path(args.csv)
    if csv_path.exists():
        print(f"WARNING: {csv_path} already exists. Test 평가는 1회만 권장 (CLAUDE.md 규칙).")
        print("기존 결과 위에 추가됩니다.")

    print("Loading data...")
    df_train = pd.read_parquet("data/processed/btc_train.parquet")
    df_test  = pd.read_parquet("data/processed/btc_test.parquet")
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}\n")

    t_start = time.time()

    # 1. ATR baseline
    if "atr_baseline" in args.sources:
        print("=== ATR Baseline (Bayesian Trial #34, fixed action [0,0]) ===")
        try:
            m = eval_atr_baseline(df_train, df_test)
            row = {
                "source": "atr_baseline", "variant": "atr", "seed_or_path": "-",
                "test_sharpe": float(m["sharpe_ratio"]),
                "test_return_pct": float(m["total_return_pct"]),
                "test_mdd_pct": float(m["max_drawdown_pct"]),
                "test_n_trades": int(m["n_trades"]),
                "test_n_cycles": int(m["n_cycles"]),
            }
            _append(csv_path, row)
            print(f"  ATR baseline Test: Sharpe {row['test_sharpe']:+.3f}, "
                  f"Return {row['test_return_pct']:+.2f}%, MDD {row['test_mdd_pct']:.2f}%, "
                  f"Trades {row['test_n_trades']}, Cycles {row['test_n_cycles']}")
        except Exception as e:
            print(f"  ERROR: ATR baseline failed - {e}")
            import traceback; traceback.print_exc()

    # 2. exp032b 40 models
    if "exp032b" in args.sources:
        print("\n=== exp032b 40 models (Val 2021-2023 best -> Test 2024+) ===")
        for v in VARIANTS:
            cfg_path = f"config/exp032b_{v}_config.yaml"
            for seed in SEEDS_EXP032B:
                mp = f"experiments/exp032b_{v}/seed_{seed}/best_model"
                if not Path(mp + ".zip").exists():
                    print(f"  skip {v} seed={seed}: model not found")
                    continue
                try:
                    m = eval_ppo_model(mp, cfg_path, df_test, df_train)
                    row = {
                        "source": "exp032b", "variant": v, "seed_or_path": str(seed),
                        "test_sharpe": float(m["sharpe_ratio"]),
                        "test_return_pct": float(m["total_return_pct"]),
                        "test_mdd_pct": float(m["max_drawdown_pct"]),
                        "test_n_trades": int(m["n_trades"]),
                        "test_n_cycles": int(m["n_cycles"]),
                    }
                    _append(csv_path, row)
                    print(f"  exp032b {v} seed={seed}: Sharpe {row['test_sharpe']:+.3f}, "
                          f"Return {row['test_return_pct']:+.2f}%, MDD {row['test_mdd_pct']:.2f}%")
                except Exception as e:
                    print(f"  ERROR {v} seed={seed}: {e}")

    # 3. exp034 60 models
    if "exp034" in args.sources:
        print("\n=== exp034 60 models (CPCV paths best -> Test 2024+) ===")
        for v in VARIANTS:
            cfg_path = f"config/exp032b_{v}_config.yaml"   # exp034 used same configs
            for pid in PATHS_EXP034:
                mp = f"experiments/exp034_cpcv/{v}/path_{pid:02d}/best_model"
                if not Path(mp + ".zip").exists():
                    print(f"  skip {v} path={pid}: model not found")
                    continue
                try:
                    m = eval_ppo_model(mp, cfg_path, df_test, df_train)
                    row = {
                        "source": "exp034", "variant": v, "seed_or_path": f"p{pid:02d}",
                        "test_sharpe": float(m["sharpe_ratio"]),
                        "test_return_pct": float(m["total_return_pct"]),
                        "test_mdd_pct": float(m["max_drawdown_pct"]),
                        "test_n_trades": int(m["n_trades"]),
                        "test_n_cycles": int(m["n_cycles"]),
                    }
                    _append(csv_path, row)
                    print(f"  exp034 {v} path={pid:02d}: Sharpe {row['test_sharpe']:+.3f}, "
                          f"Return {row['test_return_pct']:+.2f}%, MDD {row['test_mdd_pct']:.2f}%")
                except Exception as e:
                    print(f"  ERROR {v} path={pid}: {e}")

    total = time.time() - t_start
    print(f"\n{'='*60}\nTotal: {total/60:.1f} min")
    print(f"summary -> {csv_path}\n{'='*60}")


if __name__ == "__main__":
    main()
