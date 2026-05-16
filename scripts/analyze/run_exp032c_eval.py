"""
scripts/run_exp032c_eval.py

exp032c Step 1: 40 모델 (4 variants x 10 seeds) 을 Val 위에서 평가하며
                step-level trajectory 를 parquet 으로 저장.

각 step row:
    variant, seed, step_idx, val_idx (df_eval 의 원본 index),
    state_0..6 (7D obs), action_0, action_1 (raw [0,1]),
    reward, equity, cash, holdings, n_trades,
    price (df_eval close), atr_ratio, trend_short, trend_long

산출: experiments/exp032c_trajectories.parquet
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.env.trading_env import BTCGridTradingEnv
from src.agents.ppo_agent import PPOAgent


VARIANTS = ["sym", "asym", "dsr", "pt"]
SEEDS    = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def eval_one(variant: str, seed: int, df_train: pd.DataFrame, df_eval: pd.DataFrame,
             best_or_final: str = "best") -> pd.DataFrame:
    """단일 (variant, seed) 의 best_model 을 df_eval (Val 또는 Test) 전체 단일 episode 로
    평가하며 step trajectory 를 DataFrame 으로 반환."""
    cfg_path = f"config/exp032b_{variant}_config.yaml"
    cfg = load_config(cfg_path)

    # deterministic 평가: random_start=False, max_episode_steps=None
    cfg["environment"]["random_start"] = False
    cfg["training"]["max_episode_steps"] = None

    model_path = f"experiments/exp032b_{variant}/seed_{seed}/{best_or_final}_model"
    agent = PPOAgent.load(model_path, cfg, df_train, df_eval)
    model = agent.model

    # 평가용 env (df_eval 사용 - Val 또는 Test)
    env = BTCGridTradingEnv(df_eval, cfg)

    rows = []
    obs, _ = env.reset()
    done = False
    step_idx = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # SB3 returns array; normalize to 1d
        a = np.asarray(action).reshape(-1)
        prev_step = env.current_step
        obs_next, reward, terminated, truncated, info = env.step(a)

        # df_eval 원본 인덱스 (prev_step 의 row 정보)
        row = {
            "variant": variant,
            "seed":    seed,
            "step_idx": step_idx,
            "val_idx": int(prev_step),
        }
        for i in range(len(obs)):
            row[f"state_{i}"] = float(obs[i])
        row["action_0"] = float(a[0])
        row["action_1"] = float(a[1])
        row["reward"]   = float(reward)
        row["equity"]   = float(info["equity"])
        row["cash"]     = float(info["cash"])
        row["holdings"] = float(info["holdings"])
        row["n_trades"] = int(info["n_trades"])
        # market context (df_eval 의 원본 raw 값 — z-score 이전)
        if "volatility_raw" in df_eval.columns:
            row["atr_ratio"] = float(df_eval.iloc[prev_step]["volatility_raw"])
        else:
            row["atr_ratio"] = float("nan")
        for col in ["trend_short", "trend_long", "close"]:
            if col in df_eval.columns:
                row[col] = float(df_eval.iloc[prev_step][col])

        rows.append(row)
        obs = obs_next
        done = terminated or truncated
        step_idx += 1

    df = pd.DataFrame(rows)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--model", choices=["best", "final"], default="best",
                   help="best_model.zip 또는 final_model.zip 사용")
    p.add_argument("--out", default="experiments/exp032c_trajectories.parquet")
    p.add_argument("--eval-data", default="data/processed/btc_val.parquet",
                   help="평가용 parquet (default: btc_val). Test 위 평가는 btc_test.parquet")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"exp032c eval: {len(args.variants)} variants x {len(args.seeds)} seeds "
          f"= {len(args.variants)*len(args.seeds)} models ({args.model})")
    print(f"  out: {out_path}\n")

    df_train = pd.read_parquet("data/processed/btc_train.parquet")
    df_eval  = pd.read_parquet(args.eval_data)
    print(f"  train: {len(df_train):,}rows | eval ({args.eval_data}): {len(df_eval):,}rows\n")

    t0 = time.time()
    all_dfs = []
    n_total = len(args.variants) * len(args.seeds)
    n_done = 0
    for variant in args.variants:
        for seed in args.seeds:
            n_done += 1
            tt = time.time()
            try:
                df = eval_one(variant, seed, df_train, df_eval, args.model)
                all_dfs.append(df)
                print(f"  [{n_done}/{n_total}] {variant} seed={seed}: "
                      f"{len(df):,} steps, final eq {df['equity'].iloc[-1]:,.0f} "
                      f"- {time.time()-tt:.1f}s")
            except Exception as e:
                print(f"  ERROR: {variant} seed={seed} - {e}")
                import traceback; traceback.print_exc()

    if not all_dfs:
        print("no trajectories collected, abort.")
        sys.exit(1)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_parquet(out_path, index=False)
    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"completed {n_done} models in {total/60:.1f} min")
    print(f"trajectories: {len(df_all):,} rows -> {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
