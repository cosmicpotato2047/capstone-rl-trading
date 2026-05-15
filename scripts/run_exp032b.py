"""
scripts/run_exp032b.py

exp032b - 4 reward variant x 10 seeds 비교 학습 (본 논문 §5 메인 비교).

각 run:
    config/exp032b_{variant}_config.yaml 로드
    training.seed = seed (override)
    training.log_dir = experiments/exp032b_{variant}/seed_{seed}
    training.experiment_name = exp032b_{variant}_seed{seed}
    PPOAgent.train(1M) → save best + final
    best_model 로드 → evaluate → best_metrics
    final_model evaluate → final_metrics
    summary.yaml 저장 + 글로벌 summary.csv append

CSV columns:
    variant, seed, best_val_sharpe, best_return_pct, best_mdd_pct,
    best_n_trades, best_n_cycles, final_val_sharpe, final_return_pct,
    final_mdd_pct, final_n_trades, final_n_cycles, wallclock_sec

사용법:
    python scripts/run_exp032b.py
    python scripts/run_exp032b.py --variants asym dsr     # 일부만
    python scripts/run_exp032b.py --seeds 42 43 44        # 일부만
    python scripts/run_exp032b.py --timesteps 100000      # 짧게 (sanity)
"""
import argparse
import csv
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.agents.ppo_agent import PPOAgent


VARIANTS_DEFAULT = ["sym", "asym", "dsr", "pt"]
SEEDS_DEFAULT    = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

CSV_HEADER = [
    "variant", "seed",
    "best_val_sharpe", "best_return_pct", "best_mdd_pct",
    "best_n_trades", "best_n_cycles",
    "final_val_sharpe", "final_return_pct", "final_mdd_pct",
    "final_n_trades", "final_n_cycles",
    "wallclock_sec",
]


def _append_csv(csv_path: Path, row: dict) -> None:
    """summary CSV 에 한 줄 append. 없으면 header 와 함께 신설."""
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_one(variant: str, seed: int, base_cfg: dict,
            df_train: pd.DataFrame, df_val: pd.DataFrame,
            timesteps: int, csv_path: Path) -> dict:
    """단일 (variant, seed) 학습 + 평가 → summary dict."""
    cfg = deepcopy(base_cfg)
    log_dir = Path(f"experiments/exp032b_{variant}/seed_{seed}")
    cfg["training"]["seed"]            = seed
    cfg["training"]["log_dir"]         = str(log_dir)
    cfg["training"]["experiment_name"] = f"exp032b_{variant}_seed{seed}"

    summary_path = log_dir / "summary.yaml"
    if summary_path.exists():
        # 재실행 회피 (resumable)
        with open(summary_path, encoding="utf-8") as f:
            existing = yaml.safe_load(f)
        print(f"  skip (already done): {variant} seed={seed} → "
              f"best Sharpe {existing.get('best_val_sharpe', '?'):.3f}")
        return existing

    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_path  = str(log_dir / "best_model")
    final_model_path = str(log_dir / "final_model")

    t0 = time.time()
    agent = PPOAgent(cfg, df_train, df_val)
    agent.train(total_timesteps=timesteps, best_model_path=best_model_path)
    agent.save(final_model_path)

    # final metrics
    final_metrics = agent.evaluate(df_val)
    # best model 재로드 + 평가
    best_agent = PPOAgent.load(best_model_path, cfg, df_train, df_val)
    best_metrics = best_agent.evaluate(df_val)

    wallclock = time.time() - t0

    summary = {
        "variant":          variant,
        "seed":             seed,
        "best_val_sharpe":  float(best_metrics["sharpe_ratio"]),
        "best_return_pct":  float(best_metrics["total_return_pct"]),
        "best_mdd_pct":     float(best_metrics["max_drawdown_pct"]),
        "best_n_trades":    int(best_metrics["n_trades"]),
        "best_n_cycles":    int(best_metrics["n_cycles"]),
        "final_val_sharpe": float(final_metrics["sharpe_ratio"]),
        "final_return_pct": float(final_metrics["total_return_pct"]),
        "final_mdd_pct":    float(final_metrics["max_drawdown_pct"]),
        "final_n_trades":   int(final_metrics["n_trades"]),
        "final_n_cycles":   int(final_metrics["n_cycles"]),
        "wallclock_sec":    round(wallclock, 1),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)
    _append_csv(csv_path, summary)

    print(f"  [done] {variant} seed={seed}: "
          f"best Sharpe {summary['best_val_sharpe']:+.3f} (return {summary['best_return_pct']:+.2f}%, "
          f"MDD {summary['best_mdd_pct']:.2f}%) - {wallclock:.0f}s")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=VARIANTS_DEFAULT,
                   choices=VARIANTS_DEFAULT, help=f"학습할 variant (default: {VARIANTS_DEFAULT})")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT,
                   help=f"사용할 seed (default: {SEEDS_DEFAULT})")
    p.add_argument("--timesteps", type=int, default=1_000_000,
                   help="run 당 학습 step (default 1M)")
    p.add_argument("--csv", default="experiments/exp032b_summary.csv",
                   help="summary CSV 경로")
    p.add_argument("--config-tmpl", default="config/exp032b_{variant}_config.yaml",
                   help="variant 별 config 경로 템플릿")
    args = p.parse_args()

    csv_path = Path(args.csv)
    print(f"exp032b: {len(args.variants)} variant x {len(args.seeds)} seed = "
          f"{len(args.variants)*len(args.seeds)} runs x {args.timesteps:,} steps")
    print(f"  CSV: {csv_path}")
    print(f"  Variants: {args.variants}")
    print(f"  Seeds:    {args.seeds}\n")

    df_train = pd.read_parquet("data/processed/btc_train.parquet")
    df_val   = pd.read_parquet("data/processed/btc_val.parquet")
    print(f"  Train: {len(df_train):,}행  |  Val: {len(df_val):,}행\n")

    t_start = time.time()
    n_total = len(args.variants) * len(args.seeds)
    n_done = 0
    for variant in args.variants:
        cfg_path = args.config_tmpl.format(variant=variant)
        base_cfg = load_config(cfg_path)
        print(f"\n=== Variant: {variant}  ({cfg_path}) ===")
        for seed in args.seeds:
            n_done += 1
            print(f"\n[{n_done}/{n_total}] {variant} seed={seed}")
            try:
                run_one(variant, seed, base_cfg, df_train, df_val,
                        args.timesteps, csv_path)
            except Exception as e:
                print(f"  ERROR: {variant} seed={seed} 실패 - {e}")
                import traceback; traceback.print_exc()

    total = time.time() - t_start
    print(f"\n{'='*60}\n전체 완료: {n_done}/{n_total} runs in {total/60:.1f} min")
    print(f"{'='*60}")
    print(f"summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
