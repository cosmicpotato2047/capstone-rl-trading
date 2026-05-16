"""
scripts/run_exp034_cpcv.py

exp034 — CPCV (Combinatorial Purged Cross-Validation, López de Prado 2018).

설계:
    Train + Val 데이터 (2017-10 ~ 2023-12, 6.2년) 를 시간순 6 groups 로 분할.
    각 split: 4 groups train (~36k rows ≈ 4.1y), 2 groups test (~18k rows ≈ 2y).
    총 C(6, 2) = 15 splits.
    Purge: train 의 test-인접 168 시간 (warmup) 제거 to avoid leakage.

학습:
    4 variants (sym/asym/dsr/pt) × 15 paths × seed=42 = 60 runs × 1M steps.
    각 run 의 best Val Sharpe (= test partition Sharpe) 기록.

산출:
    experiments/exp034_summary.csv
    columns: variant, path_id, test_groups, train_n, test_n,
             best_sharpe, best_return_pct, best_mdd_pct, best_n_trades, wallclock_sec
"""
import argparse
import csv
import sys
import time
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.agents.ppo_agent import PPOAgent


VARIANTS = ["sym", "asym", "dsr", "pt"]
N_GROUPS = 6
N_TEST_GROUPS = 2
WARMUP = 168            # 1주 ATR window, purge buffer
SEED_DEFAULT = 42
TIMESTEPS = 1_000_000

CSV_HEADER = [
    "variant", "path_id", "test_groups", "train_n", "test_n",
    "best_sharpe", "best_return_pct", "best_mdd_pct",
    "best_n_trades", "best_n_cycles",
    "wallclock_sec",
]


def make_cpcv_splits(df: pd.DataFrame, n_groups: int = N_GROUPS,
                     n_test: int = N_TEST_GROUPS, warmup: int = WARMUP):
    """시간순 데이터를 n_groups 로 분할, C(n_groups, n_test) 개 splits 생성.

    각 split 에서:
      - test_groups 의 indices 를 모아 test partition 으로 만듦
      - 나머지 groups 의 indices 를 모아 train partition 으로 만듦
      - Purge: train rows 중 test-인접 (warmup hours 내) 제거

    Returns:
        splits: list of (path_id, test_group_ids, train_idx, test_idx)
    """
    n = len(df)
    g = n // n_groups
    boundaries = [i * g for i in range(n_groups)] + [n]  # 7 boundaries → 6 groups

    splits = []
    for path_id, test_groups in enumerate(combinations(range(n_groups), n_test)):
        test_idx = []
        train_idx = []
        for gid in range(n_groups):
            start, end = boundaries[gid], boundaries[gid + 1]
            if gid in test_groups:
                test_idx.extend(range(start, end))
            else:
                train_idx.extend(range(start, end))

        # Purge: test 와 인접한 train rows 제거
        # test_idx 의 boundaries (start, end) 양 옆 ±warmup 시간을 train 에서 제외
        test_set = set(test_idx)
        # train index 가 test boundary 와 인접한지 검사
        purged_train = []
        # test_idx 가 contiguous range 들로 구성됨; 그 boundaries 추출
        test_ranges = []
        if test_idx:
            cur_start = test_idx[0]
            for i in range(1, len(test_idx)):
                if test_idx[i] != test_idx[i-1] + 1:
                    test_ranges.append((cur_start, test_idx[i-1] + 1))
                    cur_start = test_idx[i]
            test_ranges.append((cur_start, test_idx[-1] + 1))

        # purge zones (train 에서 제외): test range 양 옆 ±warmup
        purge_zones = []
        for ts, te in test_ranges:
            purge_zones.append((max(0, ts - warmup), ts))   # before test
            purge_zones.append((te, min(n, te + warmup)))   # after test

        for i in train_idx:
            in_purge = False
            for pz_start, pz_end in purge_zones:
                if pz_start <= i < pz_end:
                    in_purge = True
                    break
            if not in_purge:
                purged_train.append(i)

        splits.append({
            "path_id":      path_id,
            "test_groups":  list(test_groups),
            "train_idx":    purged_train,
            "test_idx":     test_idx,
        })
    return splits


def _append_csv(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_one_path(variant: str, split: dict, df_all: pd.DataFrame,
                 base_cfg: dict, timesteps: int, csv_path: Path,
                 seed: int = SEED_DEFAULT) -> dict:
    cfg = deepcopy(base_cfg)
    path_id = split["path_id"]
    log_dir = Path(f"experiments/exp034_cpcv/{variant}/path_{path_id:02d}")
    cfg["training"]["seed"] = seed
    cfg["training"]["log_dir"] = str(log_dir)
    cfg["training"]["experiment_name"] = f"exp034_{variant}_path{path_id:02d}"

    summary_path = log_dir / "summary.yaml"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            existing = yaml.safe_load(f)
        print(f"  skip {variant} path={path_id}: sharpe {existing.get('best_sharpe', '?')}")
        return existing

    log_dir.mkdir(parents=True, exist_ok=True)

    df_train = df_all.iloc[split["train_idx"]].reset_index(drop=True)
    df_test  = df_all.iloc[split["test_idx"]].reset_index(drop=True)

    t0 = time.time()
    agent = PPOAgent(cfg, df_train, df_test)
    best_path = str(log_dir / "best_model")
    agent.train(total_timesteps=timesteps, best_model_path=best_path)

    # best 모델 로드 후 test 평가
    best_agent = PPOAgent.load(best_path, cfg, df_train, df_test)
    test_metrics = best_agent.evaluate(df_test)

    summary = {
        "variant":         variant,
        "path_id":         path_id,
        "test_groups":     str(split["test_groups"]),
        "train_n":         len(split["train_idx"]),
        "test_n":          len(split["test_idx"]),
        "best_sharpe":     float(test_metrics["sharpe_ratio"]),
        "best_return_pct": float(test_metrics["total_return_pct"]),
        "best_mdd_pct":    float(test_metrics["max_drawdown_pct"]),
        "best_n_trades":   int(test_metrics["n_trades"]),
        "best_n_cycles":   int(test_metrics["n_cycles"]),
        "wallclock_sec":   round(time.time() - t0, 1),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)
    _append_csv(csv_path, summary)

    print(f"  [done] {variant} path={path_id:02d} (test_groups={split['test_groups']}): "
          f"Sharpe {summary['best_sharpe']:+.3f}, return {summary['best_return_pct']:+.2f}%, "
          f"MDD {summary['best_mdd_pct']:.2f}% - {summary['wallclock_sec']:.0f}s")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--config-tmpl", default="config/exp032b_{variant}_config.yaml")
    p.add_argument("--csv", default="experiments/exp034_summary.csv")
    p.add_argument("--paths", nargs="+", type=int, default=None,
                   help="path_id 일부만 (default: 모두 0..14)")
    args = p.parse_args()

    csv_path = Path(args.csv)

    # Train + Val 통합
    print("Loading data...")
    train = pd.read_parquet("data/processed/btc_train.parquet")
    val   = pd.read_parquet("data/processed/btc_val.parquet")
    df_all = pd.concat([train, val], ignore_index=True)
    print(f"  Train+Val: {len(df_all):,} rows")

    print(f"Making CPCV splits ({N_GROUPS} groups, {N_TEST_GROUPS} test groups, "
          f"purge ±{WARMUP}h)...")
    splits = make_cpcv_splits(df_all)
    print(f"  {len(splits)} paths created")
    for s in splits:
        print(f"    path {s['path_id']:02d}: test_groups={s['test_groups']}, "
              f"train_n={len(s['train_idx']):,}, test_n={len(s['test_idx']):,}")

    selected_paths = args.paths if args.paths is not None else list(range(len(splits)))

    print(f"\nexp034 CPCV: {len(args.variants)} variants × {len(selected_paths)} paths "
          f"= {len(args.variants)*len(selected_paths)} runs × {args.timesteps:,} steps")
    print(f"  csv: {csv_path}\n")

    t_start = time.time()
    n_total = len(args.variants) * len(selected_paths)
    n_done = 0
    for variant in args.variants:
        base_cfg = load_config(args.config_tmpl.format(variant=variant))
        print(f"\n=== Variant: {variant} ===")
        for path_id in selected_paths:
            n_done += 1
            print(f"\n[{n_done}/{n_total}] {variant} path={path_id}")
            try:
                run_one_path(variant, splits[path_id], df_all, base_cfg,
                             args.timesteps, csv_path, seed=args.seed)
            except Exception as e:
                print(f"  ERROR {variant} path={path_id} - {e}")
                import traceback; traceback.print_exc()

    total = time.time() - t_start
    print(f"\n{'='*60}\nCompleted {n_done}/{n_total} runs in {total/60:.1f} min")
    print(f"summary CSV: {csv_path}\n{'='*60}")


if __name__ == "__main__":
    main()
