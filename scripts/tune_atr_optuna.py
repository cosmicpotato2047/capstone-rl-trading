"""
scripts/tune_atr_optuna.py

ATR + direction rule 계수 Bayesian 최적화 (exp027).
eval_atr_test.py의 run_atr_fixed()를 직접 사용. trial당 약 2~5초.

탐색 대상:
    A_b          : buy_hi = price × (1 - A_b × ATR × buy_mult)
    C_b          : buy_lo = price × (1 - C_b × ATR × buy_mult)
    A_s          : sell_market = price     × (1 + A_s × ATR × sell_mult)
    C_s          : sell_cost   = avg_price × (1 + C_s × ATR × sell_mult)
    n_splits     : 현금 분할 수
    trend_window : direction signal 윈도우 (72/168/336/720/1440h 중 선택)
    k            : direction 강도 계수 (0이면 기존 ATR과 동일)

사용법:
    python scripts/tune_atr_optuna.py
    python scripts/tune_atr_optuna.py --trials 150 --exp-name exp027_atr_direction
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import optuna
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from scripts.eval_atr_test import run_atr_fixed

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, df_val: pd.DataFrame, base_cfg: dict) -> float:
    cfg = deepcopy(base_cfg)

    # A_b, A_s 하한 0.05: 1h봉에서 intra-candle 동시 체결 exploit 방지
    A_b          = trial.suggest_float("A_b",          0.05,  2.0)
    C_b          = trial.suggest_float("C_b",          1.0,  20.0)
    A_s          = trial.suggest_float("A_s",          0.05,  1.0)
    C_s          = trial.suggest_float("C_s",          0.5,  10.0)
    n_splits     = trial.suggest_int(  "n_splits",     2,     8)
    trend_window = trial.suggest_categorical("trend_window", [72, 168, 336, 720, 1440])
    k            = trial.suggest_float("k",            0.0,   5.0)

    cfg["environment"]["formula_coefs"]["A_b"] = A_b
    cfg["environment"]["formula_coefs"]["C_b"] = C_b
    cfg["environment"]["formula_coefs"]["A_s"] = A_s
    cfg["environment"]["formula_coefs"]["C_s"] = C_s
    cfg["environment"]["n_splits"]             = n_splits

    try:
        m = run_atr_fixed(df_val, cfg, trend_window=trend_window, k=k)
        sharpe = float(m["sharpe_ratio"])
    except Exception as e:
        print(f"  [trial {trial.number}] 오류: {e}")
        return float("-inf")

    print(f"  [trial {trial.number:3d}] Sharpe={sharpe:+.3f} | "
          f"A_b={A_b:.3f} C_b={C_b:.3f} A_s={A_s:.3f} C_s={C_s:.3f} "
          f"n_splits={n_splits} tw={trend_window} k={k:.3f}")
    return sharpe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials",   type=int, default=150)
    p.add_argument("--exp-name", type=str, default="exp027_atr_direction")
    args = p.parse_args()

    cfg    = load_config()
    df_val = pd.read_parquet("data/processed/btc_val.parquet")

    log_dir = Path("experiments") / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # 현재 config 계수(exp023 결과)로 baseline
    baseline = run_atr_fixed(df_val, cfg)
    baseline_sharpe = float(baseline["sharpe_ratio"])

    coefs = cfg["environment"]["formula_coefs"]
    print(f"ATR 공식 계수 재최적화 (eval_atr_test 직접 시뮬레이션)")
    print(f"  trials={args.trials}  Val: {len(df_val):,}봉")
    print(f"  현재 계수: A_b={coefs['A_b']} C_b={coefs['C_b']} "
          f"A_s={coefs['A_s']} C_s={coefs['C_s']} n_splits={cfg['environment']['n_splits']}")
    print(f"  현재 Val Sharpe: {baseline_sharpe:.3f}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name="atr_direction_v1",
        storage=f"sqlite:///{log_dir}/optuna.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # exp026 결과를 시드로 추가 (k=0 → 기존 ATR과 동일한 출발점)
    study.enqueue_trial({
        "A_b": coefs["A_b"], "C_b": coefs["C_b"],
        "A_s": coefs["A_s"], "C_s": coefs["C_s"],
        "n_splits": cfg["environment"]["n_splits"],
        "trend_window": 720, "k": 0.0,
    })

    study.optimize(
        lambda trial: objective(trial, df_val, cfg),
        n_trials=args.trials,
    )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"최적 Trial #{best.number}  Val Sharpe={best.value:.3f}  "
          f"(기존 {baseline_sharpe:.3f}, 개선 {best.value - baseline_sharpe:+.3f})")
    print(f"{'='*60}")
    for k, v in best.params.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    result = {
        "best_trial":      best.number,
        "best_sharpe":     round(best.value, 4),
        "baseline_sharpe": round(baseline_sharpe, 4),
        "improvement":     round(best.value - baseline_sharpe, 4),
        "best_params":     best.params,
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
        pp = t.params
        print(f"  #{t.number:3d} Sharpe={t.value:.3f} | "
              f"A_b={pp['A_b']:.3f} C_b={pp['C_b']:.3f} "
              f"A_s={pp['A_s']:.3f} C_s={pp['C_s']:.3f} "
              f"n_splits={pp['n_splits']} tw={pp['trend_window']} k={pp['k']:.3f}")

    print(f"\n결과 저장: {out_path}")
    bp = best.params
    print(f"config 반영 값:")
    print(f"  A_b={bp['A_b']:.4f}  C_b={bp['C_b']:.4f}  "
          f"A_s={bp['A_s']:.4f}  C_s={bp['C_s']:.4f}  n_splits={bp['n_splits']}")
    print(f"  trend_window={bp['trend_window']}  k={bp['k']:.4f}")


if __name__ == "__main__":
    main()
