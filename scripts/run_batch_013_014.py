"""
exp013a/b (threshold 비교) + exp014a/b (n_buy_orders 비교) 순차 실행.
"""
import subprocess, sys, time, os, re
import yaml
from pathlib import Path

CONFIGS = [
    "config/exp013a_config.yaml",   # threshold=price
    "config/exp013b_config.yaml",   # threshold=avg_price
    "config/exp014a_config.yaml",   # n_buy_orders=3
    "config/exp014b_config.yaml",   # n_buy_orders=4
]

def get_best_sharpe(log_path: str) -> float:
    best = -999.0
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(r'Sharpe=([+-]?\d+\.\d+)', line)
                if m:
                    best = max(best, float(m.group(1)))
    except FileNotFoundError:
        pass
    return best

results = []
total_start = time.time()

for i, config_path in enumerate(CONFIGS, 1):
    exp_name = Path(config_path).stem.replace("_config", "")
    print(f"\n{'='*60}")
    print(f"[{i}/{len(CONFIGS)}] {exp_name} 시작 ({time.strftime('%H:%M:%S')})")
    print(f"{'='*60}")

    start = time.time()

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log_dir = cfg["training"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log.txt")

    with open(log_file, "w") as lf:
        proc = subprocess.run(
            [sys.executable, "scripts/train_ppo.py",
             "--config", config_path, "--no-mlflow"],
            stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent)
        )

    elapsed = time.time() - start
    best_sharpe = get_best_sharpe(log_file)
    results.append((exp_name, best_sharpe, elapsed / 60))
    print(f"  완료: {elapsed/60:.1f}분 | best Sharpe = {best_sharpe:.3f}")

# 요약
print(f"\n{'='*60}")
print(f"전체 완료: {(time.time()-total_start)/60:.1f}분")
print(f"{'='*60}")
print(f"{'실험명':<32} {'best Sharpe':>12} {'소요(분)':>10}")
print("-" * 57)
for name, sharpe, mins in results:
    marker = " ← best" if sharpe == max(r[1] for r in results) else ""
    print(f"{name:<32} {sharpe:>12.3f} {mins:>10.1f}{marker}")
print(f"\n[exp009a 베이스라인 Sharpe: 16.826]")
