"""
exp014a_v2, exp014b_v2 (n_buy_orders 신설계), exp015 (ent_coef=0.1) 순차 실행.
"""
import subprocess, sys, time, os, re
import yaml
from pathlib import Path

CONFIGS = [
    "config/exp014a_v2_config.yaml",   # n_buy_orders=3 (신설계: 가격 레벨 수)
    "config/exp014b_v2_config.yaml",   # n_buy_orders=4 (신설계)
    "config/exp015_ent_01_config.yaml", # ent_coef=0.1
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
        subprocess.run(
            [sys.executable, "scripts/train_ppo.py",
             "--config", config_path, "--no-mlflow"],
            stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent)
        )

    elapsed = time.time() - start
    best_sharpe = get_best_sharpe(log_file)
    results.append((exp_name, best_sharpe, elapsed / 60))
    print(f"  완료: {elapsed/60:.1f}분 | best Sharpe = {best_sharpe:.3f}")

print(f"\n{'='*60}")
print(f"전체 완료: {(time.time()-total_start)/60:.1f}분")
print(f"{'='*60}")
print(f"{'실험명':<30} {'best Sharpe':>12} {'소요(분)':>10}")
print("-" * 55)
for name, sharpe, mins in results:
    marker = " ← best" if sharpe == max(r[1] for r in results) else ""
    print(f"{name:<30} {sharpe:>12.3f} {mins:>10.1f}{marker}")
print(f"\n[베이스라인] exp013b (n_buy=2, avg_price): 17.579")
