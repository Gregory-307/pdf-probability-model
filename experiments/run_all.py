"""
Run all experiments and generate comprehensive summary.

Usage:
    python experiments/run_all.py

This will run:
1. 3% barrier test (quick comparison)
2. 32 feature test (does more features help?)
3. Longer horizons test (does P2 win at longer horizons?)
4. Full experiment suite (optional, slow)
"""

import subprocess
import sys
import os

# Change to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

experiments = [
    ("3% Barrier Test", "experiments/test_3pct_barrier.py"),
    ("32 Features Test", "experiments/test_32_features.py"),
    ("Longer Horizons Test", "experiments/test_longer_horizons.py"),
]

print("=" * 70)
print("TEMPORALPDF EXPERIMENT SUITE")
print("Pipeline 1 (XGBoost Classifier) vs Pipeline 2 (Distribution Params)")
print("=" * 70)

for name, script in experiments:
    print(f"\n{'='*70}")
    print(f"RUNNING: {name}")
    print(f"Script: {script}")
    print("=" * 70)

    result = subprocess.run([sys.executable, script], capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: {name} failed with return code {result.returncode}")
    else:
        print(f"\n{name} completed successfully")

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
print("""
Check the following output files:
- experiments/test_3pct_barrier.png
- experiments/test_32_features.png
- experiments/horizon_comparison.png
- experiments/horizon_comparison_results.csv

To run the full parameter sweep (slow):
    python experiments/barrier_experiment.py
""")
