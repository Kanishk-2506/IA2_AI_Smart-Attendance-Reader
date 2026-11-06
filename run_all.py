"""
run_all.py
Runs every task script in order. Task 5 must come last because it reads
the result files that tasks 2, 3 and 4 create.
"""

import os
import subprocess
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
scripts = ["task1_pca.py", "task2_kmeans.py", "task3_knn.py", "task4_ann.py", "task5_comparison.py"]

for script in scripts:
    print(f"\n===== Running {script} =====")
    # sys.executable = the same Python that is running this file (so the venv is used).
    result = subprocess.run([sys.executable, os.path.join(base_dir, script)], cwd=base_dir)
    if result.returncode != 0:
        print(f"ERROR: {script} failed. Stopping.")
        sys.exit(1)

print("\nAll tasks finished. See the figures/ and results/ folders.")
