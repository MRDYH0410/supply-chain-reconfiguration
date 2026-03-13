from __future__ import annotations

import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from experiments_1 import run_sensitivity as exp1
from experiments_2 import run_sensitivity as exp2
from experiments_3 import run_sensitivity as exp3
from experiments_4 import run_sensitivity as exp4


def _run_one(name: str, fn) -> None:
    print("=" * 88)
    print(f"Running {name}")
    print("=" * 88)
    t0 = time.time()
    fn()
    dt = time.time() - t0
    print(f"Finished {name} in {dt:.1f}s\n")


def main() -> None:
    total0 = time.time()
    _run_one("Experiment 1  16 tariff paths", exp1.main)
    _run_one("Experiment 2  ramp-up sensitivity", exp2.main)
    _run_one("Experiment 3  tariff-level sensitivity", exp3.main)
    _run_one("Experiment 4  persistence sensitivity", exp4.main)
    total_dt = time.time() - total0
    print("=" * 88)
    print(f"All experiments completed in {total_dt:.1f}s")
    print("Outputs are saved under the existing outputs/ subfolders.")
    print("=" * 88)


if __name__ == "__main__":
    main()