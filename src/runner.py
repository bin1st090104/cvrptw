from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, TypedDict


class Execution(TypedDict):
    cost: int
    status: str
    elapsed_ms: int
    routes: List[List[int]]


ROOT = Path(__file__).parent.parent.resolve()
TESTCASES = ROOT / "testcases"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

EXE_DIR = Path(sys.argv[1]).resolve()
for algorithm in (
    "branch_and_bound",
    "local_search",
    "recursion",
):
    with open(RESULTS / f"{algorithm}.txt", "w", encoding="utf-8") as writer:
        if sys.platform == "win32":
            algorithm += ".exe"

        exe = EXE_DIR / algorithm

        for customers in (5, 10, 15, 100000):
            writer.write(f"#n_customers={customers}\n")
            for file in TESTCASES.glob("*.txt"):
                process = subprocess.Popen(
                    [
                        str(exe),
                        str(file),
                        str(customers),
                        "30000",  # 30 seconds
                    ],
                    stdout=subprocess.PIPE,
                )
                stdout, _ = process.communicate()
                execution: Execution = json.loads(stdout.decode("utf-8"))

                writer.write(f"{file.stem}\t{execution['status']}\t{execution['cost']:6}\t{execution['elapsed_ms']}\t{execution['routes']}\n")
                writer.flush()

            writer.write("\n")
