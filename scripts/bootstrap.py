from __future__ import annotations

import subprocess
import sys


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.check_call(command)


if __name__ == "__main__":
    run([sys.executable, "scripts/download_dataset.py"])
    run([sys.executable, "scripts/train_model.py"])
