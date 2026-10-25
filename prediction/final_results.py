"""Read and plot actual corrected results; no hardcoded performance values."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from prediction.plots import plot_results

if __name__ == "__main__":
    plot_results()
    print((config.RUN_DIR / "evaluation.md").read_text())
