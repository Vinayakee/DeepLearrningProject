"""Rebuild corrected sequences and training-only scaling statistics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.pipeline import prepare_sequences

if __name__ == "__main__":
    prepare_sequences()
