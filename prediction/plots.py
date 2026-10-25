"""Plots consume actual saved history/metrics rather than pasted numbers."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config


def plot_results():
    history = pd.read_csv(config.RUN_DIR / "training_history.csv")
    for key in ("loss", "masked_accuracy"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history.epoch + 1, history[key], label="Train")
        ax.plot(history.epoch + 1, history["val_" + key], label="Validation")
        ax.set(xlabel="Epoch", ylabel=key.replace("_", " "))
        ax.legend(); fig.tight_layout()
        fig.savefig(config.RUN_DIR / f"training_{key}.png", dpi=150); plt.close(fig)
    results = json.loads((config.RUN_DIR / "evaluation.json").read_text())
    cm = np.asarray(results["test"]["model"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=config.CLASS_NAMES,
           yticklabels=config.CLASS_NAMES, xlabel="Predicted", ylabel="Actual", title="2025 held-out test set")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(config.RUN_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)
