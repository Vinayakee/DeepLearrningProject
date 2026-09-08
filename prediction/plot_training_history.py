import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

HISTORY_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "training_history.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

history = pd.read_csv(
    HISTORY_FILE
)

print("Training history loaded.")

print(
    "Epochs:",
    len(history)
)


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["epoch"],
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ConvLSTM Training and Validation Loss"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()


loss_path = os.path.join(
    OUTPUT_DIR,
    "training_validation_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.close()


print(
    "✓ Loss graph saved:",
    loss_path
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["epoch"],
    history["masked_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["epoch"],
    history["val_masked_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "ConvLSTM Training and Validation Accuracy"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()


accuracy_path = os.path.join(
    OUTPUT_DIR,
    "training_validation_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


print(
    "✓ Accuracy graph saved:",
    accuracy_path
)


# ============================================================
# FINAL VALUES
# ============================================================

print("\nFinal epoch results:")

print(
    "Training loss:",
    history["loss"].iloc[-1]
)

print(
    "Validation loss:",
    history["val_loss"].iloc[-1]
)

print(
    "Training accuracy:",
    history["masked_accuracy"].iloc[-1]
)

print(
    "Validation accuracy:",
    history["val_masked_accuracy"].iloc[-1]
)


print("\n✓ TRAINING GRAPHS COMPLETE")