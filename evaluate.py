"""
evaluate.py
-----------
Loads the trained model and evaluates it on the held-out test set.
Produces:
  - outputs/classification_report.txt (precision, recall, F1)
  - outputs/confusion_matrix.png

Run:
    python evaluate.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocess import get_data_generators, CLASSES

MODEL_PATH = os.path.join("model", "mask_detector.h5")
OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'. Run `python train.py` first."
        )

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Loading test data...")
    _, _, test_gen = get_data_generators()

    print("Running predictions on test set...")
    probs = model.predict(test_gen, verbose=1)
    y_pred = (probs.ravel() > 0.5).astype(int)
    y_true = test_gen.classes  # ground-truth labels (0=with_mask, 1=without_mask)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print("\nClassification Report:\n", report)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)
    print(f"Saved classification report to {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Face Mask Detection")
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
