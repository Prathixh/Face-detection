"""
train.py
--------
Builds a face mask classifier using transfer learning on MobileNetV2 and
trains it on the dataset. Saves the trained model and training curves.

Run:
    python train.py
"""

import os
import matplotlib
matplotlib.use("Agg")  # so it works without a display (servers/CI)
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocess import create_split_dataset, get_data_generators, IMG_SIZE

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "mask_detector.h5")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")

EPOCHS = 2
LEARNING_RATE = 1e-4


def build_model():
    """Builds MobileNetV2 base (frozen, ImageNet weights) + custom head."""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    )
    # Freeze the pretrained base — we only train the new head.
    # This is fast and works well even with a modest dataset.
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)  # binary: mask vs no-mask

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    print(f"Saved training curves to {HISTORY_PLOT_PATH}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Step 1: Splitting dataset into train/val/test...")
    create_split_dataset()

    print("\nStep 2: Building data generators...")
    train_gen, val_gen, _ = get_data_generators()
    print(f"Class indices: {train_gen.class_indices}")

    print("\nStep 3: Building model (MobileNetV2 transfer learning)...")
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    print("\nStep 4: Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_history(history)
    print(f"\nTraining complete. Best model saved to '{MODEL_PATH}'.")
    print("Next: run `python evaluate.py` to test performance on unseen data.")


if __name__ == "__main__":
    main()
