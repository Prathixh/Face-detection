"""
preprocess.py
--------------
Handles dataset loading, train/validation/test splitting, and image
augmentation for the Face Mask Detection project.

Expected folder structure BEFORE running:

    dataset/
    ├── with_mask/
    │   ├── img1.jpg
    │   └── ...
    └── without_mask/
        ├── img1.jpg
        └── ...

This script will create a clean split (train/val/test) automatically inside
`dataset_split/` so the originals in `dataset/` are never modified.
"""

import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------- Config ----------------------
IMG_SIZE = (224, 224)      # required input size for MobileNetV2
BATCH_SIZE = 32
RAW_DATASET_DIR = "dataset"
RANDOM_SEED = 42

# Dynamic path and naming detection for existing "Face Mask Dataset" vs custom split
if os.path.exists("Face Mask Dataset"):
    SPLIT_DATASET_DIR = "Face Mask Dataset"
    CLASSES = ["WithMask", "WithoutMask"]
    TRAIN_DIR = "Train"
    VAL_DIR = "Validation"
    TEST_DIR = "Test"
    HAS_PRE_SPLIT = True
else:
    SPLIT_DATASET_DIR = "dataset_split"
    CLASSES = ["with_mask", "without_mask"]
    TRAIN_DIR = "train"
    VAL_DIR = "val"
    TEST_DIR = "test"
    HAS_PRE_SPLIT = False

SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test
# ------------------------------------------------------


def create_split_dataset():
    """
    Splits dataset/with_mask and dataset/without_mask into
    dataset_split/train, dataset_split/val, dataset_split/test
    (each with with_mask/ and without_mask/ subfolders).

    Safe to re-run: it wipes and rebuilds dataset_split/ each time so the
    split always matches the current contents of dataset/.
    """
    if HAS_PRE_SPLIT:
        print("Using existing 'Face Mask Dataset' directly. Skipping split step.")
        return

    random.seed(RANDOM_SEED)

    if os.path.exists(SPLIT_DATASET_DIR):
        shutil.rmtree(SPLIT_DATASET_DIR)

    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            os.makedirs(os.path.join(SPLIT_DATASET_DIR, split, cls), exist_ok=True)

    for cls in CLASSES:
        src_dir = os.path.join(RAW_DATASET_DIR, cls)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(
                f"Expected folder '{src_dir}' not found. "
                f"Download the dataset and place images there (see README.md)."
            )

        images = [f for f in os.listdir(src_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for fname in train_files:
            shutil.copy(os.path.join(src_dir, fname),
                        os.path.join(SPLIT_DATASET_DIR, TRAIN_DIR, cls, fname))
        for fname in val_files:
            shutil.copy(os.path.join(src_dir, fname),
                        os.path.join(SPLIT_DATASET_DIR, VAL_DIR, cls, fname))
        for fname in test_files:
            shutil.copy(os.path.join(src_dir, fname),
                        os.path.join(SPLIT_DATASET_DIR, TEST_DIR, cls, fname))

        print(f"[{cls}] total={n} -> train={len(train_files)}, "
              f"val={len(val_files)}, test={len(test_files)}")

    print(f"\nDataset split created at '{SPLIT_DATASET_DIR}/'")


def get_data_generators():
    """
    Returns (train_generator, val_generator, test_generator) using
    Keras ImageDataGenerator with augmentation applied only to training data.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # No augmentation for val/test — only rescaling, to get honest evaluation
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(SPLIT_DATASET_DIR, TRAIN_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(SPLIT_DATASET_DIR, VAL_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(SPLIT_DATASET_DIR, TEST_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


if __name__ == "__main__":
    # Running this file directly just performs the split, for inspection.
    create_split_dataset()
