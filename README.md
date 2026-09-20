# Face Mask Detection — Step 1: Core AI Model
### (Dataset, Preprocessing, Training, Evaluation)

This is **Step 1 of 4** in your face mask detection system. This step builds and
evaluates the core CNN classifier that decides: **Mask** vs **No Mask**.

Later steps (2–4) can plug into the saved model produced here — e.g.
face detection on video frames, a Flask/Streamlit app, or a Raspberry Pi/CCTV
deployment. This step is self-contained and doesn't depend on those.

---

## 1. Dataset

For a basic college project, use the widely-used **Face Mask Detection Dataset**
on Kaggle by Ashish Jangra. It's small, clean, and perfect for a first version:

**Link:** https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

- ~12,000 images total
- 2 classes: `WithMask`, `WithoutMask`
- Already split into Train / Validation / Test folders

Alternative (also very common, simpler, ~1,300 images, good if you want something
lighter): **Face Mask Detection by Omkar Gurav**
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

### How to download
1. Create a free Kaggle account (if you don't have one).
2. Go to the dataset link above → click **Download**.
3. Unzip it.
4. Arrange it into this folder structure (rename folders if needed):

```
dataset/
├── with_mask/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── without_mask/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

The training script automatically splits this into train/validation/test —
you don't need to manually pre-split it.

> Tip for your report: mention dataset name, source, size, and class balance —
> examiners often ask this in vivas.

---

## 2. Project Structure

```
mask_detection_project/
├── dataset/
│   ├── with_mask/          <-- put images here
│   └── without_mask/       <-- put images here
├── model/
│   ├── mask_detector.h5    <-- saved trained model (created after training)
│   └── training_history.png
├── outputs/
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── preprocess.py            <-- data loading & augmentation helpers
├── train.py                 <-- builds + trains the CNN, saves the model
├── evaluate.py               <-- evaluates saved model, generates report + plots
├── requirements.txt
└── README.md
```

---

## 3. Setup

```bash
pip install -r requirements.txt
```

## 4. Run — in order

```bash
# 1. Train the model (this also validates as it trains)
python train.py

# 2. Evaluate on the held-out test set
python evaluate.py
```

Outputs:
- `model/mask_detector.h5` — the trained Keras model (used by future steps
  for live video / image inference)
- `model/training_history.png` — accuracy/loss curves
- `outputs/confusion_matrix.png` — visual performance breakdown
- `outputs/classification_report.txt` — precision, recall, F1-score per class

---

## 5. Approach used (for your report/viva)

- **Model:** Transfer learning on **MobileNetV2** (pretrained on ImageNet),
  with a custom classification head added on top. This is standard practice
  for small/medium datasets — it gives strong accuracy without needing to
  train a CNN from scratch or a huge dataset.
- **Preprocessing:** resize to 224×224, pixel normalization, data augmentation
  (rotation, zoom, shift, horizontal flip) to reduce overfitting on a small
  dataset.
- **Training:** base MobileNetV2 layers frozen, only the new head is trained
  (fast, works well on CPU/laptop-class hardware). Adam optimizer, binary
  cross-entropy loss, early stopping on validation loss.
- **Evaluation:** accuracy, precision, recall, F1-score, confusion matrix on
  an untouched test split.

## 6. What's next (Steps 2–4, not part of this step)
- Step 2: Face detection (e.g. Haar Cascade / MTCNN) to locate faces in an
  image or webcam frame before feeding them to this model.
- Step 3: Real-time video inference pipeline.
- Step 4: Deployment (web app / alert system / dashboard).
