# test_multiclass.py

import os
import cv2
import joblib
import numpy as np

from sklearn.metrics import accuracy_score, classification_report

from features import DATASET_ROOT, ALL_CLASSES, extract_enhanced_features

MODEL_PATH = "models/multiclass_best.joblib"


def build_test_data():
    X = []
    y_true = []
    labels = []

    base_dir = os.path.join(DATASET_ROOT, "val")

    for class_name in ALL_CLASSES:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            feats = extract_enhanced_features(img)
            X.append(feats)
            labels.append(class_name)

    X = np.array(X, dtype=np.float32)
    return X, labels


def main():
    print("📦 Loading multiclass model...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    le = bundle["label_encoder"]

    X_test, labels = build_test_data()
    y_true = le.transform(labels)

    print(f"X_test: {X_test.shape}, y_true: {y_true.shape}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    print("\n############################################################")
    print("🏁 MULTICLASS EVALUATION on 'test' set")
    print("############################################################")
    print(f"Overall accuracy: {acc*100:.2f}%  ({(y_true==y_pred).sum()}/{len(y_true)} correct)\n")

    print("📊 Per-class report:")
    print(classification_report(y_true, y_pred, target_names=ALL_CLASSES))
    
    # Confusion matrix
    # from sklearn.metrics import confusion_matrix
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # cm = confusion_matrix(y_true, y_pred)

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
    #             xticklabels=ALL_CLASSES,
    #             yticklabels=ALL_CLASSES)
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.show()


if __name__ == "__main__":
    main()