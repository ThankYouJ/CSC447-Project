# train_models.py

import os
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from features import extract_data_for_model, ALL_CLASSES


# Define candidate models to compare
def get_candidate_models():
    """
    คืน dict ของโมเดลหลายแบบ เพื่อนำไปเทรนแล้วเลือกตัวที่ดีที่สุด
    """
    models = {
        "logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier()
    }
    return models


def train_best_model_for_disease(target_disease, save_dir="models"):
    """
    1. ดึงข้อมูล X_train, y_train, X_test, y_test สำหรับโรค target_disease
    2. เทรนหลายโมเดล + StandardScaler
    3. เลือกตัวที่ accuracy บน test สูงสุด
    4. เซฟโมเดลที่ดีที่สุดเป็น .joblib
    """
    print("=" * 60)
    print(f"Training models for disease: {target_disease}")
    print("=" * 60)

    data = extract_data_for_model(target_disease)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    models = get_candidate_models()

    best_name = None
    best_model = None
    best_acc = -1.0

    for name, clf in models.items():
        print(f"\nTraining model: {name}")

        # Use a pipeline so scaling + model stay together
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy on test for {name}: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred, digits=4))

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = pipe

    # Ensure model directory exists
    os.makedirs(save_dir, exist_ok=True)

    # e.g. models/ALGAL_LEAF_SPOT_random_forest_best.joblib
    filename = os.path.join(
        save_dir,
        f"{target_disease}_{best_name}_best.joblib"
    )

    joblib.dump(best_model, filename)
    print(f"\nBest model for {target_disease}: {best_name} (acc={best_acc:.4f})")
    print(f"Saved to: {filename}")

    return filename, best_acc


def train_all_diseases():
    """
    เทรนโมเดล 1 ตัวต่อ 1 disease ใน ALL_CLASSES
    แต่ละตัวคือ binary classifier: is_target_disease vs not
    """
    summary = []

    for disease in ALL_CLASSES:
        model_path, acc = train_best_model_for_disease(disease)
        summary.append((disease, acc, model_path))

    print("\n" + "#" * 60)
    print("Training Summary")
    print("#" * 60)
    for disease, acc, path in summary:
        print(f"{disease:25s}  acc={acc:.4f}  model={path}")


if __name__ == "__main__":
    # เทรนทุกโรคทีเดียว
    train_all_diseases()