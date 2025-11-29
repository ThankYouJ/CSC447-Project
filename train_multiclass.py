# train_multiclass.py
# Train multiple multiclass models, compare on val set, save the best as .joblib

import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from features import build_multiclass_dataset, ALL_CLASSES


def get_candidate_models():
    """
    Return a dict of candidate classifiers for multiclass problems.
    All will be wrapped in a Pipeline with StandardScaler.
    """
    models = {
        # Logistic Regression
        "logreg_c1": LogisticRegression(max_iter=2000, C=1, n_jobs=-1),
        "logreg_c01": LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1),
        "logreg_c10": LogisticRegression(max_iter=2000, C=10, n_jobs=-1),

        # SVM (RBF)
        "svm_rbf_c1": SVC(kernel="rbf", C=1, gamma="scale", probability=True),
        "svm_rbf_c10": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "svm_rbf_c01": SVC(kernel="rbf", C=0.1, gamma="scale", probability=True),

        # Random Forest
        "rf_200": RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, class_weight="balanced"),
        "rf_300": RandomForestClassifier(n_estimators=300, max_depth=15, n_jobs=-1, class_weight="balanced"),
        "rf_400": RandomForestClassifier(n_estimators=400, max_depth=10, n_jobs=-1, class_weight="balanced"),

        # Gradient Boosting
        "gb_lr01": GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=3),
        "gb_lr005": GradientBoostingClassifier(learning_rate=0.05, n_estimators=300, max_depth=3),
        "gb_lr02": GradientBoostingClassifier(learning_rate=0.2, n_estimators=150, max_depth=2),
    }
    return models


def train_and_select_best(save_path="models/multiclass_best.joblib"):
    # Build train/val datasets (combined features)
    print("Building train/val multiclass datasets...")
    X_train, y_train, le = build_multiclass_dataset("train")
    X_val, y_val, _ = build_multiclass_dataset("val")

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}\n")

    models = get_candidate_models()

    best_name = None
    best_model = None
    best_acc = -1.0
    all_results = {}

    for name, clf in models.items():
        print("=" * 60)
        print(f"Training model: {name}")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        all_results[name] = acc

        print(f"Validation accuracy for {name}: {acc:.4f}")
        print("Classification report (val):")
        print(classification_report(y_val, y_pred, target_names=ALL_CLASSES))

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = pipe

    # Save the best model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bundle = {
        "model": best_model,
        "label_encoder": le,
        "model_name": best_name,
        "val_accuracy": best_acc,
        "all_results": all_results,
    }

    joblib.dump(bundle, save_path)

    print("=" * 60)
    print(f"Best model: {best_name}  (val accuracy = {best_acc:.4f})")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    train_and_select_best()