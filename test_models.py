import os
import cv2
import joblib
import numpy as np

from features import (
    DATASET_ROOT,
    COLOR_GROUP,
    ALL_CLASSES,
    get_color_features,
    get_texture_features,
)

MODELS_DIR = "models"
PHASE = "test"  # train, test,or val is cool ja


def load_model_for_disease(target_disease, models_dir=MODELS_DIR):
    """
    Load the saved .joblib model for a given disease.
    It looks for a file that starts with the disease name, e.g.
    'ALGAL_LEAF_SPOT_....joblib'
    """
    for fname in os.listdir(models_dir):
        if fname.startswith(target_disease) and fname.endswith(".joblib"):
            model_path = os.path.join(models_dir, fname)
            print(f"   Loaded model for {target_disease} from: {model_path}")
            return joblib.load(model_path)

    raise FileNotFoundError(
        f"No model file found for disease '{target_disease}' in folder '{models_dir}'"
    )


def load_all_models():
    """
    Load every disease model once and keep in a dict:
    { disease_name: model }
    """
    models = {}
    print("Loading models...")
    for disease in ALL_CLASSES:
        models[disease] = load_model_for_disease(disease)
    print("All models loaded.\n")
    return models


def extract_features_for_disease(img_bgr, target_disease):
    """
    Use the same rule as in training:
    - diseases in COLOR_GROUP  -> color features
    - others                    -> texture/edge features
    """
    if target_disease in COLOR_GROUP:
        feats = get_color_features(img_bgr)
    else:
        feats = get_texture_features(img_bgr)

    feats = np.array(feats, dtype=np.float32).reshape(1, -1)
    return feats


def evaluate_dataset(phase=PHASE):
    """
    Evaluate the whole system on Durian_Leaf_Disease_Dataset/<phase>/

    For each image:
      - true label = folder name
      - predicted label = argmax over disease models (probability of class 1)
    """
    models = load_all_models()

    # overall counters
    total = 0
    correct = 0

    # per-class counters
    per_class_total = {cls: 0 for cls in ALL_CLASSES}
    per_class_correct = {cls: 0 for cls in ALL_CLASSES}

    base_dir = os.path.join(DATASET_ROOT, phase)

    print(f"   Evaluating on phase: {phase}")
    print(f"   Dataset path: {base_dir}\n")

    for true_class in ALL_CLASSES:
        class_dir = os.path.join(base_dir, true_class)
        if not os.path.isdir(class_dir):
            print(f"⚠️  Skipping {true_class}: folder not found at {class_dir}")
            continue

        print(f"📂 Class: {true_class}")
        for fname in os.listdir(class_dir):
            # (Optional) filter some non-image files if any
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                print(f"   ⚠️ Cannot read image: {fpath}")
                continue

            total += 1
            per_class_total[true_class] += 1

            # collect probabilities from each model
            probs = []  # list of (disease, prob)

            for disease in ALL_CLASSES:
                model = models[disease]
                X = extract_features_for_disease(img, disease)

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0][1]  # prob of class 1 for this disease
                else:
                    # fall back if no predict_proba (shouldn't happen with current models)
                    y_pred = model.decision_function(X)
                    # just turn decision score into something monotonic; won't be calibrated
                    proba = float(y_pred[0])

                probs.append((disease, float(proba)))

            # argmax over all diseases
            pred_class, pred_prob = max(probs, key=lambda x: x[1])

            is_correct = (pred_class == true_class)
            if is_correct:
                correct += 1
                per_class_correct[true_class] += 1

        print(f"   ➜ Done {true_class}, images: {per_class_total[true_class]}\n")

    # overall accuracy
    if total == 0:
        print("No images found for evaluation.")
        return

    overall_acc = correct / total * 100.0

    print("\n" + "#" * 60)
    print(f"FINAL EVALUATION RESULT on '{phase}' set")
    print("#" * 60)
    print(f"Overall accuracy: {overall_acc:.2f}%  ({correct}/{total} correct)\n")

    print("Per-class accuracy:")
    for cls in ALL_CLASSES:
        t = per_class_total[cls]
        c = per_class_correct[cls]
        if t == 0:
            acc = 0.0
        else:
            acc = c / t * 100.0
        print(f" - {cls:25s}: {acc:6.2f}%  ({c}/{t})")


if __name__ == "__main__":
    evaluate_dataset()