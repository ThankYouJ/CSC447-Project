# features.py

import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops

# กำหนดกลุ่มโรค
COLOR_GROUP = ['ALGAL_LEAF_SPOT', 'LEAF_BLIGHT']  # กลุ่มเน้นสี
TEXTURE_GROUP = ['ALLOCARIDARA_ATTACK', 'PHOMOPSIS_LEAF_SPOT', 'HEALTHY_LEAF']  # กลุ่มเน้นลาย/ขอบ

ALL_CLASSES = ['ALGAL_LEAF_SPOT',
               'ALLOCARIDARA_ATTACK',
               'HEALTHY_LEAF',
               'LEAF_BLIGHT',
               'PHOMOPSIS_LEAF_SPOT']

DATASET_ROOT = "Durian_Leaf_Disease_Dataset"


def get_color_features(img):
    """
    Extract simple color statistics in HSV space:
    mean & std for H, S, V.
    Returns a list of 6 float features.
    """
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    (h_m, h_s) = cv2.meanStdDev(hsv[:, :, 0])
    (s_m, s_s) = cv2.meanStdDev(hsv[:, :, 1])
    (v_m, v_s) = cv2.meanStdDev(hsv[:, :, 2])

    return [
        float(h_m[0][0]), float(h_s[0][0]),
        float(s_m[0][0]), float(s_s[0][0]),
        float(v_m[0][0]), float(v_s[0][0]),
    ]


def get_texture_features(img):
    """
    Extract texture (GLCM) + edge density features.
    Returns a list of 5 float features.
    """
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # GLCM at distances [1] and angles [0, 45, 90 degrees]
    glcm = graycomatrix(
        gray,
        [1],
        [0, np.pi / 4, np.pi / 2],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges) / (edges.shape[0] * edges.shape[1]))

    return [contrast, homogeneity, energy, correlation, edge_density]


def extract_data_for_model(target_disease):
    """
    Create a binary classification dataset for a specific disease:
      y = 1 if image is target_disease, else 0

    Returns:
        data = {
            'X_train': np.ndarray (N_train, n_features),
            'y_train': np.ndarray (N_train,),
            'X_test': np.ndarray (N_test, n_features),
            'y_test': np.ndarray (N_test,),
        }
    """
    data = {}

    # เลือกวิธีสกัดตามกลุ่มโรค
    if target_disease in COLOR_GROUP:
        extraction_func = get_color_features
    else:
        extraction_func = get_texture_features

    for phase in ['train', 'test']:
        features = []
        labels = []

        for class_name in ALL_CLASSES:
            path = os.path.join(DATASET_ROOT, phase, class_name)
            is_target = 1 if class_name == target_disease else 0

            if not os.path.exists(path):
                continue

            for fname in os.listdir(path):
                fpath = os.path.join(path, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue

                feats = extraction_func(img)
                features.append(feats)
                labels.append(is_target)

        data[f"X_{phase}"] = np.array(features, dtype=np.float32)
        data[f"y_{phase}"] = np.array(labels, dtype=np.int64)

    return data


# Add By Peak

from sklearn.preprocessing import LabelEncoder

def get_combined_features(img):
    """
    Combine color + texture for ALL images.
    Returns 11 features: 6 (color) + 5 (texture)
    """
    color_feats = get_color_features(img)      # 6
    texture_feats = get_texture_features(img)  # 5
    return np.array(color_feats + texture_feats, dtype=np.float32)

def build_multiclass_dataset(phase='train'):
    """
    Build a multiclass dataset:
        X_phase, y_phase, label_encoder

    X_phase: (N, 11)  combined features
    y_phase: (N,)     encoded labels (0..4)
    """
    X = []
    y = []
    labels = []

    base_dir = os.path.join(DATASET_ROOT, phase)

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

            feats = get_combined_features(img)
            X.append(feats)
            labels.append(class_name)

    X = np.array(X, dtype=np.float32)

    # Fit encoder on ALL_CLASSES to keep mapping stable
    le = LabelEncoder()
    le.fit(ALL_CLASSES)
    y = le.transform(labels)

    return X, y, le
