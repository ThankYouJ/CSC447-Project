# features.py

import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis

# กำหนดกลุ่มโรค
ALL_CLASSES = ['ALGAL_LEAF_SPOT',
               'ALLOCARIDARA_ATTACK',
               'HEALTHY_LEAF',
               'LEAF_BLIGHT',
               'PHOMOPSIS_LEAF_SPOT']

DATASET_ROOT = "Durian_Leaf_Disease_Dataset"

from sklearn.preprocessing import LabelEncoder

def extract_enhanced_features(img):
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats = []

    # 1. HSV Histogram (32 bins)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    feats += list(h_hist) + list(s_hist) + list(v_hist)
    feats = [float(f) for f in feats]

    # 2. LBP Texture (uniform)
    lbp = local_binary_pattern(gray, 16, 2, method="uniform")
    (lbp_hist, _) = np.histogram(lbp, bins=np.arange(0, 19), range=(0, 18))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    feats += list(lbp_hist)

    # 3. Multi-distance GLCM
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, levels=256,
                        symmetric=True, normed=True)

    properties = ["contrast", "homogeneity", "energy", "correlation"]
    for prop in properties:
        prop_vals = graycoprops(glcm, prop).flatten()
        feats += list(prop_vals)

    # 4. HSV Color Moments
    for i in range(3):
        ch = hsv[:, :, i].flatten()
        feats.append(float(np.mean(ch)))
        feats.append(float(np.std(ch)))
        feats.append(float(skew(ch)))
        feats.append(float(kurtosis(ch)))

    # 5. Edge Orientation Histogram
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    angles = np.arctan2(gy, gx)
    angles = (angles + np.pi) * (180.0 / np.pi)  # 0–360 degrees
    (ang_hist, _) = np.histogram(angles, bins=8, range=(0, 360))
    feats += list(ang_hist)

    return np.array(feats, dtype=np.float32)

def build_multiclass_dataset(phase="train"):
    """
    Build X, y, label_encoder using enhanced features for a given phase
    (train / val / test).
    """
    X = []
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

            feats = extract_enhanced_features(img)
            X.append(feats)
            labels.append(class_name)

    X = np.array(X, dtype=np.float32)

    le = LabelEncoder()
    le.fit(ALL_CLASSES)          # fixed class order
    y = le.transform(labels)

    return X, y, le
