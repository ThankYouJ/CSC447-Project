import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops

# กำหนดกลุ่มโรค
COLOR_GROUP = ['ALGAL_LEAF_SPOT', 'LEAF_BLIGHT'] # กลุ่มเน้นสี
TEXTURE_GROUP = ['ALLOCARIDARA_ATTACK', 'PHOMOPSIS_LEAF_SPOT', 'HEALTHY_LEAF'] # กลุ่มเน้นลาย/ขอบ

ALL_CLASSES = ['ALGAL_LEAF_SPOT', 'ALLOCARIDARA_ATTACK', 'HEALTHY_LEAF', 'LEAF_BLIGHT', 'PHOMOPSIS_LEAF_SPOT']
DATASET_ROOT = "Durian_Leaf_Disease_Dataset"

# 1. ฟังก์ชันสกัดคุณลักษณะ: สี (Color)
def get_color_features(img):
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (h_m, h_s) = cv2.meanStdDev(hsv[:,:,0])
    (s_m, s_s) = cv2.meanStdDev(hsv[:,:,1])
    (v_m, v_s) = cv2.meanStdDev(hsv[:,:,2])
    return [h_m[0][0], h_s[0][0], s_m[0][0], s_s[0][0], v_m[0][0], v_s[0][0]]

# 2. ฟังก์ชันสกัดคุณลักษณะ: ลวดลายและขอบ (Texture & Edge)
def get_texture_features(img):
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    return [contrast, homogeneity, energy, correlation, edge_density]

# 3. ฟังก์ชันหลัก: สร้าง Dataset แยกตามโมเดลโรค
def extract_data_for_model(target_disease):
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
            
            if os.path.exists(path):
                for fname in os.listdir(path):
                    img = cv2.imread(os.path.join(path, fname))
                    if img is not None:
                        feats = extraction_func(img)
                        features.append(feats)
                        labels.append(is_target)
        
        data[f"X_{phase}"] = np.array(features).astype(np.float32)
        data[f"y_{phase}"] = np.array(labels)
        
    return data

# --- ตัวอย่างการเรียกใช้ ---
# 1. สำหรับโรค ALGAL (ใช้ฟีเจอร์สี)
algal_data = extract_data_for_model('ALGAL_LEAF_SPOT')
X_train_algal = algal_data['X_train']
y_train_algal = algal_data['y_train']

# 2. สำหรับโรค ALLOCARIDARA (ใช้ฟีเจอร์ Texture/Edge)
allo_data = extract_data_for_model('ALLOCARIDARA_ATTACK')
X_train_allo = allo_data['X_train']
y_train_allo = allo_data['y_train']