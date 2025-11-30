from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import numpy as np
import cv2
import joblib
from features import extract_enhanced_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# เสิร์ฟ static ที่ /static เท่านั้น
app.mount("/static", StaticFiles(directory="static"), name="static")

# หน้าเว็บ = / 
@app.get("/")
def root():
    return FileResponse("static/index.html")


# โหลดโมเดล
bundle = joblib.load("models/multiclass_best.joblib")
model = bundle["model"]
le = bundle["label_encoder"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    feats = extract_enhanced_features(img).reshape(1, -1)

    pred_idx = model.predict(feats)[0]
    class_name = le.inverse_transform([pred_idx])[0]

    # Probabilities
    prob_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats)[0]
        prob_dict = {
            le.inverse_transform([i])[0]: float(probs[i])
            for i in range(len(probs))
        }

    return {
        "predicted": class_name,
        "probabilities": prob_dict,
    }
