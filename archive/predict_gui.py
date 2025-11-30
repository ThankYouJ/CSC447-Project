# predict_gui.py
# Simple GUI to choose an image and classify leaf disease

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

# ใช้ฟีเจอร์ที่ใช้ตอน train
from features import extract_enhanced_features, ALL_CLASSES

MODEL_PATH = "models/multiclass_best.joblib"


class LeafDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Durian Leaf Disease Classifier")

        # Try to load model bundle
        try:
            bundle = joblib.load(MODEL_PATH)
            self.model = bundle["model"]
            self.le = bundle["label_encoder"]
            self.model_name = bundle.get("model_name", "unknown")
            self.val_acc = bundle.get("val_accuracy", None)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            root.destroy()
            return

        # UI elements
        self.image_label = tk.Label(
            root,
            text="No image selected",
            width=40,
            height=15,
            bg="#333",
            fg="white",
        )
        self.image_label.pack(padx=10, pady=10)

        self.result_label = tk.Label(
            root,
            text="Prediction: -",
            font=("Arial", 14, "bold")
        )
        self.result_label.pack(pady=5)

        self.detail_label = tk.Label(
            root,
            text="",
            font=("Arial", 10),
            justify="left"
        )
        self.detail_label.pack(pady=5)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        select_btn = tk.Button(
            btn_frame,
            text="Choose Image...",
            command=self.choose_image
        )
        select_btn.pack(side=tk.LEFT, padx=5)

        info_text = f"Using model: {self.model_name}"
        if self.val_acc is not None:
            info_text += f"  (val acc ~ {self.val_acc*100:.1f}%)"

        info_label = tk.Label(root, text=info_text, font=("Arial", 8))
        info_label.pack(pady=(0, 10))

        self.current_photo = None  # keep reference to avoid GC

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select leaf image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return

        self.predict_image(path)

    def predict_image(self, path):
        # Load and show image
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")
            return

        display_img = pil_img.copy()
        display_img.thumbnail((256, 256))
        self.current_photo = ImageTk.PhotoImage(display_img)
        self.image_label.configure(image=self.current_photo, text="")

        # Convert to BGR for OpenCV-based feature extractor
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # ✨ ใช้ฟังก์ชันที่ถูกต้องสำหรับทำฟีเจอร์ (ตรงกับ TRAIN)
        feats = extract_enhanced_features(img_bgr).reshape(1, -1)

        # Predict
        preds = self.model.predict(feats)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feats)[0]
        else:
            probs = None

        class_idx = int(preds[0])
        class_name = self.le.inverse_transform([class_idx])[0]

        if class_name == "HEALTHY_LEAF":
            readable = "Healthy leaf 🌿"
        else:
            readable = f"Disease: {class_name}"

        self.result_label.config(text=f"Prediction: {readable}")

        # Build probability text (if available)
        if probs is not None:
            lines = []
            for idx, p in enumerate(probs):
                name = self.le.inverse_transform([idx])[0]
                lines.append(f"{name}: {p*100:.1f}%")
            self.detail_label.config(text="\n".join(lines))
        else:
            self.detail_label.config(text="(Model has no probability output.)")


def main():
    root = tk.Tk()
    app = LeafDiseaseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
