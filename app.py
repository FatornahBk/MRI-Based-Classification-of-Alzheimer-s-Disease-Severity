import os
import io
from typing import List

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from huggingface_hub import hf_hub_download
from model_def import build_model

# =========================
# Page / Defaults
# =========================
st.set_page_config(page_title="Lab 6 • Deploy Model to Web App", layout="centered")

DEVICE = "cpu"
IMAGE_SIZE = 600

# ====== CONFIG: ตั้ง repo/ไฟล์น้ำหนักของคุณบน HF ======
HF_REPO_ID   = st.secrets.get("HF_REPO_ID",   "FatornahBk/MRI-Based-Classification-of-Alzheimer-s-Disease-Severity.git")
HF_FILENAME  = st.secrets.get("HF_FILENAME",  "inception_v3_fold0_state_dict.pt")
CLASSES_FILE = "classes.txt"

# =========================
# Utils
# =========================
@st.cache_resource(show_spinner=True)
def load_model_and_classes():
    # โหลด classes
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    num_classes = len(classes)

    # ดาวน์โหลด weights จาก HF (cache อัตโนมัติ)
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    model = build_model(num_classes=num_classes, device=DEVICE)

    # รองรับทั้ง plain state_dict และ checkpoint ที่ห่อด้วย dict
    # PyTorch 2.6 default: weights_only=True → ถ้าไฟล์เก่า อาจ error
    try:
        sd = torch.load(weights_path, map_location=DEVICE)  # default (2.6=weights_only=True)
    except Exception:
        sd = torch.load(weights_path, map_location=DEVICE, weights_only=False)

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # แก้ key prefix กรณี train ด้วย lightning/ddp
    fixed = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("model.", "module.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        fixed[nk] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.eval()
    return model, classes

@st.cache_resource
def get_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

def predict(img: Image.Image, model: torch.nn.Module, classes: List[str]):
    tfm = get_transform()
    x = tfm(img.convert("RGB")).unsqueeze(0)  # [1,3,H,W]
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        probs  = F.softmax(logits, dim=1).cpu().numpy().flatten()
    return list(zip(classes, probs))

# =========================
# UI
# =========================
st.title("MRI-Based-Classification-of-Alzheimer-s-Disease-Severity")

# โหลดโมเดล/คลาสครั้งเดียว
with st.spinner("Loading model & weights..."):
    model, classes = load_model_and_classes()

uploaded = st.file_uploader("อัปโหลดภาพ", type=["jpg", "jpeg", "png", "bmp", "webp"])
col1, col2 = st.columns([1, 1])

if uploaded:
    img = Image.open(uploaded)
    col1.image(img, caption="Input", use_container_width=True)

    if st.button("🔮 Predict"):
        results = predict(img, model, classes)

        # จัดเรียงผลตามลำดับที่คุณต้องการ (ตัวอย่าง: Mild, Moderate, No, Very Mild)
        # ถ้าต้องการลำดับตามไฟล์ classes.txt ให้คอมเมนต์บล็อกนี้ทิ้งได้
        preferred_order = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]
        order_map = {name: i for i, name in enumerate(preferred_order)}
        results_sorted = sorted(results, key=lambda x: order_map.get(x[0], 999))

        with col2:
            st.subheader("ผลการทำนาย")
            for cls, p in results_sorted:
                st.write(f"**{cls}** : {p*100:.2f}%")
            # สรุป class ที่น่าจะเป็นที่สุด
            top_cls, top_p = max(results, key=lambda x: x[1])
            st.success(f"Predict: **{top_cls}** ({top_p*100:.2f}%)")