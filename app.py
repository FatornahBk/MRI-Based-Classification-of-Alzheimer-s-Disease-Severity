import os
from typing import List, Tuple

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from huggingface_hub import hf_hub_download

from model_def import build_model

# ---------------- Page ----------------
st.set_page_config(page_title="Alzheimer's MRI — Inception v3", layout="centered")
st.title("🧠 MRI-Based Classification of Alzheimer's Disease Severity (Inception v3)")

# ---------------- Config ----------------
DEVICE = "cpu"
IMAGE_SIZE = 299                     # Inception v3 ใช้ 299x299
CLASSES_FILE = "classes.txt"

# ตั้งค่าจาก Secrets (แก้ใน Streamlit Cloud > Settings > Secrets)
HF_REPO_ID  = st.secrets.get("HF_REPO_ID",  "FatornahBk/MRI-Based-Classification-of-Alzheimer-s-Disease-Severity.git")
HF_FILENAME = st.secrets.get("HF_FILENAME", "inception_v3_checkpoint_fold0.pt")

@st.cache_resource(show_spinner=True)
def load_model_and_classes():
    # โหลดคลาส
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    num_classes = len(classes)

    # ดาวน์โหลด checkpoint จาก HF (cache อัตโนมัติที่เซิร์ฟเวอร์)
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)

    model = build_model(num_classes=num_classes, device=DEVICE)

    # รองรับทั้ง plain state_dict และ checkpoint dict + ป้องกันปัญหา torch 2.6 (weights_only=True)
    try:
        sd = torch.load(weights_path, map_location=DEVICE)  # อาจเป็น weights_only=True ตามดีฟอลต์
    except Exception:
        sd = torch.load(weights_path, map_location=DEVICE, weights_only=False)

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # ลอก prefix ที่มาจาก lightning/ddp ออก
    fixed = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("model.", "module.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        fixed[nk] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing:  # debug log ในคอนโซล
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.eval()
    return model, classes

@st.cache_resource
def get_transform():
    # Inception v3 ปกติใช้ Normalization ตาม ImageNet
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

def predict(img: Image.Image, model: torch.nn.Module, classes: List[str]) -> List[Tuple[str, float]]:
    x = get_transform()(img.convert("RGB")).unsqueeze(0)  # [1,3,299,299]
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    return list(zip(classes, probs))

# ---------------- UI ----------------
uploaded = st.file_uploader("อัปโหลดภาพ (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"])
col1, col2 = st.columns(2)

if uploaded:
    img = Image.open(uploaded)
    col1.image(img, caption="Input", use_container_width=True)

    if st.button("🔮 Predict"):
        with st.spinner("กำลังทำนาย..."):
            model, classes = load_model_and_classes()
            results = predict(img, model, classes)

        # เรียงตามที่คุณอยากแสดง (ถ้าอยากเรียงตาม classes.txt ให้คอมเมนต์บล็อกนี้)
        preferred_order = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]
        rank = {n: i for i, n in enumerate(preferred_order)}
        results_sorted = sorted(results, key=lambda x: rank.get(x[0], 999))

        with col2:
            st.subheader("ผลการทำนาย")
            for cls, p in results_sorted:
                st.write(f"**{cls}** : {p*100:.2f}%")
            top_cls, top_p = max(results, key=lambda x: x[1])
            st.success(f"Predict: **{top_cls}** ({top_p*100:.2f}%)")
else:
    st.info("อัปโหลดภาพ MRI เพื่อเริ่มทำนาย")