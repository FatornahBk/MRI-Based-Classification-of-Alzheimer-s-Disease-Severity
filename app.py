import os
from typing import List, Tuple
import tempfile, pathlib, requests

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from model_def import build_model

# ---------------- Page ----------------
st.set_page_config(page_title="Alzheimer's MRI — Inception v3", layout="centered")
st.title("🧠 MRI-Based Classification of Alzheimer's Disease Severity (Inception v3)")

# ---------------- Config ----------------
DEVICE = "cpu"
IMAGE_SIZE = 299
CLASSES_FILE = "classes.txt"

# Secret: direct URL to .pt file (จาก GitHub Releases)
MODEL_URL = st.secrets.get("MODEL_URL", None)

def _download_from_url(url: str) -> str:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    fd, tmp_path = tempfile.mkstemp(suffix=pathlib.Path(url).suffix or ".pt")
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tmp_path

@st.cache_resource(show_spinner=True)
def load_model_and_classes():
    # โหลด classes
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        classes = [ln.strip() for ln in f if ln.strip()]
    num_classes = len(classes)

    if not MODEL_URL:
        st.error("❌ กรุณาตั้งค่า Secrets: MODEL_URL = <ลิงก์ release asset>")
        st.stop()
    weights_path = _download_from_url(MODEL_URL)

    # สร้างโมเดลและโหลดน้ำหนัก
    model = build_model(num_classes=num_classes, device=DEVICE)
    try:
        sd = torch.load(weights_path, map_location=DEVICE)
    except Exception:
        sd = torch.load(weights_path, map_location=DEVICE, weights_only=False)

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    fixed = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("model.", "module.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        fixed[nk] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing: print("Missing keys:", missing)
    if unexpected: print("Unexpected keys:", unexpected)

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

def predict(img: Image.Image, model: torch.nn.Module, classes: List[str]) -> List[Tuple[str, float]]:
    x = get_transform()(img.convert("RGB")).unsqueeze(0)
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