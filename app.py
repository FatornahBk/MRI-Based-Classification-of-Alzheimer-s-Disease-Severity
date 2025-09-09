# app.py
import os
import io
from collections import OrderedDict

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from timm import create_model

# =========================
# Page / Defaults
# =========================
st.set_page_config(
    page_title="MRI-Based Classification of Alzheimer's Disease Severity",
    layout="centered",
)

# ---- Custom CSS for Dark Theme ----
st.markdown(
    """
    <style>
    body {
        background-color: #111827;  /* Dark background */
        color: #E5E7EB;  /* Light gray text */
    }
    .stApp {
        background-color: #111827;
        color: #E5E7EB;
    }
    .result-card {
        background: #1F2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .class-name {
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .progress-container {
        width: 100%;
        background: #374151;
        border-radius: 10px;
        height: 18px;
        overflow: hidden;
        margin-top: 6px;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Config
# =========================
DEVICE = "cpu"
IMAGE_SIZE = 299  # Inception-v3 standard input size
CLASSES_TXT = os.environ.get("CLASSES_TXT", "classes.txt")

# ลำดับการ "แสดงผล" ในหน้า UI
DESIRED_ORDER = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
]

# พาธไฟล์น้ำหนัก (ตั้งผ่าน ENV ได้) + Fallback
DEFAULT_WEIGHTS = "weights/inception_v3_fold0_state_dict.pt"
FALLBACK_WEIGHTS = "inception_v3_fold0_state_dict.pt"
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", DEFAULT_WEIGHTS)

# URL สำหรับดาวน์โหลดไฟล์น้ำหนักอัตโนมัติ (ถ้ามี)
MODEL_WEIGHTS_URL = os.environ.get("MODEL_WEIGHTS_URL", "").strip()

if not os.path.exists(MODEL_WEIGHTS) and os.path.exists(FALLBACK_WEIGHTS):
    MODEL_WEIGHTS = FALLBACK_WEIGHTS

# =========================
# Helpers
# =========================
def _norm_name(s: str) -> str:
    return " ".join(s.lower().strip().split())

@st.cache_resource(show_spinner=False)
def load_classes():
    if not os.path.exists(CLASSES_TXT):
        st.error(f"classes.txt not found at: {CLASSES_TXT}")
        st.stop()
    with open(CLASSES_TXT, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    if len(classes) == 0:
        st.error("classes.txt is empty.")
        st.stop()
    return classes

def _download(url: str, dst_path: str) -> bool:
    """ดาวน์โหลดไฟล์น้ำหนัก (ถ้าตั้ง URL)"""
    try:
        import requests
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with st.spinner("Downloading model weights..."):
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            with open(dst_path, "wb") as f:
                f.write(r.content)
        return True
    except Exception as e:
        st.error(f"Download weights failed: {e}")
        return False

@st.cache_resource(show_spinner=False)
def ensure_weights_local(path: str, url: str | None = None) -> str | None:
    if os.path.exists(path):
        return path
    if url:
        ok = _download(url, path)
        if ok and os.path.exists(path):
            return path
    return None

@st.cache_resource(show_spinner=False)
def load_model(num_classes: int):
    """
    โหลด Inception-v3 (timm) + ใส่ head ตามจำนวนคลาส
    รองรับการโหลด state_dict แบบ plain
    """
    # สร้างโมเดล
    model = create_model(
        "inception_v3",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    model.eval()
    model.to(DEVICE)

    # ให้แน่ใจว่ามีไฟล์น้ำหนัก (จะดาวน์โหลดถ้ามี URL)
    weights_path = ensure_weights_local(MODEL_WEIGHTS, MODEL_WEIGHTS_URL)

    if not weights_path or not os.path.exists(weights_path):
        st.warning(
            f"Model weights not found at: {MODEL_WEIGHTS}\n"
            "Running with randomly-initialized head. "
            "Upload weights, set env MODEL_WEIGHTS to a valid path, "
            "หรือกำหนด MODEL_WEIGHTS_URL เพื่อดาวน์โหลดอัตโนมัติ."
        )
        return model

    # โหลด state_dict แบบปลอดภัย
    try:
        sd = torch.load(weights_path, map_location=DEVICE)
        # รองรับกรณี save({'state_dict': ...}) หรือ key มี prefix "model."
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict):
            sd = {k.replace("model.", "").replace("module.", ""): v for k, v in sd.items()}

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            st.info(
                f"Loaded with non-strict mode. Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}"
            )
    except Exception as e:
        st.error(
            "Failed to load model weights. Make sure it's a plain state_dict compatible with timm Inception-v3 head.\n\n"
            f"{type(e).__name__}: {e}"
        )
    return model

def build_transform(size: int = IMAGE_SIZE):
    # Inception-v3 ใช้ ImageNet mean/std ได้ตามปกติ
    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def predict_image(model, img: Image.Image, classes):
    tfm = build_transform(IMAGE_SIZE)
    x = tfm(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        # บาง implementation อาจคืน logits อย่างเดียว (ไม่มี aux ใน timm)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    idx_to_class = {i: c for i, c in enumerate(classes)}
    class_to_prob = OrderedDict((idx_to_class[i], float(p)) for i, p in enumerate(probs))
    return class_to_prob

def render_progress_block(name: str, percent: float, is_top: bool):
    """
    แสดงชื่อคลาส + แถบเปอร์เซ็นต์ (progress bar) ใต้ชื่อ
    """
    percent_clamped = max(0.0, min(100.0, percent))
    color_bar = "#2563EB" if is_top else "#6B7280"  # blue for top, gray for others
    color_text = "#60A5FA" if is_top else "#E5E7EB"

    html = f"""
    <div class="class-name" style="color:{color_text};">
        {name} : {percent_clamped:.2f}%
    </div>
    <div class="progress-container">
        <div class="progress-bar" style="width:{percent_clamped}%; background:{color_bar};"></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# =========================
# UI
# =========================
st.title("🧠 MRI-Based Classification of Alzheimer's Disease Severity")
st.caption("อัปโหลดภาพ MRI แล้วกด **Predict** เพื่อดูผลการประเมินความรุนแรง (Inception-v3)")

classes = load_classes()
model = load_model(num_classes=len(classes))

norm_map = {_norm_name(name): name for name in classes}
desired_norm = [_norm_name(s) for s in DESIRED_ORDER]

uploaded = st.file_uploader("อัปโหลดภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, caption="Input image", use_container_width=True)
    except Exception as e:
        st.error(f"Unable to open the image: {e}")
        st.stop()

    if st.button("Predict", type="primary"):
        with st.spinner("Running inference..."):
            class_probs = predict_image(model, image, classes)

        # ====== HEADER CARD ======
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Prediction Result</div>', unsafe_allow_html=True)

        # เตรียมข้อมูลเรียงตาม DESIRED_ORDER
        rows = []
        for want_norm in desired_norm:
            if want_norm in norm_map:
                real_name = norm_map[want_norm]
                prob = class_probs.get(real_name, 0.0)
                rows.append((real_name, prob))
            else:
                rows.append((f"[Missing in classes.txt] {want_norm}", 0.0))

        max_idx = max(range(len(rows)), key=lambda i: rows[i][1]) if rows else -1

        # Render progress bars
        for i, (name, p) in enumerate(rows):
            render_progress_block(name, p * 100.0, is_top=(i == max_idx))

        st.markdown("</div>", unsafe_allow_html=True)  # close result-card

        # ====== Info Model/Device ======
        st.markdown(
            f"<div style='margin-top:1rem; font-size:0.9rem; color:#9CA3AF;'>"
            f"Model: Inception-v3 (timm) · Device: {DEVICE.upper()} · Input: {IMAGE_SIZE}×{IMAGE_SIZE}</div>",
            unsafe_allow_html=True,
        )

        mismatches = [dn for dn in desired_norm if dn not in norm_map]
        if mismatches:
            st.warning(
                "บางชื่อที่ต้องการแสดง ไม่พบใน classes.txt โปรดตรวจสะกด/ตัวพิมพ์:\n- "
                + "\n- ".join(mismatches)
            )
else:
    st.info("อัปโหลดภาพ แล้วกด Predict")