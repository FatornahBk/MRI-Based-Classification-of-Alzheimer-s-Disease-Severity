# app.py
import os
import io
from collections import OrderedDict

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# =========================
# Page / Defaults
# =========================
st.set_page_config(
    page_title="MRI-Based Classification of Alzheimer's Disease Severity",
    layout="centered",
)

# ---- Custom CSS: Dark Theme & UI ----
st.markdown(
    """
    <style>
    body { background:#111827; color:#E5E7EB; }
    .stApp { background:#111827; color:#E5E7EB; }

    .result-card{
        background:#1F2937;
        border-radius:16px;
        padding:24px;
        margin-bottom:20px;
        box-shadow:0 6px 18px rgba(0,0,0,.45);
        border:1px solid #0B1220;
    }
    .result-title{
        font-size:2.2rem;
        font-weight:800;
        color:#fff;
        margin:0 0 .75rem 0;
        letter-spacing:.3px;
    }
    .predicted-title{
        font-size:2.0rem;
        font-weight:800;
        color:#E5E7EB;
        text-align:center;
        margin:.25rem 0 .5rem 0;
    }
    .pill-wrap{ text-align:center; margin-bottom:1rem; }
    .pill{
        display:inline-block;
        padding:8px 16px;
        border-radius:999px;
        font-weight:700;
        border:1px solid rgba(37,99,235,.35);
        background:rgba(37,99,235,.10);
        color:#93C5FD;
        backdrop-filter:blur(3px);
    }
    .class-name{
        font-size:1.4rem;
        font-weight:600;
        margin:.5rem 0;
    }
    .progress-container{
        width:100%;
        background:#374151;
        border-radius:10px;
        height:16px;
        overflow:hidden;
        margin-top:6px;
    }
    .progress-bar{ height:100%; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Config (ตั้งให้ใช้ Inception v3 + weight ที่ระบุ)
# =========================
MODEL_NAME = os.environ.get("MODEL_NAME", "inception_v3").strip()

IMAGE_SIZE_MAP = {
    "efficientnet_b0": 224, "efficientnet_b1": 240, "efficientnet_b2": 260,
    "efficientnet_b3": 300, "efficientnet_b4": 380, "efficientnet_b5": 456,
    "efficientnet_b6": 528, "efficientnet_b7": 600,
    "inception_v3": 299, "resnet50": 224,
}
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", IMAGE_SIZE_MAP.get(MODEL_NAME, 299)))

DEVICE = os.environ.get("DEVICE", "cpu").lower()
CLASSES_TXT = os.environ.get("CLASSES_TXT", "classes.txt")

# ใช้ไฟล์น้ำหนัก inception_v3_fold0_state_dict.pt ตามที่ขอ
DEFAULT_WEIGHTS  = "weights/inception_v3_fold0_state_dict.pt"
FALLBACK_WEIGHTS = "inception_v3_fold0_state_dict.pt"
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", DEFAULT_WEIGHTS)
if not os.path.exists(MODEL_WEIGHTS) and os.path.exists(FALLBACK_WEIGHTS):
    MODEL_WEIGHTS = FALLBACK_WEIGHTS

# ลำดับการ "แสดงผล"
DESIRED_ORDER = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
]

# ลดการใช้ thread ของ Torch บนเครื่องเล็ก ๆ
try:
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
except Exception:
    pass

# =========================
# Helpers
# =========================
def _norm_name(s: str) -> str:
    return " ".join(s.lower().strip().split())

@st.cache_resource(show_spinner=False)
def load_classes_safe():
    path = CLASSES_TXT
    if not os.path.exists(path):
        st.warning(f"⚠️ classes.txt not found at: {path}. The UI is loaded but prediction is disabled.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
    except Exception as e:
        st.warning(f"⚠️ Cannot read classes.txt: {e}")
        return None
    if not classes:
        st.warning("⚠️ classes.txt is empty. The UI is loaded but prediction is disabled.")
        return None
    return classes

@st.cache_resource(show_spinner=False)
def load_model(num_classes: int):
    """Lazy import timm + โหลด weights; ถ้าโหลดไม่ได้จะใช้หัวสุ่มแทน"""
    try:
        from timm import create_model
        model = create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
    except Exception as e:
        st.error(f"สร้างโมเดล '{MODEL_NAME}' ไม่สำเร็จ: {e}")
        raise

    model.eval().to(DEVICE)

    if not os.path.exists(MODEL_WEIGHTS):
        st.warning(
            f"Model weights not found at: {MODEL_WEIGHTS}\n"
            "Running with randomly initialized head."
        )
        return model

    try:
        sd = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        if isinstance(sd, dict) and "state_dict" in sd and all(
            not k.startswith("model.") for k in sd["state_dict"].keys()
        ):
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            st.info(f"Loaded with non-strict mode. Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    except Exception as e:
        st.warning(
            "⚠️ Failed to load model weights. Using randomly initialized head instead.\n\n"
            f"{type(e).__name__}: {e}"
        )
    return model

def build_transform(size: int = IMAGE_SIZE):
    # Inception v3 ใช้ 299x299; normalize แบบ ImageNet
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def predict_image(model, img: Image.Image, classes):
    tfm = build_transform(IMAGE_SIZE)
    x = tfm(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().tolist()
    idx_to_class = {i: c for i, c in enumerate(classes)}
    return OrderedDict((idx_to_class[i], float(p)) for i, p in enumerate(probs))

def render_progress_block(name: str, percent: float, is_top: bool):
    percent = max(0.0, min(100.0, percent))
    color_bar  = "#2563EB" if is_top else "#6B7280"
    color_text = "#60A5FA" if is_top else "#E5E7EB"
    st.markdown(
        f"""
        <div class="class-name" style="color:{color_text};">{name} : {percent:.2f}%</div>
        <div class="progress-container">
            <div class="progress-bar" style="width:{percent}%; background:{color_bar};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# UI
# =========================
st.title("🧠 MRI-Based Classification of Alzheimer's Disease Severity")
st.caption("อัปโหลดภาพ MRI แล้วกด **Predict** เพื่อดูผลการประเมินความรุนแรง")

classes = load_classes_safe()
norm_map = {_norm_name(name): name for name in classes} if classes else {}
desired_norm = [_norm_name(s) for s in DESIRED_ORDER]

uploaded = st.file_uploader("อัปโหลดภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, caption="Input image", width=320)
    except Exception as e:
        st.error(f"Unable to open the image: {e}")
        st.stop()

    if st.button("Predict", type="primary"):
        if not classes:
            st.error("Cannot predict: classes.txt is missing or empty.")
        else:
            with st.spinner(f"Preparing model ({MODEL_NAME})..."):
                model = load_model(num_classes=len(classes))

            with st.spinner("Running inference..."):
                class_probs = predict_image(model, image, classes)

            # ====== CARD: Prediction Result ======
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-title">Prediction Result</div>', unsafe_allow_html=True)

            if class_probs:
                top_name, top_prob = max(class_probs.items(), key=lambda kv: kv[1])
            else:
                top_name, top_prob = "-", 0.0

            st.markdown(
                f"""
                <div class="predicted-title">Predicted: {top_name}</div>
                <div class="pill-wrap"><span class="pill">Confidence: {top_prob*100:.2f}%</span></div>
                """,
                unsafe_allow_html=True,
            )

            rows = []
            for want_norm in desired_norm:
                if want_norm in norm_map:
                    real = norm_map[want_norm]
                    rows.append((real, class_probs.get(real, 0.0)))
                else:
                    rows.append((f"[Missing in classes.txt] {want_norm}", 0.0))

            max_idx = max(range(len(rows)), key=lambda i: rows[i][1]) if rows else -1
            for i, (name, p) in enumerate(rows):
                render_progress_block(name, p * 100.0, is_top=(i == max_idx))

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                f"<div style='margin-top:1rem; font-size:0.9rem; color:#9CA3AF;'>"
                f"Model: {MODEL_NAME} (timm) · ImageSize: {IMAGE_SIZE} · Device: {DEVICE.upper()}</div>",
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