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
        font-size:2.4rem;
        font-weight:800;
        color:#E5E7EB;
        text-align:center;
        margin:.5rem 0 .75rem 0;
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
        font-size:1.6rem;
        font-weight:600;
        margin:.5rem 0;
    }
    .progress-container{
        width:100%;
        background:#374151;
        border-radius:10px;
        height:18px;
        overflow:hidden;
        margin-top:6px;
    }
    .progress-bar{ height:100%; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEVICE      = "cpu"
IMAGE_SIZE  = 600
CLASSES_TXT = os.environ.get("CLASSES_TXT", "classes.txt")

# ลำดับการ "แสดงผล"
DESIRED_ORDER = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
]

# พาธไฟล์น้ำหนัก (ไม่มีก็รันต่อได้)
DEFAULT_WEIGHTS  = "weights/efficientnet_b7_fold1_state_dict.pt"
FALLBACK_WEIGHTS = "efficientnet_b7_fold1_state_dict.pt"
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", DEFAULT_WEIGHTS)
if not os.path.exists(MODEL_WEIGHTS) and os.path.exists(FALLBACK_WEIGHTS):
    MODEL_WEIGHTS = FALLBACK_WEIGHTS

# =========================
# Helpers
# =========================
def _norm_name(s: str) -> str:
    return " ".join(s.lower().strip().split())

@st.cache_resource(show_spinner=False)
def load_classes_safe():
    """
    เวอร์ชันปลอดภัย: ถ้าไม่มี/ว่าง -> ไม่ stop ให้ UI ขึ้นได้ก่อน
    """
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
    """
    Lazy import timm + โหลด weights แบบไม่ล้ม ถ้าไฟล์หายก็ใช้โมเดลเปล่า
    """
    from timm import create_model  # lazy import เพื่อลดดีเลย์ตอนบูต

    model = create_model(
        "efficientnet_b7",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    model.eval().to(DEVICE)

    # ถ้าไม่มีไฟล์น้ำหนัก ให้ใช้หัวแบบสุ่มไปก่อน (ไม่หยุดแอป)
    if not os.path.exists(MODEL_WEIGHTS):
        st.warning(
            f"Model weights not found at: {MODEL_WEIGHTS}\n"
            "Running with randomly initialized head."
        )
        return model

    try:
        sd = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        # รองรับ checkpoint ที่เก็บในคีย์ 'state_dict'
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
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    idx_to_class = {i: c for i, c in enumerate(classes)}
    return OrderedDict((idx_to_class[i], float(p)) for i, p in enumerate(probs))

def render_progress_block(name: str, percent: float, is_top: bool):
    """แสดงชื่อคลาส + แถบเปอร์เซ็นต์ (progress bar) ใต้ชื่อ"""
    percent = max(0.0, min(100.0, percent))
    color_bar  = "#2563EB" if is_top else "#6B7280"  # blue / gray
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

classes = load_classes_safe()  # ไม่ stop ถ้าไม่มีไฟล์
norm_map = {_norm_name(name): name for name in classes} if classes else {}
desired_norm = [_norm_name(s) for s in DESIRED_ORDER]

uploaded = st.file_uploader("อัปโหลดภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        image = Image.open(io.BytesIO(uploaded.read()))
        # รูปเล็กลง (เช่น 360px)
        st.image(image, caption="Input image", width=360)
    except Exception as e:
        st.error(f"Unable to open the image: {e}")
        st.stop()

    if st.button("Predict", type="primary"):
        if not classes:
            st.error("Cannot predict: classes.txt is missing or empty.")
        else:
            with st.spinner("Preparing model..."):
                model = load_model(num_classes=len(classes))  # โหลดตอนกดปุ่ม เพื่อลดดีเลย์บูต

            with st.spinner("Running inference..."):
                class_probs = predict_image(model, image, classes)

            # ====== CARD: Prediction Result ======
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-title">Prediction Result</div>', unsafe_allow_html=True)

            # Top-1 เพื่อโชว์บนการ์ด Predicted
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

            # เตรียมข้อมูลแสดงตาม DESIRED_ORDER
            rows = []
            for want_norm in desired_norm:
                if want_norm in norm_map:
                    real = norm_map[want_norm]
                    rows.append((real, class_probs.get(real, 0.0)))
                else:
                    rows.append((f"[Missing in classes.txt] {want_norm}", 0.0))

            max_idx = max(range(len(rows)), key=lambda i: rows[i][1]) if rows else -1

            # Progress bars ต่อคลาส
            for i, (name, p) in enumerate(rows):
                render_progress_block(name, p * 100.0, is_top=(i == max_idx))

            # ปิดการ์ด
            st.markdown("</div>", unsafe_allow_html=True)

            # Info Model/Device
            st.markdown(
                f"<div style='margin-top:1rem; font-size:0.9rem; color:#9CA3AF;'>"
                f"Model: EfficientNet-B7 (timm) · Device: {DEVICE.upper()}</div>",
                unsafe_allow_html=True,
            )

            # แจ้งเตือนชื่อไม่ตรงใน classes.txt
            mismatches = [dn for dn in desired_norm if dn not in norm_map]
            if mismatches:
                st.warning(
                    "บางชื่อที่ต้องการแสดง ไม่พบใน classes.txt โปรดตรวจสะกด/ตัวพิมพ์:\n- "
                    + "\n- ".join(mismatches)
                )
else:
    st.info("อัปโหลดภาพ แล้วกด Predict")