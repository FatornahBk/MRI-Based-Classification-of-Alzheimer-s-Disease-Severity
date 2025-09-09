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
st.set_page_config(page_title="MRI-based Classification (EfficientNet-B7)", layout="centered")

DEVICE      = "cpu"
IMAGE_SIZE  = 600
CLASSES_TXT = os.environ.get("CLASSES_TXT", "classes.txt")

# ลำดับการ "แสดงผล" ตามที่ต้องการ (ไม่กระทบค่าจริงจากโมเดล)
DESIRED_ORDER = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
]

# พาธไฟล์น้ำหนัก (ตั้ง ENV MODEL_WEIGHTS ได้)
DEFAULT_WEIGHTS  = "weights/efficientnet_b7_fold1_state_dict.pt"
FALLBACK_WEIGHTS = "efficientnet_b7_fold1_state_dict.pt"
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", DEFAULT_WEIGHTS)
if not os.path.exists(MODEL_WEIGHTS):
    if os.path.exists(FALLBACK_WEIGHTS):
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

@st.cache_resource(show_spinner=False)
def load_model(num_classes: int):
    model = create_model(
        "efficientnet_b7",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.0
    )
    model.eval()
    model.to(DEVICE)

    if not os.path.exists(MODEL_WEIGHTS):
        st.warning(
            f"Model weights not found at: {MODEL_WEIGHTS}\n"
            "Upload weights or set env MODEL_WEIGHTS to a valid path."
        )
        return model

    try:
        sd = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        # รองรับกรณี pack เป็น {'state_dict': {...}}
        if isinstance(sd, dict) and "state_dict" in sd and all(not k.startswith("model.") for k in sd["state_dict"].keys()):
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            st.info(f"Loaded with non-strict mode. Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    except Exception as e:
        st.error(
            "Failed to load model weights. Make sure it's a plain state_dict compatible with timm EfficientNet-B7 head.\n\n"
            f"{type(e).__name__}: {e}"
        )
    return model

def build_transform(size: int = IMAGE_SIZE):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def predict_image(model, img: Image.Image, classes):
    tfm = build_transform(IMAGE_SIZE)
    x = tfm(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    # แม็ป index → ชื่อคลาสตาม classes.txt
    idx_to_class = {i: c for i, c in enumerate(classes)}
    class_to_prob = OrderedDict((idx_to_class[i], float(p)) for i, p in enumerate(probs))
    # Top-1
    top_idx = int(torch.tensor(probs).argmax().item())
    top_label = idx_to_class[top_idx]
    top_conf  = probs[top_idx]
    return class_to_prob, top_label, top_conf

# =========================
# UI
# =========================
st.title("MRI-based Classification (EfficientNet-B7)")
st.caption("อัปโหลดภาพ แล้วกด **Predict** – ระบบจะแสดงทุกคลาสและเปอร์เซ็นต์ โดยเรียงตามที่คุณกำหนดไว้")

classes = load_classes()
model   = load_model(num_classes=len(classes))

# สร้างแผนที่ชื่อ (normalize) สำหรับการเรียง "แสดงผล"
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
            class_probs, top_label, top_conf = predict_image(model, image, classes)

        st.subheader("ผลลัพธ์")
        st.write(f"**Predicted:** {top_label} ({top_conf*100:.2f}%)")
        st.write(f"**Model:** EfficientNet-B7 (timm) | **Device:** {DEVICE.upper()}")

        st.markdown("### แสดงทุกคลาส (เรียงตามที่กำหนด)")
        rows = []
        for want_norm in desired_norm:
            if want_norm in norm_map:
                real_name = norm_map[want_norm]
                prob = class_probs.get(real_name, 0.0)
                rows.append((real_name, prob))
            else:
                rows.append((f"[Missing in classes.txt] {want_norm}", 0.0))

        for name, p in rows:
            st.write(f"- {name}: **{p*100:.2f}%**")

        st.markdown("### Top-k (เรียงจากมากไปน้อย)")
        sorted_items = sorted(class_probs.items(), key=lambda kv: kv[1], reverse=True)
        for name, p in sorted_items:
            st.write(f"- {name}: {p*100:.2f}%")

        mismatches = [dn for dn in desired_norm if dn not in norm_map]
        if mismatches:
            st.warning(
                "บางชื่อที่ต้องการแสดง ไม่พบใน classes.txt โปรดตรวจสะกด/ตัวพิมพ์:\n- " +
                "\n- ".join(mismatches)
            )
else:
    st.info("อัปโหลดภาพก่อน แล้วค่อยกด Predict")