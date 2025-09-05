# app.py
import streamlit as st
import torch
from PIL import Image
from prediction import pred_class
import os

# ---------------- UI ตามที่ต้องการ ----------------
st.set_page_config(
    page_title="MRI-Based Classification of Alzheimer's Disease Severity",
    layout="centered",
)
st.title("MRI-Based Classification of Alzheimer Disease Severity")
st.header("Please up load picture")

# ---------------- ปรับค่าที่นี่ให้ตรงกับโมเดล ----------------
CKPT_PATH = "efficientnet_b7_checkpoint_fold1.pt"   # ชื่อไฟล์เช็กพอยต์ของคุณ
MODEL_NAME = "tf_efficientnet_b7_ns"                # ถ้าเทรนด้วย efficientnet_b7 ให้เปลี่ยนเป็น "efficientnet_b7"

# รายชื่อคลาส (ตามงานของคุณ)
CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
NUM_CLASSES = len(CLASS_NAMES)

# ---------------- ตัวช่วย: ดึง state_dict จาก checkpoint ----------------
def _extract_state_dict(ckpt_obj):
    # 1) ถ้าเป็น dict ที่มี state_dict ซ้อนอยู่
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "weights", "net", "model"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        # 2) ถ้าเป็น dict ที่เป็น state_dict ตรงๆ
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    # 3) ถ้าเป็นวัตถุโมเดลทั้งตัว
    if hasattr(ckpt_obj, "state_dict"):
        return ckpt_obj.state_dict()
    return None

def _load_checkpoint_safely(path: str):
    """
    ลองโหลดหลายวิธีเพื่อให้รอดจาก PyTorch 2.6:
      A) weights_only=True (ปลอดภัยสุด)
      B) weights_only=True + allowlist Lightning (ถ้ามี)
      C) fallback weights_only=False (เฉพาะไฟล์ที่ไว้ใจ)
    """
    # A) ปลอดภัยสุด
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        sd = _extract_state_dict(ckpt)
        if sd is not None:
            return sd, "weights_only"
    except Exception:
        pass

    # B) allowlist Lightning (ถ้ามี)
    try:
        from torch.serialization import safe_globals
        try:
            import lightning.fabric.wrappers as lfw
            allow = [lfw._FabricModule]
        except Exception:
            allow = []
        if allow:
            with safe_globals(allow):
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
            sd = _extract_state_dict(ckpt)
            if sd is not None:
                return sd, "weights_only+allowlist"
    except Exception:
        pass

    # C) fallback (เฉพาะไฟล์ที่เชื่อถือได้จริง)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(ckpt)
    if sd is None:
        raise RuntimeError("Cannot extract state_dict from the checkpoint.")
    return sd, "unsafe"

# ---------------- โหลดโมเดล (แคช) ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(CKPT_PATH):
        st.error(f"Checkpoint not found: {CKPT_PATH}")
        st.stop()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # สร้างสถาปัตยกรรม EfficientNet-B7 จาก timm ให้ตรงกับตอนเทรน
    import timm
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

    # โหลดเฉพาะน้ำหนัก
    state_dict, how = _load_checkpoint_safely(CKPT_PATH)

    # ลบ prefix 'module.' จาก DataParallel ถ้ามี
    clean_state = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        clean_state[nk] = v

    # ใส่น้ำหนัก (strict=False กันเคสหัว classifier ชื่อไม่ตรง)
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing or unexpected:
        st.info(f"load_state_dict(strict=False): missing={len(missing)}, unexpected={len(unexpected)}  [{how}]")

    model.to(device).eval()
    return model

model = load_model()

# ---------------- อัปโหลด → แสดงรูป → ปุ่มทำนาย → แสดงผล ----------------
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Prediction'):
        top_idx, top_name, probs = pred_class(
            model=model,
            image=image,
            class_names=CLASS_NAMES,
            image_size=(224, 224),  # 224 เร็วขึ้น; ถ้าเทรนใหญ่กว่านี้ปรับได้
        )

        st.write("## Prediction Result")
        for i, name in enumerate(CLASS_NAMES):
            color = "blue" if i == top_idx else "inherit"
            st.write(
                f"## <span style='color:{color}'>{name} : {probs[i]*100:.2f}%</span>",
                unsafe_allow_html=True
            )