# app.py
import os
import io
import importlib
from collections import OrderedDict

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from timm import create_model
from huggingface_hub import hf_hub_download


# =========================
# Page / Constants
# =========================
st.set_page_config(page_title="MRI-based Classification (EfficientNet-B7)", layout="centered")

# <<< ปรับค่าให้ตรงของคุณ >>>
# ถ้าอัปโหลดไฟล์ที่ "แปลงเป็น state_dict แล้ว" แนะนำตั้งชื่อเป็น *_state_dict.pt
HF_REPO_ID   = "your-org/weights"              # เช่น "FatornahBk/ad-severity-weights"
HF_FILENAME  = "efficientnet_b7_state_dict.pt" # หรือ "efficientnet_b7_checkpoint_fold1.pt"
HF_REPO_TYPE = "model"                         # หรือ "dataset" ถ้าเก็บเป็น dataset
DEVICE       = "cpu"
IMAGE_SIZE   = 600
CLASSES_TXT  = "classes.txt"

WEIGHTS_DIR  = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# =========================
# Helper: read classes
# =========================
def load_classes():
    if os.path.exists(CLASSES_TXT):
        with open(CLASSES_TXT, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        if classes:
            return classes
    # Fallback ถ้าไม่มีไฟล์
    return [f"Class {i}" for i in range(2)]


# =========================
# HF download with token
# =========================
def ensure_weights_exist():
    """
    ดาวน์โหลดไฟล์น้ำหนักจาก Hugging Face (รองรับ private ด้วย HF_TOKEN)
    คืน path ไฟล์ที่ดาวน์โหลดมา (ในโฟลเดอร์ weights)
    """
    token = os.getenv("HF_TOKEN", None)
    # ลองอ่านจาก Streamlit Secrets ด้วย (Manage App → Secrets)
    if "HF_TOKEN" in st.secrets:
        token = st.secrets["HF_TOKEN"]

    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type=HF_REPO_TYPE,
        token=token,
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False,
        cache_dir=None,  # ให้ดึงลง local_dir ชัดเจน
    )

    # กันกรณีดาวน์โหลดได้ pointer เล็ก ๆ (ผิดปกติ)
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 10_000:
        raise RuntimeError(
            "Weight file looks invalid (too small). "
            "Check HF repo/file permissions or use a state_dict file."
        )
    return local_path


# =========================
# Safe checkpoint loading
# =========================
def _torch_supports_weights_only() -> bool:
    import inspect
    return "weights_only" in inspect.signature(torch.load).parameters

def _allow_safe_globals():
    """allowlist คลาสยอดฮิตเวลาบันทึกด้วย Lightning/Fabric"""
    try:
        from torch.serialization import add_safe_globals
    except Exception:
        return
    safe = []
    try:
        mod = importlib.import_module("lightning.fabric.wrappers")
        if hasattr(mod, "_FabricModule"):
            safe.append(getattr(mod, "_FabricModule"))
    except Exception:
        pass
    try:
        mod = importlib.import_module("pytorch_lightning.core.module")
        if hasattr(mod, "LightningModule"):
            safe.append(getattr(mod, "LightningModule"))
    except Exception:
        pass
    if safe:
        try:
            add_safe_globals(safe)
        except Exception:
            pass

def _load_checkpoint_safely(path: str):
    """
    คืน (state_dict, how)
    Strategy:
      1) torch.load(weights_only=True) ถ้าเวอร์ชันรองรับ
      2) allow safe globals แล้ว torch.load ปกติ → ดึง state_dict จากคีย์ยอดนิยม
      3) ถ้าได้เป็นอ็อบเจ็กต์ → ดึง .state_dict()
    """
    # 1) weights_only
    if _torch_supports_weights_only():
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(obj, dict):
                for k in ("state_dict", "model", "net", "weights"):
                    if k in obj and isinstance(obj[k], dict):
                        return obj[k], f"weights_only:{k}"
                return obj, "weights_only:raw_dict"
        except Exception:
            pass

    # 2) pickle ปลอดภัย
    _allow_safe_globals()
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return obj["state_dict"], "pickle:state_dict"
            for k in ("model", "net", "weights"):
                if k in obj and isinstance(obj[k], dict):
                    return obj[k], f"pickle:{k}"
            # dict ของพารามิเตอร์ตรง ๆ
            return obj, "pickle:raw_dict"
        if hasattr(obj, "state_dict"):
            return obj.state_dict(), "object.state_dict"
    except ModuleNotFoundError as e:
        st.error(
            f"Missing dependency while unpickling checkpoint: {e}. "
            "Add the missing package to requirements.txt or re-save as pure state_dict."
        )
        raise
    return None, "failed"

def _fix_state_dict_keys(sd: dict):
    """ลบ prefix 'module.' / 'model.' ออกจากคีย์"""
    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."): nk = nk[7:]
        if nk.startswith("model."):  nk = nk[6:]
        new_sd[nk] = v
    return new_sd


# =========================
# Build / Load model (cached)
# =========================
@st.cache_resource(show_spinner=True)
def get_model_and_classes():
    # 1) โหลดไฟล์จาก HF
    ckpt_path = ensure_weights_exist()

    # 2) โหลดคลาส
    classes = load_classes()
    num_classes = len(classes)

    # 3) สร้างโมเดล
    model = create_model("tf_efficientnet_b7_ns", pretrained=False, num_classes=num_classes)
    model.to(DEVICE)

    # 4) โหลดเช็กพอยต์
    sd, how = _load_checkpoint_safely(ckpt_path)
    if sd is None:
        raise RuntimeError(
            "Cannot load checkpoint safely. Consider converting to pure state_dict on your dev machine."
        )

    sd = _fix_state_dict_keys(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        st.warning(f"Missing keys: {list(missing)[:5]}{' ...' if len(missing)>5 else ''}")
    if unexpected:
        st.warning(f"Unexpected keys: {list(unexpected)[:5]}{' ...' if len(unexpected)>5 else ''}")

    model.eval()
    return model, classes


# =========================
# Inference utils
# =========================
def get_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

@torch.inference_mode()
def predict_image(model, classes, image: Image.Image, topk=3):
    x = get_transform()(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = probs.argsort()[::-1][:min(topk, len(classes))]
    return [(classes[i], float(probs[i])) for i in top_idx]


# =========================
# UI
# =========================
st.title("🧠 MRI-based Classification (EfficientNet-B7)")

with st.expander("ℹ️ Notes", expanded=False):
    st.markdown(
        "- น้ำหนักโมเดลจะถูกดาวน์โหลดจาก Hugging Face (รองรับ private ผ่าน HF_TOKEN)\n"
        "- ถ้าโหลดเช็กพอยต์ไม่ผ่าน ให้ตรวจ dependencies และพิจารณาแปลงเป็น **pure state_dict**\n"
        "- โมเดลใช้ tf_efficientnet_b7_ns จาก timm"
    )

# โหลดโมเดล (cached)
try:
    model, classes = get_model_and_classes()
    st.success(f"Model loaded ✅ | Classes = {len(classes)}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded = st.file_uploader("อัปโหลดภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Input image", use_container_width=True)

    with st.spinner("Predicting..."):
        results = predict_image(model, classes, img, topk=3)

    st.subheader("ผลการทำนาย")
    for label, prob in results:
        st.write(f"- **{label}** : {prob:.4f}")
else:
    st.info("อัปโหลดภาพเพื่อเริ่มทำนาย")