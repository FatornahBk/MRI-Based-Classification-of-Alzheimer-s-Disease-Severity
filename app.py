# app.py
import os
import io
import urllib.request
import importlib
from collections import OrderedDict

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from timm import create_model


# =========================
# Configs
# =========================
st.set_page_config(page_title="MRI-based Classification (EfficientNet-B7)", layout="centered")

MODEL_URL  = "https://huggingface.co/your-org/weights/resolve/main/efficientnet_b7_state_dict.pt"
WEIGHTS_DIR = "weights"
CKPT_PATH  = os.path.join(WEIGHTS_DIR, "efficientnet_b7_state_dict.pt")

CLASSES_TXT = "classes.txt"   # หนึ่งคลาสต่อหนึ่งบรรทัด
IMAGE_SIZE  = 600             # tf_efficientnet_b7_ns ใช้ 600x600
DEVICE      = "cpu"           # ใช้ CPU บน Streamlit Cloud


# =========================
# Utils
# =========================
def ensure_weights_exist():
    """ดาวน์โหลดไฟล์โมเดล ถ้าไม่มี หรือขนาดเล็กผิดปกติ (เช่นเป็น LFS pointer)"""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if (not os.path.exists(CKPT_PATH)) or (os.path.getsize(CKPT_PATH) < 10_000):
        st.write("⬇️ Downloading model weights...")
        urllib.request.urlretrieve(MODEL_URL, CKPT_PATH)
    # กันไฟล์ที่โหลดมาไม่ครบ
    if os.path.getsize(CKPT_PATH) < 10_000:
        raise RuntimeError("Downloaded weight file looks invalid (too small). Check MODEL_URL or hosting permissions.")


def _torch_supports_weights_only() -> bool:
    import inspect
    return "weights_only" in inspect.signature(torch.load).parameters


def _allow_safe_globals():
    """อนุญาต safe globals สำหรับเช็คพอยต์ที่เซฟจาก Lightning/Fabric"""
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
    พยายามโหลดเช็คพอยต์อย่างปลอดภัยและคืนค่า state_dict
    Strategy:
      1) weights_only=True (ถ้าได้)
      2) allow safe globals แล้ว torch.load ปกติ → ดึง state_dict จากคีย์ยอดนิยม
      3) ถ้าเป็นอ็อบเจ็กต์โมเดล → ดึง .state_dict()
    """
    # 1) weights_only=True (ถ้ามีในเวอร์ชัน PyTorch)
    if _torch_supports_weights_only():
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(obj, dict) and any(k in obj for k in ("state_dict", "model", "net", "weights")):
                for k in ("state_dict", "model", "net", "weights"):
                    if k in obj and isinstance(obj[k], dict):
                        return obj[k], "weights_only:" + k
            if isinstance(obj, dict):
                return obj, "weights_only:raw_dict"
        except Exception:
            pass

    # 2) allow safe globals แล้วโหลดแบบปกติ
    _allow_safe_globals()
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return obj["state_dict"], "pickle:state_dict"
            for k in ("model", "net", "weights"):
                if k in obj and isinstance(obj[k], dict):
                    return obj[k], f"pickle:{k}"
            # บางกรณีเป็น dict ของพารามิเตอร์ตรง ๆ
            return obj, "pickle:raw_dict"
        if hasattr(obj, "state_dict"):
            return obj.state_dict(), "object.state_dict"
    except ModuleNotFoundError as e:
        st.error(f"Missing dependency while unpickling checkpoint: {e}. "
                 f"Add the missing package to requirements.txt or re-save as pure state_dict.")
        raise
    return None, "failed"


def _fix_state_dict_keys(sd: dict):
    """ลบ prefix ที่พบบ่อย ('module.', 'model.') ออกจากคีย์ของ state_dict"""
    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_sd[nk] = v
    return new_sd


def load_classes():
    if os.path.exists(CLASSES_TXT):
        with open(CLASSES_TXT, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        if classes:
            return classes
    # fallback
    return [f"Class {i}" for i in range(2)]  # ปรับตามงานจริงของคุณ


# =========================
# Model loader (cached)
# =========================
@st.cache_resource(show_spinner=True)
def get_model_and_classes():
    ensure_weights_exist()

    # 1) สร้างสถาปัตยกรรม (num_classes อิงจาก classes.txt)
    classes = load_classes()
    num_classes = len(classes)

    # tf_efficientnet_b7_ns = EfficientNet-B7 (Noisy Student)
    model = create_model("tf_efficientnet_b7_ns", pretrained=False, num_classes=num_classes)
    model.to(DEVICE)

    # 2) โหลดเช็คพอยต์แบบปลอดภัย → state_dict
    sd, how = _load_checkpoint_safely(CKPT_PATH)
    if sd is None:
        raise RuntimeError("Cannot load checkpoint safely. Consider re-saving as pure state_dict (only weights).")

    sd = _fix_state_dict_keys(sd)

    # 3) โหลดพารามิเตอร์ (allow missing/unexpected สำหรับความเข้ากันได้)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        st.warning(f"Missing keys in state_dict: {list(missing)[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        st.warning(f"Unexpected keys in state_dict: {list(unexpected)[:5]}{'...' if len(unexpected)>5 else ''}")

    model.eval()
    return model, classes


# =========================
# Preprocessing & Inference
# =========================
def get_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        # EfficientNet expects float32 in [0,1]; timm model has own norm inside in many cases,
        # แต่ถ้าต้อง normalize เอง ใช้ mean/std ของ ImageNet:
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

@torch.inference_mode()
def predict_image(model, classes, image: Image.Image, topk=3):
    tfm = get_transform()
    x = tfm(image.convert("RGB")).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = probs.argsort()[::-1][:topk]
    return [(classes[i], float(probs[i])) for i in top_idx]


# =========================
# Streamlit UI
# =========================
st.title("🧠 MRI-based Classification (EfficientNet-B7)")

with st.expander("ℹ️ Notes", expanded=False):
    st.markdown(
        "- โมเดลจะถูกดาวน์โหลดอัตโนมัติจาก URL ที่กำหนด (กันปัญหา LFS pointer บน GitHub)\n"
        "- ถ้าโหลดเช็คพอยต์ไม่ผ่าน ให้ตรวจ `requirements.txt` ว่ามีแพ็กเกจที่จำเป็นครบ เช่น `timm`, `torch`, `lightning` (ถ้าเคยใช้)\n"
        "- ถ้ามาจาก Lightning/Fabric การโหลดจะ allowlist คลาสที่จำเป็นชั่วคราวให้"
    )

# โหลดโมเดล (cached)
try:
    model, classes = get_model_and_classes()
    st.success(f"Model loaded. Classes = {len(classes)}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded = st.file_uploader("อัปโหลดภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Input image", use_container_width=True)

    with st.spinner("Predicting..."):
        results = predict_image(model, classes, img, topk=min(3, len(classes)))
    st.subheader("ผลการทำนาย")
    for label, prob in results:
        st.write(f"- **{label}** : {prob:.4f}")
else:
    st.info("อัปโหลดภาพเพื่อเริ่มทำนาย")