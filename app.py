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

# ----- (Optional) ใช้ HF ถ้าตั้งค่าไว้ -----
try:
    from huggingface_hub import hf_hub_download
    _HAS_HF = True
except Exception:
    _HAS_HF = False


# =========================
# Page / Defaults
# =========================
st.set_page_config(page_title="MRI-based Classification (EfficientNet-B7)", layout="centered")

DEVICE      = "cpu"
IMAGE_SIZE  = 600
CLASSES_TXT = "classes.txt"
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# =========================
# Read Secrets/ENV (แก้ที่ Secrets จะไม่ต้องแก้โค้ด)
# =========================
def _get_secret_env(key, default=None):
    # ลองดึงจาก Streamlit Secrets ก่อน → ไม่มีก็ไป ENV
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

HF_REPO_ID   = _get_secret_env("HF_REPO_ID",   "")   # e.g. "FatornahBk/ad-severity-weights"
HF_FILENAME  = _get_secret_env("HF_FILENAME",  "")   # e.g. "efficientnet_b7_state_dict.pt"
HF_REPO_TYPE = _get_secret_env("HF_REPO_TYPE", "model")  # or "dataset"
HF_TOKEN     = _get_secret_env("HF_TOKEN",     None)

# ถ้าไม่ใช้ HF ให้ตั้ง MODEL_URL เป็นลิงก์ไฟล์ตรง (public)
MODEL_URL    = _get_secret_env("MODEL_URL",    "")   # e.g. "https://.../efficientnet_b7_state_dict.pt"

# พาธไฟล์ในเครื่อง (ตั้งตาม HF_FILENAME หรือชื่อที่ดึงจาก URL)
CKPT_LOCAL   = os.path.join(WEIGHTS_DIR, HF_FILENAME or os.path.basename(MODEL_URL) or "model.pt")


# =========================
# Utils
# =========================
def load_classes():
    if os.path.exists(CLASSES_TXT):
        with open(CLASSES_TXT, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        if classes:
            return classes
    return [f"Class {i}" for i in range(2)]  # fallback

def _download_via_hf(repo_id, filename, repo_type, token) -> str:
    if not _HAS_HF:
        raise RuntimeError("huggingface_hub ยังไม่ได้ติดตั้งใน environment (เพิ่มใน requirements.txt).")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token,
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False,
        cache_dir=None,
    )

def _download_via_url(url, out_path):
    import urllib.request
    req = urllib.request.Request(url)
    # ถ้าเป็น private แบบมี header token (กรณีพิเศษ)
    if HF_TOKEN and "huggingface.co" in url:
        req.add_header("Authorization", f"Bearer {HF_TOKEN}")
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        f.write(resp.read())
    return out_path

def ensure_weights_exist() -> str:
    """
    ดาวน์โหลดไฟล์น้ำหนัก 2 ทางเลือก:
      1) Hugging Face: ตั้งค่า HF_REPO_ID + HF_FILENAME (รองรับ private ด้วย HF_TOKEN)
      2) URL ตรง: ตั้ง MODEL_URL เป็นลิงก์ public (หรือใช้ Bearer ถ้าเป็น HF+token)
    คืน path ไฟล์น้ำหนักในเครื่อง
    """
    # ใช้ Hugging Face ถ้าตั้งค่าไว้ครบ
    if HF_REPO_ID and HF_FILENAME:
        st.write("⬇️ Downloading weights from Hugging Face…")
        local_path = _download_via_hf(HF_REPO_ID, HF_FILENAME, HF_REPO_TYPE, HF_TOKEN)
    elif MODEL_URL:
        st.write("⬇️ Downloading weights from direct URL…")
        local_path = _download_via_url(MODEL_URL, CKPT_LOCAL)
    else:
        raise RuntimeError(
            "ไม่ได้ตั้งค่าที่มาของน้ำหนักโมเดล\n"
            "- ตั้ง Secrets/ENV: HF_REPO_ID + HF_FILENAME (และ HF_TOKEN ถ้า private) "
            "หรือ\n- ตั้ง MODEL_URL เป็นลิงก์ไฟล์โดยตรง"
        )

    # กันกรณีได้ pointer/ไฟล์ไม่ครบ
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 10_000:
        raise RuntimeError(
            "Weight file ดูเหมือนไม่ถูกต้อง (ขนาดเล็กผิดปกติ). "
            "ตรวจ repo/ไฟล์ และสิทธิ์การเข้าถึงอีกครั้ง"
        )
    return local_path

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
      1) torch.load(weights_only=True) ถ้ารองรับ
      2) allow safe globals → torch.load ปกติ → ดึง state_dict จากคีย์ยอดนิยม
      3) ถ้าเป็นอ็อบเจ็กต์โมเดล → .state_dict()
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
            return obj, "pickle:raw_dict"
        if hasattr(obj, "state_dict"):
            return obj.state_dict(), "object.state_dict"
    except ModuleNotFoundError as e:
        st.error(
            f"Missing dependency while unpickling checkpoint: {e}. "
            "เพิ่มแพ็กเกจที่หายไปใน requirements.txt หรือแปลงเป็น pure state_dict"
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
    # 1) ดาวน์โหลดไฟล์น้ำหนัก
    ckpt_path = ensure_weights_exist()

    # 2) โหลดคลาส
    classes = load_classes()
    num_classes = len(classes)

    # 3) สร้างโมเดล (EfficientNet-B7)
    model = create_model("tf_efficientnet_b7_ns", pretrained=False, num_classes=num_classes)
    model.to(DEVICE)

    # 4) โหลดเช็กพอยต์
    sd, how = _load_checkpoint_safely(ckpt_path)
    if sd is None:
        raise RuntimeError("โหลดเช็กพอยต์ไม่สำเร็จ: แนะนำแปลงเป็น pure state_dict บนเครื่อง dev แล้วอัปโหลดใหม่")

    sd = _fix_state_dict_keys(sd)

    # 5) ใส่พารามิเตอร์
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

with st.expander("ℹ️ Setup tips", expanded=False):
    st.markdown(
        "- ตั้ง Secrets/ENV อย่างน้อย 1 ทาง:\n"
        "  - **HF_REPO_ID + HF_FILENAME** (+ HF_TOKEN ถ้า private) เพื่อโหลดจาก Hugging Face\n"
        "  - หรือ **MODEL_URL** เป็นลิงก์ไฟล์โดยตรง (public)\n"
        "- ถ้าเป็น private Hugging Face ใส่ `HF_TOKEN` ใน Secrets ด้วย\n"
        "- แนะนำใช้ไฟล์ที่แปลงเป็น **pure state_dict** เพื่อเลี่ยง dependency แปลก ๆ"
    )

try:
    model, classes = get_model_and_classes()
    st.success(f"Model loaded ✅ | Classes = {len(classes)}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded = st.file_uploader("อัปโหลดภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    from PIL import Image
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Input image", use_container_width=True)

    with st.spinner("Predicting..."):
        results = predict_image(model, classes, img, topk=3)
    st.subheader("ผลการทำนาย")
    for label, prob in results:
        st.write(f"- **{label}** : {prob:.4f}")
else:
    st.info("อัปโหลดภาพเพื่อเริ่มทำนาย")