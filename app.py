import os
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from timm import create_model

# ---------------- CONFIG ----------------
DEVICE = "cpu"
IMAGE_SIZE = 600
CLASSES_TXT = "classes.txt"
WEIGHTS_DIR = "weights"
WEIGHTS_FILE = "weights/efficientnet_b7_fold1_state_dict.pt"

# ตอนโหลด (ตอนนี้ไม่ต้องใส่ weights_only=False แล้ว)
sd = torch.load(WEIGHTS_FILE, map_location=DEVICE)  # 2.6 ดีฟอลต์ weights_only=True โอเค
model.load_state_dict(sd, strict=False)   # ใช้ไฟล์ที่แปลงแล้ว
MODEL_NAME = "tf_efficientnet_b7_ns (fold1)"

# ลำดับที่ผู้ใช้ต้องการให้โชว์ผลลัพธ์ (เรียงคงที่)
TARGET_ORDER = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
]

# ---------------- LOAD CLASSES ----------------
@st.cache_resource(show_spinner=False)
def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

# ---------------- MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_model(weights_path, num_classes):
    model = create_model("tf_efficientnet_b7_ns", pretrained=False, num_classes=num_classes)
    model.to(DEVICE)
    # โหลด state_dict เพียวๆ ที่เราแปลงมาแล้ว
    sd = torch.load(weights_path, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        st.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    model.eval()
    return model

# ---------------- TRANSFORM ----------------
@st.cache_resource(show_spinner=False)
def get_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def predict_full(model, img, class_names):
    """คืน dict {class_name: prob_float} ของทุกคลาส"""
    tfm = get_transform()
    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    # จับคู่ชื่อคลาสกับ prob
    out = {cls: float(p) for cls, p in zip(class_names, probs)}
    return out

# ---------------- UI ----------------
st.set_page_config(page_title="MRI-based Classification (EfficientNet-B7)")
st.title("MRI-Based Classification of Alzheimer's Disease Severity")
st.caption(f"Model: **{MODEL_NAME}**")

# โหลดชื่อคลาส + เช็คไฟล์น้ำหนัก
class_names = load_class_names(CLASSES_TXT)
weights_path = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)
if not os.path.exists(weights_path):
    st.error(f"ไม่พบไฟล์น้ำหนัก {weights_path}")
    st.stop()

# แจ้งเตือนถ้าจำนวน/ชื่อคลาสไม่ตรงกับ TARGET_ORDER
if len(class_names) != len(TARGET_ORDER):
    st.warning(f"จำนวนคลาสใน classes.txt ({len(class_names)}) ไม่ตรงกับ TARGET_ORDER ({len(TARGET_ORDER)})")

# สร้าง mapping แบบไม่สนช่องว่าง/ตัวพิมพ์ใหญ่เล็ก เพื่อความทนทาน
norm = lambda s: " ".join(s.split()).strip().lower()
class_names_norm = {norm(c): c for c in class_names}
target_norm = [norm(t) for t in TARGET_ORDER]
missing_in_model = [t for t in target_norm if t not in class_names_norm]
if missing_in_model:
    st.warning("พบชื่อคลาสใน TARGET_ORDER ที่ไม่อยู่ใน classes.txt: " +
               ", ".join([TARGET_ORDER[target_norm.index(m)] for m in missing_in_model]))

model = load_model(weights_path, num_classes=len(class_names))

# เก็บรูป/ผลลัพธ์ใน session_state
if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None
if "dist" not in st.session_state:
    st.session_state.dist = None

uploaded = st.file_uploader("อัปโหลดภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    st.session_state.uploaded_img = Image.open(uploaded).convert("RGB")
    st.image(st.session_state.uploaded_img, caption="ภาพที่อัปโหลด", use_column_width=True)
    st.session_state.dist = None  # เคลียร์ผลเดิมเมื่อเลือกรูปใหม่

# ปุ่ม Predict
btn = st.button("🔮 Predict", type="primary", disabled=st.session_state.uploaded_img is None)
if btn and st.session_state.uploaded_img is not None:
    with st.spinner("กำลังพยากรณ์..."):
        full_dist = predict_full(model, st.session_state.uploaded_img, class_names)
        st.session_state.dist = full_dist

# แสดงผลลัพธ์ "ทุกคลาส" ตามลำดับที่กำหนด
if st.session_state.dist:
    st.subheader("ผลลัพธ์ตามลำดับที่กำหนด")
    # แสดงทีละคลาสตาม TARGET_ORDER
    for t_name in TARGET_ORDER:
        key = norm(t_name)
        # หากชื่อใน TARGET_ORDER ไม่ตรงกับ classes.txt แบบเป๊ะ ให้หาแบบ normalize
        if key in class_names_norm:
            actual_name = class_names_norm[key]
            p = st.session_state.dist.get(actual_name, 0.0)
        else:
            # ไม่พบชื่อคลาสนี้ในโมเดล → โชว์ 0% และเตือนด้านบนแล้ว
            p = 0.0
        st.markdown(f"- **{t_name}**: {p*100:.2f}%")
        st.progress(min(max(p, 0.0), 1.0))

    # สรุปคลาสที่น่าจะใช่มากที่สุด (Top-1)
    top_class = max(st.session_state.dist.items(), key=lambda kv: kv[1])
