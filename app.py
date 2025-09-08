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
WEIGHTS_FILE = "efficientnet_b7_fold1_state_dict.pt"   # ใช้ไฟล์ที่แปลงแล้ว

# ---------------- LOAD CLASSES ----------------
@st.cache_resource
def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

# ---------------- MODEL ----------------
@st.cache_resource
def load_model(weights_path, num_classes):
    model = create_model("tf_efficientnet_b7_ns", pretrained=False, num_classes=num_classes)
    model.to(DEVICE)

    # โหลด state_dict เพียวๆ
    sd = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# ---------------- TRANSFORM ----------------
@st.cache_resource
def get_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def predict(model, img, class_names, topk=3):
    tfm = get_transform()
    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        scores, idxs = probs.topk(min(topk, len(class_names)))
        return [(class_names[i], float(scores[j])) for j, i in enumerate(idxs)]

# ---------------- UI ----------------
st.set_page_config(page_title="MRI-based Classification (EfficientNet-B7 fold1)")
st.title("EfficientNet-B7 (fold1) — Streamlit Demo")

class_names = load_class_names(CLASSES_TXT)
weights_path = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)

if not os.path.exists(weights_path):
    st.error(f"ไม่พบไฟล์น้ำหนัก {weights_path}")
    st.stop()

model = load_model(weights_path, num_classes=len(class_names))

uploaded = st.file_uploader("อัปโหลดภาพ (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)
    with st.spinner("กำลังพยากรณ์..."):
        results = predict(model, img, class_names, topk=3)
    st.subheader("ผลลัพธ์ (Top-3)")
    for cls, p in results:
        st.write(f"- **{cls}**: {p:.3f}")