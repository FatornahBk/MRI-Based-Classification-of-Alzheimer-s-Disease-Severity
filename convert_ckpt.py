# convert_ckpt.py
import sys
import torch
from collections import OrderedDict

IN_PATH  = "inception_v3_checkpoint_fold0.pt"   # <-- ไฟล์ต้นฉบับของคุณ
OUT_PATH = "inception_v3_state_dict.pt"         # <-- ผลลัพธ์เป็น state_dict ล้วน

def load_with_allowlist(path: str):
    """
    โหลด checkpoint ที่ถูกห่อด้วย Lightning Fabric โดย allowlist คลาสที่จำเป็น
    และบังคับ weights_only=False เพื่อให้ unpickle ได้
    """
    try:
        # พยายามโหลดแบบปกติก่อน (ในบางเครื่องอาจผ่านได้เลย)
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print("[info] normal torch.load failed:", type(e).__name__, e)
        print("[info] retry with allowlisted lightning.fabric.wrappers._FabricModule ...")

    # ---- allowlist _FabricModule ของ Lightning Fabric ----
    try:
        from lightning.fabric.wrappers import _FabricModule
        from torch.serialization import add_safe_globals  # PyTorch 2.6+
        add_safe_globals([_FabricModule])
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print("[error] even with allowlist failed:", type(e).__name__, e)
        print("        - ตรวจว่าได้ติดตั้ง lightning แล้วหรือยัง: pip install lightning")
        print("        - ถ้าไฟล์ไม่ใช่ของที่ไว้ใจ ไม่ควร unpickle")
        sys.exit(1)

def main():
    print(f"Loading: {IN_PATH}")
    obj = load_with_allowlist(IN_PATH)

    # เคส 1: เป็น state_dict อยู่แล้ว
    if isinstance(obj, OrderedDict):
        sd = obj
        print("Detected: plain OrderedDict (state_dict).")

    # เคส 2: checkpoint dict ที่มี 'state_dict'
    elif isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        print("Detected: dict with 'state_dict'.")

    # เคส 3: เป็นโมเดลเต็ม (มี .state_dict())
    elif hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        sd = obj.state_dict()
        print("Detected: full model object -> extracted .state_dict().")

    else:
        print("\n[!] ไม่รู้จักรูปแบบ checkpoint:")
        print("    type(obj) =", type(obj))
        sys.exit(2)

    # ลอก prefix ที่พบบ่อย
    fixed = OrderedDict()
    for k, v in sd.items():
        nk = k
        for prefix in ("model.", "module.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        fixed[nk] = v

    torch.save(fixed, OUT_PATH)
    print(f"\n[OK] Saved plain state_dict to: {OUT_PATH}")

if __name__ == "__main__":
    main()