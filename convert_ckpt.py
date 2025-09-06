# convert_ckpt.py
import importlib
import torch
from pathlib import Path

IN_PATH  = Path("efficientnet_b7_checkpoint_fold1.pt")
OUT_PATH = Path("efficientnet_b7_state_dict.pt")

assert IN_PATH.exists(), f"Checkpoint not found: {IN_PATH.resolve()}"

# --- 1) อนุญาต safe globals ที่ใช้ตอนเซฟ (Lightning/Fabric) ---
try:
    from torch.serialization import add_safe_globals
    try:
        # พยายามนำเข้า _FabricModule เพื่อลงใน allowlist
        fabric_mod = importlib.import_module("lightning.fabric.wrappers")
        _FabricModule = getattr(fabric_mod, "_FabricModule", None)
        if _FabricModule is not None:
            add_safe_globals([_FabricModule])
            print("[OK] Allowlisted lightning.fabric.wrappers._FabricModule")
        else:
            print("[WARN] _FabricModule not found in lightning.fabric.wrappers (but we'll try to load anyway).")
    except Exception as e:
        print(f"[WARN] Cannot import/allowlist _FabricModule: {e}")
except Exception:
    # PyTorch เก่ากว่า หรือไม่มีฟังก์ชันนี้ ก็จะลองโหลดแบบปกติด้านล่าง
    pass

# --- 2) โหลด checkpoint แบบ weights_only=False (เพราะเราตั้งใจจะแปลง) ---
print("[INFO] Loading checkpoint with weights_only=False (do this only if you trust the file).")
obj = torch.load(str(IN_PATH), map_location="cpu", weights_only=False)

# --- 3) ดึง state_dict ออกมา ---
if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
    sd = obj["state_dict"]
elif isinstance(obj, dict):
    # บาง checkpoint ใช้คีย์อื่น เช่น 'model', 'net', 'weights'
    for k in ("model", "net", "weights"):
        if k in obj and isinstance(obj[k], dict):
            sd = obj[k]
            break
    else:
        # กรณีเป็น dict ของพารามิเตอร์ตรง ๆ
        sd = obj
elif hasattr(obj, "state_dict"):
    sd = obj.state_dict()
else:
    raise RuntimeError("Cannot extract state_dict from checkpoint.")

print(f"[OK] Extracted state_dict with {len(sd.keys())} keys")

# --- 4) เซฟเป็น pure state_dict ---
torch.save(sd, str(OUT_PATH))
print(f"[DONE] Saved pure state_dict to: {OUT_PATH.resolve()}")