# convert_ckpt.py
# แปลง checkpoint (Lightning/timm) -> state_dict เพียวๆ เพื่อ deploy ง่ายและปลอดภัย
# ทำเฉพาะกับไฟล์ที่คุณ "เชื่อถือ" เท่านั้น

import os, sys, types
import torch

SRC = "inception_v3_checkpoint_fold0.pt"    # ไฟล์ต้นฉบับ
DST = "inception_v3_fold0_state_dict.pt"    # ไฟล์ผลลัพธ์ (state_dict)

def allowlist_classes():
    """อนุญาตคลาสที่มักอยู่ใน ckpt ของ timm + lightning สำหรับ weights_only=True"""
    try:
        from torch.serialization import add_safe_globals
    except Exception:
        return
    # timm EfficientNet core class
    try:
        from timm.models.efficientnet import EfficientNet
        add_safe_globals([EfficientNet])
    except Exception as e:
        print(f"[allowlist] timm EfficientNet: {e}")
    # timm layers ที่เจอบ่อยใน EfficientNet
    try:
        from timm.layers.conv2d_same import Conv2dSame
        add_safe_globals([Conv2dSame])
    except Exception as e:
        print(f"[allowlist] timm Conv2dSame: {e}")
    # lightning wrapper
    try:
        from lightning.fabric.wrappers import _FabricModule
        add_safe_globals([_FabricModule])
    except Exception as e:
        print(f"[allowlist] lightning _FabricModule: {e}")

def extract_state_dict(obj):
    """ดึง state_dict ไม่ว่าจะมาในรูป dict, lightning ckpt หรือโมเดลจริงทั้งตัว"""
    # 1) ถ้าเป็น dict ครอบ state_dict
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    # 2) ถ้าเป็น dict ของ tensor/param อยู่แล้ว
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        sd = obj
    # 3) ถ้าเป็นโมเดลจริง (nn.Module) หรือวัตถุที่มีเมธอด state_dict()
    elif hasattr(obj, "state_dict") and isinstance(getattr(obj, "state_dict"), types.MethodType):
        try:
            sd = obj.state_dict()
        except Exception:
            # บางทีอยู่ใต้ .module (เช่น lightning/fabric)
            mod = getattr(obj, "module", None)
            if mod is not None and hasattr(mod, "state_dict"):
                sd = mod.state_dict()
            else:
                raise
    else:
        raise TypeError("ไม่รู้จักรูปแบบ ckpt: ต้องเป็น dict หรือโมเดลที่มี state_dict()")

    # ล้างคีย์ให้สะอาด
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for pref in ("model.", "module."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        if nk.endswith("num_batches_tracked"):
            continue
        new_sd[nk] = v
    return new_sd

def main():
    if not os.path.exists(SRC):
        print(f"ไม่พบไฟล์: {SRC}")
        sys.exit(1)

    # 1) พยายามโหลดแบบ weights_only=True (ปลอดภัยกว่า)
    allowlist_classes()
    ckpt = None
    try:
        ckpt = torch.load(SRC, map_location="cpu", weights_only=True)
        print("[OK] โหลดด้วย weights_only=True สำเร็จ")
    except Exception as e:
        print(f"[weights_only=True] ไม่สำเร็จ: {e}")

    # 2) ถ้ายังไม่ได้ → fallback เป็น weights_only=False (ทำเฉพาะไฟล์ที่คุณเชื่อถือ)
    if ckpt is None:
        print("WARNING: ใช้ weights_only=False (อนุญาต unpickle). ทำเฉพาะไฟล์ที่คุณเชื่อถือ!")
        ckpt = torch.load(SRC, map_location="cpu", weights_only=False)

    # 3) ดึง state_dict ไม่ว่าจะเป็น dict หรือโมเดลจริง
    sd = extract_state_dict(ckpt)

    # 4) เซฟเป็น state_dict เพียวๆ
    torch.save(sd, DST)
    size_mb = os.path.getsize(DST) / 1e6
    print(f"[DONE] เซฟ state_dict -> {DST}  ({size_mb:.2f} MB)")
    print("นำไฟล์นี้ไปใช้ในแอปได้เลย (torch.load(...), model.load_state_dict(...))")

if __name__ == "__main__":
    main()