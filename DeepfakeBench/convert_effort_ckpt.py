import torch
import sys
from collections import OrderedDict

def strip_module_prefix(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def extract_state_dict(obj):
    # viele Checkpoints sind Diktate mit 'state_dict' oder 'model'
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "weights", "params"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    # falls direkt ein state_dict übergeben wurde
    if isinstance(obj, dict):
        return obj
    raise RuntimeError("Unerwartetes Checkpoint-Format")

def main(inp, outp):
    ckpt = torch.load(inp, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = strip_module_prefix(sd)
    # Optional: nur die für Effort relevanten Keys behalten (nicht nötig, aber sauber)
    # sd = {k: v for k, v in sd.items() if k.startswith(("backbone.", "head."))}
    torch.save(sd, outp)
    print(f"Konvertiert gespeichert nach: {outp}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/convert_effort_ckpt.py <input_ckpt.pth> <output_ckpt_stripped.pth>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
