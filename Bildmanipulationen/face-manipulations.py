import cv2
import albumentations as A
from pathlib import Path
import argparse
import numpy as np

# ====== dlib optional & lazy laden ======
try:
    import dlib
except ImportError:
    dlib = None

_DLIB_FACE_DET = None
_DLIB_PREDICTOR = None
_DLIB_PREDICTOR_PATH = None

# Globale Optionen
FIXED_OUTLINE_PX = 1  # 0 = keine Outline; 1..2 = sehr dezent
EYES_WORK_SIZE = 1600

# =============================================================================
# I/O
# =============================================================================

def load_image(image_path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def save_image(image_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_bgr)

def process_folder(input_folder: Path, output_folder: Path, transform: A.Compose) -> None:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    for img_path in input_folder.rglob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        image = load_image(img_path)
        if image is None:
            continue
        transformed = transform(image=image)["image"]
        rel_path = img_path.relative_to(input_folder)
        save_path = output_folder / rel_path
        save_image(transformed, save_path)
        print(f"Gespeichert: {save_path}")

# =============================================================================
# Text-Helper
# =============================================================================

def _wrap_text_lines(text: str, font_scale: float, thickness: int, max_width_px: int):
    words = (text or "").split()
    if not words:
        return [""], [(0, 0)]
    lines, cur = [], words[0]
    sizes = []
    for w in words[1:]:
        trial = cur + " " + w
        (tw, th), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if tw <= max_width_px:
            cur = trial
        else:
            lines.append(cur)
            sizes.append(cv2.getTextSize(cur, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0])
            cur = w
    lines.append(cur)
    sizes.append(cv2.getTextSize(cur, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0])
    return lines, sizes

def _measure_block(text, fs, th, max_w, line_spacing):
    lines, sizes = _wrap_text_lines(text, fs, th, max_w)
    maxw = max((sz[0] for sz in sizes), default=0)
    line_h = max((sz[1] for sz in sizes), default=int(32 * fs))
    gap = int(line_h * (line_spacing - 1.0))
    block_h = len(lines) * line_h + (len(lines) - 1) * gap
    return lines, sizes, maxw, line_h, gap, block_h

def _fit_scale_to_width_and_height(text, base_fs, base_th, target_w, max_block_h, line_spacing):
    """
    Findet eine Schriftgröße, sodass:
      - längste Zeile ≈ target_w
      - Blockhöhe <= max_block_h
      - Strichstärke skaliert *proportional* zur Schrift (wie bei Bottom)
    """
    fs = max(0.1, float(base_fs))
    th = max(1, int(round(base_th)))
    # Breite iterativ annähern
    for _ in range(3):
        lines, sizes, maxw, *_ = _measure_block(text, fs, th, target_w, line_spacing)
        if maxw <= 0:
            break
        adj = target_w / max(1, maxw)
        fs *= adj
        th = max(1, int(round(base_th * (fs / max(1e-6, base_fs)))))
    # Höhe begrenzen
    for _ in range(2):
        lines, sizes, maxw, line_h, gap, block_h = _measure_block(text, fs, th, target_w, line_spacing)
        if block_h <= max_block_h or block_h == 0:
            break
        shrink = (max_block_h / block_h)
        fs = max(0.1, fs * shrink)
        th = max(1, int(round(base_th * (fs / max(1e-6, base_fs)))))
    return _measure_block(text, fs, th, target_w, line_spacing) + (fs, th)

def _pick_text_color_by_luma(img_rgb: np.ndarray) -> tuple:
    if img_rgb.size == 0:
        return (255, 255, 255)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return (0, 0, 0) if float(np.mean(gray)) > 127 else (255, 255, 255)

def _draw_line_with_outline(img, text, org, fs, color, th):
    if FIXED_OUTLINE_PX > 0:
        outline = (255, 255, 255) if color == (0, 0, 0) else (0, 0, 0)
        img = cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                          fs, outline, FIXED_OUTLINE_PX, cv2.LINE_AA)
    img = cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                      fs, color, th, cv2.LINE_AA)
    return img

# =============================================================================
# Bottom
# =============================================================================

def _draw_text_block_bottom(img_rgb: np.ndarray, text: str,
                            font_scale=3.0, thickness=3, margin_ratio=0.05,
                            line_spacing=1.2) -> np.ndarray:
    """
    Zentriert im unteren Bild – identische Logik für Fitting wie Eyes.
    """
    h, w = img_rgb.shape[:2]
    margin_px = int(min(h, w) * float(margin_ratio))
    avail_w = max(10, w - 2 * margin_px)
    target_w = int(avail_w * 0.90)          # ~90% der Breite
    max_block_h = int(h * 0.20)              # ~20% der Höhe

    lines, sizes, maxw, line_h, gap, block_h, fs, th = _fit_scale_to_width_and_height(
        text, base_fs=font_scale, base_th=thickness,
        target_w=target_w, max_block_h=max_block_h, line_spacing=line_spacing
    )

    # unten platzieren
    top_y = min(int(h * 0.95), h - margin_px) - (block_h - line_h)
    y = max(margin_px, min(h - margin_px - block_h, top_y))

    # Farbe aus lokaler ROI
    x0, x1 = margin_px, w - margin_px
    y0, y1 = max(0, y), min(h, y + block_h)
    color = _pick_text_color_by_luma(img_rgb[y0:y1, x0:x1])

    out = img_rgb.copy()
    cur_y = y
    for line, (tw, text_h) in zip(lines, sizes):
        baseline_y = cur_y + text_h
        x = max(margin_px, (w - tw) // 2)
        out = _draw_line_with_outline(out, line, (x, baseline_y), fs, color, th)
        cur_y += text_h + gap
    return out

# =============================================================================
# dlib / Augen-Region
# =============================================================================

def _ensure_dlib(shape_predictor_path: str):
    global _DLIB_FACE_DET, _DLIB_PREDICTOR, _DLIB_PREDICTOR_PATH
    if dlib is None:
        raise RuntimeError("dlib ist nicht installiert. Bitte `pip install dlib`.")
    if (_DLIB_PREDICTOR is None) or (_DLIB_PREDICTOR_PATH != shape_predictor_path):
        _DLIB_FACE_DET = dlib.get_frontal_face_detector()
        _DLIB_PREDICTOR = dlib.shape_predictor(shape_predictor_path)
        _DLIB_PREDICTOR_PATH = shape_predictor_path

def _eyes_center_and_box(shape):
    """
    center = (cx,cy) zwischen den Augen-Innenwinkeln;
    bounds = (x_min, x_max, y_min, y_max) der Augen+Brauen-Box.
    """
    n = int(getattr(shape, "num_parts")()) if callable(getattr(shape, "num_parts", None)) else int(getattr(shape, "num_parts", 0))
    if n < 68:
        return None, (None, None), (None, None, None, None)

    import numpy as _np
    pts = _np.array([[shape.part(i).x, shape.part(i).y] for i in range(n)], dtype=_np.float32)
    brows_idx = list(range(17, 27))
    eyes_idx  = list(range(36, 48))
    region = pts[brows_idx + eyes_idx]

    x_min, y_min = region.min(axis=0)
    x_max, y_max = region.max(axis=0)
    w = float(x_max - x_min)
    h = float(y_max - y_min)

    inner_left, inner_right = pts[39], pts[42]
    cx = float((inner_left[0] + inner_right[0]) / 2.0)
    eyes_mean_y  = float(pts[eyes_idx][:, 1].mean())
    brows_mean_y = float(pts[brows_idx][:, 1].mean())
    cy = (eyes_mean_y + brows_mean_y) / 2.0

    return (cx, cy), (w, h), (int(x_min), int(x_max), int(y_min), int(y_max))

def _find_eyes_anchor(img_rgb: np.ndarray, shape_predictor_path: str):
    _ensure_dlib(shape_predictor_path)
    rects = _DLIB_FACE_DET(img_rgb, 1)
    if not rects:
        return None, (None, None), (None, None, None, None)
    rect = max(rects, key=lambda r: r.width() * r.height())
    shape = _DLIB_PREDICTOR(img_rgb, rect)
    return _eyes_center_and_box(shape)

# =============================================================================
# Eyes (identisch zur Bottom-Logik; zusätzlich in Box klemmen)
# =============================================================================

def _draw_text_block_centered_scaled(img_rgb: np.ndarray, center_xy, text: str,
                                     region_w: float, region_h: float,
                                     bounds,  # (x_min, x_max, y_min, y_max)
                                     line_spacing: float = 1.2, margin_ratio: float = 0.05,
                                     base_font_scale: float = 3.0, base_thickness: int = 3):
    """
    Verwendet dieselbe Fitting-Logik wie Bottom.
    Der Textblock wird *innerhalb* der Augen+Brauen-Box gehalten und auf deren
    volle Breite gefittet (winziger Innenabstand, um Clipping zu vermeiden).
    """
    h, w_img = img_rgb.shape[:2]
    x_min, x_max, y_min, y_max = bounds
    if None in (x_min, x_max, y_min, y_max):
        # Fallback: wie Bottom unten
        return _draw_text_block_bottom(img_rgb, text, base_font_scale, base_thickness, margin_ratio, line_spacing)

    
    target_w = max(10, int(x_max - x_min))         # ≈ 100% Boxbreite
    max_block_h = max(10, int(y_max - y_min))      # ≤ Boxhöhe

    # Identische Fit-Logik wie Bottom
    lines, sizes, maxw, line_h, gap, block_h, fs, th = _fit_scale_to_width_and_height(
        text, base_fs=base_font_scale, base_th=base_thickness,
        target_w=target_w, max_block_h=max_block_h, line_spacing=line_spacing
    )

    # Block um center zentrieren, aber *innerhalb* der Box klemmen
    cx, cy = int(center_xy[0]), int(center_xy[1])
    margin_px = int(min(h, w_img) * margin_ratio)

    y_top_ideal = cy - block_h // 2
    y_top = max(y_min, min(y_max - block_h, y_top_ideal))
    y_top = max(margin_px, min(h - margin_px - block_h, y_top))

    # Farbe anhand der geplanten ROI (innerhalb der Box)
    roi = img_rgb[max(0, y_top):min(h, y_top + block_h),
                  max(0, x_min):min(w_img, x_max)]
    color = _pick_text_color_by_luma(roi)

    out = img_rgb.copy()
    cur_y = y_top
    for line, (tw, text_h) in zip(lines, sizes):
        baseline_y = cur_y + text_h
        x_ideal = cx - tw // 2
        x = max(x_min, min(x_max - tw, x_ideal))
        x = max(margin_px, min(w_img - margin_px - tw, x))
        out = _draw_line_with_outline(out, line, (x, baseline_y), fs, color, th)
        cur_y += text_h + gap
    return out

# =============================================================================
# Manipulations-Funktionen
# =============================================================================

def resize_256(input_path, output_path):
    def _transform_fn(image):
        return A.Resize(width=256, height=256, p=1.0)(image=image)["image"]
    apply_transform_func(input_path, output_path, _transform_fn)

def black_white(input_path: str, output_path: str) -> None:
    transform = A.Compose([A.ToGray(p=1.0)])
    apply_transform(input_path, output_path, transform)

def rotate_90_left(input_path: str, output_path: str) -> None:
    transform = A.Compose([A.Rotate(limit=(90, 90), p=1.0)])
    apply_transform(input_path, output_path, transform)

def jpeg_compress(input_path: str, output_path: str, quality: int = 40) -> None:
    transform = A.Compose([A.ImageCompression(quality_lower=quality, quality_upper=quality, p=1.0)])
    apply_transform(input_path, output_path, transform)

# ====== Variante 1: Text unten ======
class ADD_BOTTOM(A.ImageOnlyTransform):
    def __init__(self, text="Demo", font_scale=3, thickness=3,
                 margin_ratio=0.05, line_spacing=1.2, p=1.0):
        super().__init__(p=p)
        self.text = text
        self.font_scale = float(font_scale)
        self.thickness = int(thickness)
        self.margin_ratio = float(margin_ratio)
        self.line_spacing = float(line_spacing)

    def apply(self, img, **params):
        return _draw_text_block_bottom(
            img, self.text,
            font_scale=self.font_scale,
            thickness=self.thickness,
            margin_ratio=self.margin_ratio,
            line_spacing=self.line_spacing
        )

# ====== Variante 2: Text auf den Augen ======
class ADD_EYES(A.ImageOnlyTransform):
    def __init__(self, text="Demo", font_scale=3, thickness=3,
                 margin_ratio=0.05, line_spacing=1.2, shape_predictor=None, p=1.0):
        super().__init__(p=p)
        self.text = text
        self.font_scale = float(font_scale)
        self.thickness = int(thickness)
        self.margin_ratio = float(margin_ratio)
        self.line_spacing = float(line_spacing)
        self.shape_predictor = shape_predictor

    def apply(self, img, **params):
        # 1) auf 500×500 hochskalieren (Arbeitsbild)
        h0, w0 = img.shape[:2]
        if (w0, h0) != (EYES_WORK_SIZE, EYES_WORK_SIZE):
            work = cv2.resize(img, (EYES_WORK_SIZE, EYES_WORK_SIZE), interpolation=cv2.INTER_CUBIC)
        else:
            work = img

        # 2) Eyes-Anchor im Arbeitsbild suchen
        try:
            center, (box_w, box_h), bounds = _find_eyes_anchor(work, self.shape_predictor)
        except Exception:
            center, box_w, box_h, bounds = None, None, None, (None, None, None, None)

        # 3) Zeichnen (wie bisher) – auf dem Arbeitsbild!
        if (center is None) or (box_w is None) or (box_w <= 1):
            drawn = _draw_text_block_bottom(
                work, self.text,
                font_scale=self.font_scale,
                thickness=self.thickness,
                margin_ratio=self.margin_ratio,
                line_spacing=self.line_spacing
            )
        else:
            drawn = _draw_text_block_centered_scaled(
                work, center, self.text,
                region_w=box_w, region_h=box_h, bounds=bounds,
                line_spacing=self.line_spacing, margin_ratio=self.margin_ratio,
                base_font_scale=self.font_scale, base_thickness=self.thickness
            )

        # 4) zurück auf Ursprungsmaße skalieren
        if (w0, h0) != (EYES_WORK_SIZE, EYES_WORK_SIZE):
            drawn = cv2.resize(drawn, (w0, h0), interpolation=cv2.INTER_AREA)

        return drawn

def add_text(input_path, output_path, text="Demo", font_scale=3,
             place="bottom", shape_predictor=None):
    if place == "eyes":
        if dlib is None:
            print("[WARN] dlib nicht verfügbar – Fallback auf bottom.")
            transform = A.Compose([ADD_BOTTOM(text=text, font_scale=font_scale, thickness=3,
                                              margin_ratio=0.05, line_spacing=1.2, p=1.0)])
        else:
            transform = A.Compose([ADD_EYES(text=text, font_scale=font_scale, thickness=3,
                                            margin_ratio=0.05, line_spacing=1.2,
                                            shape_predictor=shape_predictor, p=1.0)])
    else:
        transform = A.Compose([ADD_BOTTOM(text=text, font_scale=font_scale, thickness=3,
                                          margin_ratio=0.05, line_spacing=1.2, p=1.0)])
    apply_transform(input_path, output_path, transform)

# =============================================================================
# Apply
# =============================================================================

def apply_transform(input_path, output_path, transform):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.is_dir():
        process_folder(input_path, output_path, transform)
    else:
        image = load_image(input_path)
        if image is None:
            return
        result = transform(image=image)["image"]

        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / input_path.name

        save_image(result, output_path)

def apply_transform_func(input_path, output_path, func):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.is_dir():
        for img_path in input_path.rglob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            image = load_image(img_path)
            if image is None:
                continue
            transformed = func(image)
            rel_path = img_path.relative_to(input_path)
            save_path = output_path / rel_path
            save_image(transformed, save_path)
            print(f"Gespeichert: {save_path}")
    else:
        image = load_image(input_path)
        result = func(image)
        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / input_path.name
        save_image(result, output_path)

# ==============================
# CLI
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bildmanipulationen mit Albumentations")
    parser.add_argument("--function", type=str, required=True, help="Name der Manipulationsfunktion")
    parser.add_argument("--input", type=str, required=True, help="Pfad zu Bild oder Ordner")
    parser.add_argument("--output", type=str, required=True, help="Pfad für Ausgabe")
    parser.add_argument("--quality", type=int, default=50, help="Qualität für JPEG-Kompression")
    parser.add_argument("--text", type=str, default="Demo", help="Text für add_text")
    parser.add_argument("--place", type=str, default="bottom", choices=["bottom", "eyes"],
                        help="Textposition: bottom oder eyes")
    parser.add_argument("--shape_predictor", type=str, default="../DeepfakeBench/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat",
                        help="Pfad zur dlib *.dat (nur für place=eyes)")

    args = parser.parse_args()

    functions = {
        "black_white": black_white,
        "rotate_90_left": rotate_90_left,
        "resize_256": resize_256,
        "jpeg": lambda i, o: jpeg_compress(i, o, quality=args.quality),
        "add_text": lambda i, o: add_text(i, o, text=args.text,
                                          font_scale=3.0,
                                          place=args.place,
                                          shape_predictor=args.shape_predictor),
    }

    if args.function not in functions:
        print(f"Unbekannte Funktion: {list(functions.keys())}")
        raise SystemExit(1)

    if args.function == "add_text" and args.place == "eyes" and dlib is not None:
        try:
            _ensure_dlib(args.shape_predictor)
        except Exception as e:
            print(f"[WARN] dlib Init fehlgeschlagen: {e}\n→ Fallback auf bottom.")
            args.place = "bottom"

    functions[args.function](args.input, args.output)

