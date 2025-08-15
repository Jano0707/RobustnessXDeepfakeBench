import cv2
import albumentations as A
from pathlib import Path
import argparse
import numpy as np

# ==============================
# Hilfsfunktionen
# ==============================

def load_image(image_path):
    return cv2.imread(str(image_path))

def save_image(image, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

def process_folder(input_folder, output_folder, transform):
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

# ==============================
# Manipulations-Funktionen
# ==============================

def black_white(input_path, output_path):
    transform = A.Compose([A.ToGray(p=1.0)])
    apply_transform(input_path, output_path, transform)

def rotate_90_left(input_path, output_path):
    transform = A.Compose([A.Rotate(limit=(90, 90), p=1.0)])
    apply_transform(input_path, output_path, transform)

def jpeg_compress(input_path, output_path, quality=50):
    transform = A.Compose([
        A.ImageCompression(quality_range=(quality, quality), compression_type="jpeg", p=1.0)
    ])
    apply_transform(input_path, output_path, transform)


def scale_up_25(input_path, output_path):
    def _transform_fn(image):
        h, w = image.shape[:2]
        return A.Resize(width=int(w * 1.25), height=int(h * 1.25), p=1.0)(image=image)["image"]
    apply_transform_func(input_path, output_path, _transform_fn)
"""
def scale_up_25(input_path, output_path):
    def _transform_fn(image):
        return A.Resize(width=256, height=256, p=1.0)(image=image)["image"]
    apply_transform_func(input_path, output_path, _transform_fn)
"""

def scale_down_25(input_path, output_path):
    def _transform_fn(image):
        h, w = image.shape[:2]
        return A.Resize(width=int(w * 0.75), height=int(h * 0.75))(image=image)["image"]
    apply_transform_func(input_path, output_path, _transform_fn)

class ADD(A.ImageOnlyTransform):
    """
    Schreibt mehrzeiligen Text zentriert im unteren Viertel des Bildes.
    - Automatischer Zeilenumbruch anhand der Bildbreite
    - Farbe: weiß/schwarz je nach Helligkeit des Hintergrunds
    """
    def __init__(self, text="Demo", font_scale=3, thickness=3,
                 margin_ratio=0.05, line_spacing=1.2, p=1.0):
        super(ADD, self).__init__(p=p)
        self.text = text
        self.font_scale = float(font_scale)
        self.thickness = int(thickness)
        self.margin_ratio = float(margin_ratio)      # Rand in % der kürzeren Bildkante
        self.line_spacing = float(line_spacing)      # Zeilenabstand (Multiplikator auf Texthöhe)

    def _wrap_text(self, text, max_width_px):
        """Bricht den Text in Zeilen um, die maximal max_width_px breit sind."""
        words = text.split()
        if not words:
            return [""]

        lines, cur = [], words[0]
        for w in words[1:]:
            trial = cur + " " + w
            (tw, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX,
                                         self.font_scale, self.thickness)
            if tw <= max_width_px:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    def apply(self, img, **params):
        h, w = img.shape[:2]
        margin_px = int(min(h, w) * self.margin_ratio)
        # maximal erlaubte Breite für eine Zeile
        max_line_width = max(10, w - 2 * margin_px)

        # Text in Zeilen umbrechen
        lines = self._wrap_text(self.text, max_line_width)

        # Höhe/Breite je Zeile bestimmen
        line_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                      self.font_scale, self.thickness)[0] for line in lines]
        line_h = max(s[1] for s in line_sizes) if line_sizes else 0
        gap = int(line_h * (self.line_spacing - 1.0))
        block_height = len(lines) * line_h + (len(lines) - 1) * gap

        # Basislinie für den Textblock: unteres fünftel, aber nicht unter den unteren Rand
        baseline_y = int(h * 0.8)
        # Oberkante des Blocks so setzen, dass der Block in den Bildbereich passt
        top_y = min(baseline_y, h - margin_px) - (block_height - line_h)

        # ROI für Farbauswahl (gesamter Textblock)
        x0 = margin_px
        x1 = w - margin_px
        y0 = max(0, top_y - line_h)                 # etwas über die Oberkante schauen
        y1 = min(h, top_y + block_height)
        roi = img[y0:y1, x0:x1]
        if roi.size > 0:
            mean_brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        else:
            mean_brightness = 128
        color = (0, 0, 0) if mean_brightness > 127 else (255, 255, 255)

        out = img.copy()
        y = top_y
        for (line, (tw, th)) in zip(lines, line_sizes):
            # zentrierte X‑Position je Zeile
            x = max(margin_px, (w - tw) // 2)
            y = min(h - margin_px, y)  # nicht unters Bildende
            out = cv2.putText(out, line, (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              self.font_scale, color, self.thickness, cv2.LINE_AA)
            y += th + gap

        return out


def add_text_to_image(input_path, output_path, text="Demo", font_scale=3):
    """
    CLI‑Wrapper:
    - schreibt 'text' zentriert im unteren Viertel,
    - nutzt Schriftgröße 'font_scale',
    - bricht Zeilen automatisch um.
    """
    transform = A.Compose([
        ADD(text=text, font_scale=font_scale, thickness=3,
            margin_ratio=0.05, line_spacing=1.2, p=1.0)
    ])
    apply_transform(input_path, output_path, transform)


# ==============================
# Allgemeine Apply-Funktionen
# ==============================

def apply_transform(input_path, output_path, transform):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.is_dir():
        process_folder(input_path, output_path, transform)
    else:
        image = load_image(input_path)
        result = transform(image=image)["image"]

        # Falls output nur ein Ordner ist -> Dateiname vom Input übernehmen
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

        # Falls output nur ein Ordner ist -> Dateiname vom Input übernehmen
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
    parser.add_argument("--text", type=str, default="Demo", help="Text für add_text_to_image")
    args = parser.parse_args()

    functions = {
        "black_white": black_white,
        "rotate_90_left": rotate_90_left,
        "jpeg_compress": lambda i, o: jpeg_compress(i, o, quality=args.quality),
        "scale_up_25": scale_up_25,
        "scale_down_25": scale_down_25,
        "add_text": lambda i, o: add_text_to_image(i, o, text=args.text),
    }

    if args.function not in functions:
        print(f"Unbekannte Funktion: {args.function}")
        print(f"Verfügbare Funktionen: {list(functions.keys())}")
        exit(1)

    functions[args.function](args.input, args.output)
