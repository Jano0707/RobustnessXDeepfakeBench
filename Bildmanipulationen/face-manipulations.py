import cv2
import albumentations as A
from pathlib import Path
import argparse
import numpy as np

# =============================================================================
# Hilfsfunktionen (I/O & Batch-Verarbeitung)
# =============================================================================

def load_image(image_path: Path) -> np.ndarray:
    """
    Lädt ein Bild mit OpenCV als RGB-Array

    Anmerkung:
    - cv2.imread() liefert BGR (nicht RGB). Für JPEG/PNG-Dateien nicht zwingend 
    ein Problem, da diese Formate selbst kein „BGR“ oder „RGB“ speichern, sondern 
    einfach Pixelwerte in einer bestimmten Byte-Reihenfolge.
    - Dennoch konvertiere ich vorab nach RGB
    """
    img_bgr = cv2.imread(str(image_path))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def save_image(image: np.ndarray, output_path: Path) -> None:
    """
    Speichert ein Bild

    Maßnahmen:
    - Stellt sicher, dass der Zielordner existiert
    - Vor dem Speichern konvertiere ich erneut zu RGB
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_path), output_rgb)

def process_folder(input_folder: Path, output_folder: Path, transform: A.Compose) -> None:
    """
    Batch-Verarbeitung: iteriert rekursiv über input_folder, wendet 'transform'
    auf zulässige Bilddateien an und spiegelt die Ordnerstruktur nach output_folder
    """
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
# Manipulations-Funktionen
# =============================================================================

def black_white(input_path: str, output_path: str) -> None:
    """
    Konvertiert Bilder in Graustufen
    - Pixelmaße bleiben erhalten; weiterhin auch mit 3-Kanälen, Grauwert wird in
      allen Kanälen gesetzt
    """
    transform = A.Compose([A.ToGray(p=1.0)])
    apply_transform(input_path, output_path, transform)

def rotate_90_left(input_path: str, output_path: str) -> None:
    """
    Rotiert das Bild exakt um +90° (linksdrehend)
    - limit=(90,90) erzwingt deterministisch genau 90°
    """
    transform = A.Compose([A.Rotate(limit=(90, 90), p=1.0)])
    apply_transform(input_path, output_path, transform)

def jpeg_compress(input_path: str, output_path: str, quality: int = 40) -> None:
    """
    JPEG-Kompression mit konstanter Qualität
    - Pixelmaße bleiben unverändert; nur die visuelle Qualität/Artefakte ändern sich.
    """
    transform = A.Compose([
        A.ImageCompression(quality_range=(quality, quality),
                           compression_type="jpeg", p=1.0)
    ])
    apply_transform(input_path, output_path, transform)

"""
Test:
def resize_256(input_path, output_path):
    def _transform_fn(image):
        return A.Resize(width=256, height=256, p=1.0)(image=image)["image"]
    apply_transform_func(input_path, output_path, _transform_fn)
    
def scale_up_25(input_path, output_path):
    def _transform_fn(image):
        h, w = image.shape[:2]
        return A.Resize(width=int(w * 1.25), height=int(h * 1.25), p=1.0)(image=image)["image"]
    apply_transform_func(input_path, output_path, _transform_fn)

def scale_down_25(input_path, output_path):
    def _transform_fn(image):
        h, w = image.shape[:2]
        return A.Resize(width=int(w * 0.75), height=int(h * 0.75))(image=image)["image"]
    apply_transform_func(input_path, output_path, _transform_fn)
"""
    
class ADD(A.ImageOnlyTransform):
    """
    Fügt Text zentriert im unteren Fünftel des Bildes ein

    - Automatischer Zeilenumbruch anhand der Bildbreite
    - Farbe: weiß/schwarz je nach Helligkeit des Hintergrunds
    """
    def __init__(self, text="Demo", font_scale=4, thickness=5,
                 margin_ratio=0.05, line_spacing=1.2, p=1.0):
        super(ADD, self).__init__(p=p)
        self.text = text
        self.font_scale = float(font_scale)
        self.thickness = int(thickness)
        self.margin_ratio = float(margin_ratio)
        self.line_spacing = float(line_spacing)

    def _wrap_text(self, text, max_width_px):
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
        max_line_width = max(10, w - 2 * margin_px)
        lines = self._wrap_text(self.text, max_line_width)

        line_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                      self.font_scale, self.thickness)[0] for line in lines]
        line_h = max(s[1] for s in line_sizes) if line_sizes else 0
        gap = int(line_h * (self.line_spacing - 1.0))
        block_height = len(lines) * line_h + (len(lines) - 1) * gap

        baseline_y = int(h * 0.8)
        top_y = min(baseline_y, h - margin_px) - (block_height - line_h)

        x0 = margin_px
        x1 = w - margin_px
        y0 = max(0, top_y - line_h)
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
            x = max(margin_px, (w - tw) // 2)
            y = min(h - margin_px, y)
            out = cv2.putText(out, line, (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              self.font_scale, color, self.thickness, cv2.LINE_AA)
            y += th + gap

        return out

def add_text_to_image(input_path, output_path, text="Demo", font_scale=4):
    """
    CLI-Wrapper für die Textoperation:
    - Ergänzt einen Textblock (zentriert, unten)
    """
    transform = A.Compose([
        ADD(text=text, font_scale=font_scale, thickness=5,
            margin_ratio=0.05, line_spacing=1.2, p=1.0)
    ])
    apply_transform(input_path, output_path, transform)

"""
Um den Text auf dem Mund zu fixieren, muss erst noch mit Landmarks gearbeitet werden,
so einfach grobe Schätzung auf Höhe 70% des Bildes

class ADD_Mouth(A.ImageOnlyTransform):
    """
"""
    Schreibt mehrzeiligen Text zentriert auf Mundhöhe (ca. 70% der Bildhöhe).
    - Automatischer Zeilenumbruch anhand der Bildbreite
    - Farbe: weiß/schwarz je nach Helligkeit des Hintergrunds
    """
"""
    def __init__(self, text="Demo", font_scale=3, thickness=3,
                 margin_ratio=0.05, line_spacing=1.2, p=1.0):
        super(ADD_Mouth, self).__init__(p=p)
        self.text = text
        self.font_scale = float(font_scale)
        self.thickness = int(thickness)
        self.margin_ratio = float(margin_ratio)
        self.line_spacing = float(line_spacing)

    def _wrap_text(self, text, max_width_px):
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
        max_line_width = max(10, w - 2 * margin_px)
        lines = self._wrap_text(self.text, max_line_width)

        line_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                      self.font_scale, self.thickness)[0] for line in lines]
        line_h = max(s[1] for s in line_sizes) if line_sizes else 0
        gap = int(line_h * (self.line_spacing - 1.0))
        block_height = len(lines) * line_h + (len(lines) - 1) * gap

        # Mundhöhe = etwa 70% der Bildhöhe
        baseline_y = int(h * 0.70)
        top_y = baseline_y - block_height // 2

        # ROI für Farbanalyse
        x0 = margin_px
        x1 = w - margin_px
        y0 = max(0, top_y)
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
            x = max(margin_px, (w - tw) // 2)
            y = min(h - margin_px, y)
            out = cv2.putText(out, line, (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              self.font_scale, color, self.thickness, cv2.LINE_AA)
            y += th + gap

        return out


def add_text_on_mouth(input_path, output_path, text="Demo", font_scale=3):
    transform = A.Compose([
        ADD_Mouth(text=text, font_scale=font_scale, thickness=3,
                  margin_ratio=0.05, line_spacing=1.2, p=1.0)
    ])
    apply_transform(input_path, output_path, transform)
"""

# =============================================================================
# Allgemeine Apply-Funktionen
# =============================================================================

def apply_transform(input_path, output_path, transform):
    """
    Führt eine Albumentations-Transformation auf:
    - Einem einzelnen Bild ODER
    - Rekursiv auf einem Ordner (Delegation an process_folder).

    I/O-Verhalten:
    - Wenn 'output_path' ein Ordner ist (oder keine Dateiendung besitzt),
      wird der Quelldateiname übernommen
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.is_dir():
        process_folder(input_path, output_path, transform)
    else:
        image = load_image(input_path)
        result = transform(image=image)["image"]

        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / input_path.name

        save_image(result, output_path)

def apply_transform_func(input_path, output_path, func):
    """
    Wie 'apply_transform', aber für freie Funktions-Transforms.
    Eignet sich für einfache Resize/Custom-Operationen, die direkt ein Bild -> Bild abbilden.
    """
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
    parser.add_argument("--text", type=str, default="Demo", help="Text für add_text_to_image")
    args = parser.parse_args()

    functions = {
        "black_white": black_white,
        "rotate_90_left": rotate_90_left,
        "jpeg_compress": lambda i, o: jpeg_compress(i, o, quality=args.quality),
        #"resize_256": resize_256,
        #"scale_up_25": scale_up_25,
        #"scale_down_25": scale_down_25,
        "add_text": lambda i, o: add_text_to_image(i, o, text=args.text),
        #"add_text_mouth": lambda i, o: add_text_on_mouth(i, o, text=args.text),
    }

    if args.function not in functions:
        print(f"Unbekannte Funktion: {args.function}")
        print(f"Verfügbare Funktionen: {list(functions.keys())}")
        exit(1)

    functions[args.function](args.input, args.output)
