# DeepfakeBench/analysis/build_konfmatrix.py

import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- Utils ----------
def _load_json(fp: Path):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_binary(y):
    y = np.asarray(y).squeeze()
    vals = np.unique(y)
    if set(vals.tolist()) == {0, 1}:
        return y.astype(int)
    if set(vals.tolist()) == {-1, 1}:
        return ((y + 1) // 2).astype(int)
    raise ValueError(f"y_true nicht binär (gefunden: {vals})")

def _compute_cm_and_stats(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "ACC": float(acc), "Precision": float(prec), "Recall": float(rec), "F1": float(f1),
    }

def _latex_escape(s: str) -> str:
    if s is None: return ""
    return (str(s).replace("&", r"\&").replace("%", r"\%")
                  .replace("_", r"\_").replace("#", r"\#")
                  .replace("{", r"\{").replace("}", r"\}")
                  .replace("$", r"\$").replace("^", r"\^{}")
                  .replace("~", r"\~{}"))

def _write_latex_cm(tp, fp, fn, tn, caption, out_tex: Path):
    """
    LaTeX im exakten Layout (NiceTabular) – wie von dir vorgegeben.
    RP=TP, FP=FP, FN=FN, RN=TN
    """
    tex = rf"""\definecolor{{
headerblue
}}{{RGB}}{{200,210,255}}
\definecolor{{
subblue
}}{{RGB}}{{230,235,255}}
\begin{{table}}[h]
  \centering
  \setlength{{\tabcolsep}}{{8pt}}
  \renewcommand{{\arraystretch}}{{3}}
  \begin{{NiceTabular}}{{>{{\centering\arraybackslash}}p{{2.3cm}}
                     >{{\raggedright\arraybackslash}}p{{1.8cm}}
                     >{{\centering\arraybackslash}}p{{2.1cm}}
                     >{{\centering\arraybackslash}}p{{2.1cm}}}}
    % Erste Headerzeile
    \RowStyle{{\rowcolor{{headerblue}}}}
    & & \multicolumn{{2}}{{c}}{{\textbf{{Echtes Label}}}} \\
    % Zweite Headerzeile: manuell einfärben
    \cellcolor{{headerblue}} & \cellcolor{{subblue}} & \cellcolor{{subblue}}\textbf{{Positiv}} & \cellcolor{{subblue}}\textbf{{Negativ}} \\
    % Linke Blockzelle + Daten
    \Block[fill=headerblue]{{2-1}}{{\rotatebox{{90}}{{\parbox{{2.6cm}}{{\centering \textbf{{Entscheidung}}\\\textbf{{des}}\\\textbf{{Detektors}}}}}}}}
      & \cellcolor{{subblue}}\textbf{{Positiv}} & {tp} & {fp} \\
      & \cellcolor{{subblue}}\textbf{{Negativ}} & {fn} & {tn} \\
  \end{{NiceTabular}}
  \caption{{{_latex_escape(caption)}}}
  \label{{tab:confusion-matrix}}
\end{{table}}
"""
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex, encoding="utf-8")

# ---------- PNG-Renderer (LaTeX-Look in Matplotlib) ----------
def _render_png_cm(tp, fp, fn, tn, caption: str, out_png: Path):
    """
    Zeichnet eine PNG, die dem LaTeX-Layout sehr nahe kommt:
    - headerblue: RGB(200,210,255)  -> #C8D2FF
    - subblue   : RGB(230,235,255)  -> #E6EBFF
    - gleiche Spaltenbreiten wie in der Vorlage (proportional zu cm-Werten)
    - links eine 2x1 Blockzelle mit vertikaler Beschriftung
    """
    HEADER = "#C8D2FF"
    SUB    = "#E6EBFF"
    EDGE   = "#D0D7DE"
    BLACK  = "#000000"

    # Spaltenbreiten im Verhältnis der cm-Angaben
    col_widths = [2.3, 1.8, 2.1, 2.1]
    total_w = sum(col_widths)

    # Zeilenhöhen (verhältnismäßig, damit es schön aussieht)
    h_header1 = 1.2
    h_header2 = 1.2
    h_data    = 1.4  # pro Datenreihe
    total_h = h_header1 + h_header2 + 2*h_data

    # Figurgröße - skaliert, damit Text gut lesbar ist
    scale = 0.9
    fig_w = total_w * scale
    fig_h = (total_h + 0.9) * scale  # + Platz für Caption

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h + 0.9)
    ax.axis("off")

    # Hilfsfunktionen
    def x_at(ci):  # linke Kante der Spalte ci
        return sum(col_widths[:ci])

    def draw_cell(x, y, w, h, face, text=None, ha="center", va="center",
                  weight="normal", rotation=0, fontsize=8.5, color=BLACK):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=EDGE, linewidth=0.8))
        if text is not None:
            ax.text(x + w/2, y + h/2, text, ha=ha, va=va, color=color,
                    fontsize=fontsize, weight=weight, rotation=rotation)

    # Koordinatensystem: y von unten nach oben; wir bauen von oben nach unten:
    y_top = total_h + 0.9

    # Header-Zeile 1 (RowStyle rowcolor headerblue)
    y = y_top - h_header1
    # Spalte 0 (leer, aber grau gefüllt) – in LaTeX steht hier nichts
    draw_cell(0, y, col_widths[0], h_header1, HEADER, text="", ha="center", va="center")
    # Spalte 1 (leer, aber grau)
    draw_cell(x_at(1), y, col_widths[1], h_header1, HEADER, text="", ha="center", va="center")
    # Spalten 2+3: zusammengefasster Header "Echtes Label"
    draw_cell(x_at(2), y, col_widths[2], h_header1, HEADER, text=None)
    draw_cell(x_at(3), y, col_widths[3], h_header1, HEADER, text=None)
    ax.text(x_at(2) + (col_widths[2]+col_widths[3])/2, y + h_header1/2,
            r"\textbf{Echtes Label}", ha="center", va="center", fontsize=9.5, weight="bold")
    y -= 0  # nur Info

    # Header-Zeile 2
    y = y - h_header2
    # Erste Zelle (Spalte 0) bleibt im headerblue (wie \cellcolor{headerblue})
    draw_cell(0, y, col_widths[0], h_header2, HEADER, text="")
    # Spalte 1: subblue + "Positiv" (links ausgerichtet laut LaTeX p-Spalte, aber wir nehmen fett zentriert)
    draw_cell(x_at(1), y, col_widths[1], h_header2, SUB, text=r"\textbf{Positiv}",
              fontsize=9.5, weight="bold")
    # Spalte 2: subblue + "Positiv"
    draw_cell(x_at(2), y, col_widths[2], h_header2, SUB, text=r"\textbf{Positiv}",
              fontsize=9.5, weight="bold")
    # Spalte 3: subblue + "Negativ"
    draw_cell(x_at(3), y, col_widths[3], h_header2, SUB, text=r"\textbf{Negativ}",
              fontsize=9.5, weight="bold")

    # Datenzeilen
    # Linke Blockzelle (2 Zeilen hoch, Spalte 0), headerblue, gedrehter Text
    y_block = y - 2*h_data
    draw_cell(0, y_block, col_widths[0], 2*h_data, HEADER, text=None)
    ax.text(0 + col_widths[0]/2, y_block + (2*h_data)/2,
            r"\textbf{Entscheidung}\\\textbf{des}\\\textbf{Detektors}",
            ha="center", va="center", fontsize=9.5, weight="bold", rotation=90)

    # Datenzeile 1 (Detector Positiv)
    y1 = y - h_data
    draw_cell(x_at(1), y1, col_widths[1], h_data, SUB, text=r"\textbf{Positiv}",
              ha="center", va="center", fontsize=9.5, weight="bold")
    draw_cell(x_at(2), y1, col_widths[2], h_data, "white", text=f"{int(tp)}")
    draw_cell(x_at(3), y1, col_widths[3], h_data, "white", text=f"{int(fp)}")

    # Datenzeile 2 (Detector Negativ)
    y2 = y1 - h_data
    draw_cell(x_at(1), y2, col_widths[1], h_data, SUB, text=r"\textbf{Negativ}",
              ha="center", va="center", fontsize=9.5, weight="bold")
    draw_cell(x_at(2), y2, col_widths[2], h_data, "white", text=f"{int(fn)}")
    draw_cell(x_at(3), y2, col_widths[3], h_data, "white", text=f"{int(tn)}")

    # Caption
    ax.text(total_w/2, 0.35, _latex_escape(caption), ha="center", va="center",
            fontsize=9.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def build_confusions_from_saved(results_dir: Path,
                                out_dir: Path,
                                invert_labels: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(Path(results_dir).rglob("*.json"))

    for jf in json_files:
        try:
            meta = _load_json(jf)
            y_true_path = Path(meta["y_true_path"])
            y_score_path = Path(meta["y_score_path"])
            if not y_true_path.exists() or not y_score_path.exists():
                print(f"[SKIP] Fehlende Dateien für {jf.name}")
                continue

            y_true = np.load(y_true_path).squeeze()
            y_score = np.load(y_score_path).astype(float).squeeze()
            y_true = _ensure_binary(y_true)
            if invert_labels:
                y_true = 1 - y_true

            # Feste Schwelle 0.5
            t = 0.5
            y_pred = (y_score >= t).astype(int)

            stats = _compute_cm_and_stats(y_true, y_pred)
            tp, fp, fn, tn = stats["TP"], stats["FP"], stats["FN"], stats["TN"]

            stem = jf.stem  # z.B. effort__Celeb-DF-v2-FACE__rob-Face-Smoothing

            caption = f"Konfusionsmatrix ({meta.get('detector','')} | {meta.get('dataset','')} | t=0.5)"

            # LaTeX
            _write_latex_cm(tp, fp, fn, tn, caption, out_dir / f"{stem}_confusion_0.5.tex")

            # PNG (LaTeX-Look in Matplotlib)
            _render_png_cm(tp, fp, fn, tn, caption, out_dir / f"{stem}_confusion_0.5.png")

            print(f"[OK] {stem}: TP={tp} FP={fp} FN={fn} TN={tn}")
        except Exception as e:
            print(f"[WARN] Confusion für {jf} übersprungen: {e}")

# ---------- CLI ----------
def main():
    # Gleicher CLI-Stil & Defaults wie create_tables.py (relativ zum CWD)
    ap = argparse.ArgumentParser(
        description="Erzeuge Konfusionsmatrizen (LaTeX + PNG) aus gespeicherten y_true/y_score."
    )
    ap.add_argument("--results_dir", type=str, default="analysis_outputs/metrics",
                    help="Ordner mit *.json (mit y_true_path/y_score_path).")
    ap.add_argument("--outdir", type=str, default="analysis_outputs/tables/confusions",
                    help="Zielordner für LaTeX/PNG.")
    ap.add_argument("--invert_labels", action="store_true",
                    help="Falls 1='Real' statt 'Fake' ist, Labels invertieren.")
    args = ap.parse_args()

    build_confusions_from_saved(
        results_dir=Path(args.results_dir),
        out_dir=Path(args.outdir),
        invert_labels=args.invert_labels
    )

if __name__ == "__main__":
    main()

