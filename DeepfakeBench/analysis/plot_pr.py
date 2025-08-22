#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precision�Recall-Kurven zeichnen (pro Dataset) aus DeepfakeBench-Ausgaben.

Nutzt:
- JSON-Metriken (wie in analysis_outputs/metrics erzeugt)
- zugehörige y_true.npy / y_score.npy (wird automatisch gesucht oder aus JSON gelesen)

Ausgabe:
- <outdir>/pr_<DATASET>.png / .pdf
- <outdir>/pr_<DATASET>__curvepoints.csv (Recall/Precision je Kurve)

Aufrufbeispiel:
python analysis/plot_pr.py \
  --results_dir analysis_outputs/metrics \
  --outdir analysis_outputs/plots/pr
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Hilfsfunktionen
# ---------------------------
def _lower(d): return {str(k).lower(): v for k, v in d.items()}

def _pretty_detector(name: str) -> str:
    m = {"effort": "Effort", "xception": "Xception"}
    s = (name or "").strip()
    return m.get(s.lower(), s[:1].upper() + s[1:])

def _infer_from_name(stem: str, idx: int, default=None):
    parts = stem.split("__")
    return parts[idx] if 0 <= idx < len(parts) else default

def load_metric_file(fp: Path):
    obj = json.loads(Path(fp).read_text(encoding="utf-8"))
    lo  = _lower(obj)
    det = (lo.get("detector") or lo.get("model") or _infer_from_name(fp.stem,0,"unknown")).strip()
    dset= (lo.get("dataset")  or _infer_from_name(fp.stem,1,"unknown")).strip()
    tag = (lo.get("tag")      or _infer_from_name(fp.stem,2,"baseline")).strip()
    # optional: Pfade zu Predictions direkt im JSON
    ytrue = lo.get("y_true_path") or lo.get("ytrue_path")
    yscore= lo.get("y_score_path") or lo.get("yscore_path")
    return {
        "detector": _pretty_detector(det),
        "dataset": dset,
        "tag": tag,
        "__file__": str(fp),
        "y_true_path": ytrue,
        "y_score_path": yscore,
    }

def load_all_json(results_dir: Path) -> pd.DataFrame:
    files = list(Path(results_dir).rglob("*.json"))
    if not files: raise SystemExit("Keine JSON-Dateien gefunden.")
    rows = []
    for f in files:
        try: rows.append(load_metric_file(f))
        except Exception as e:
            print(f"[WARN] Überspringe {f}: {e}")
    if not rows: raise SystemExit("Keine verwertbaren JSON-Dateien.")
    df = pd.DataFrame(rows)
    # Reihenfolge der Datasets so, wie sie vorkommen
    order = list(dict.fromkeys(df["dataset"].tolist()))
    df["dataset"] = pd.Categorical(df["dataset"], categories=order, ordered=True)
    return df

def _guess_pred_files(search_root: Path, det: str, dset: str, tag: str):
    """Robustes Suchen nach *_y_true.npy / *_y_score.npy in/unter search_root."""
    candidates = []
    candidates += list(search_root.rglob(f"*{det}*{dset}*{tag}*y_true*.npy"))
    candidates += list(search_root.rglob(f"*{det}*{dset}*{tag}*y_score*.npy"))
    candidates += list(search_root.rglob(f"*{dset}*y_true*.npy"))
    candidates += list(search_root.rglob(f"*{dset}*y_score*.npy"))

    y_true, y_score = None, None
    for p in candidates:
        name = p.name.lower()
        if "y_true" in name and y_true is None:  y_true  = p
        if "y_score" in name and y_score is None: y_score = p
    return y_true, y_score

def _load_preds(y_true_path: Path, y_score_path: Path):
    y_true  = np.load(y_true_path)
    y_score = np.load(y_score_path)
    y_true  = y_true.astype(np.int64).reshape(-1)
    y_score = y_score.reshape(-1,) if y_score.ndim==1 else y_score[:,1]
    return y_true, y_score

# --- PR-Kurve (numpy-only, sklearn-kompatibel) ---
def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
    """
    Liefert precision, recall, thresholds (wie sklearn.metrics.precision_recall_curve)
    und Average Precision (AP, wie sklearn.metrics.average_precision_score).
    """
    # Sort by descending score
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score= y_score[order]

    # True Positive/False Positive cumulative
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    # Precision/Recall
    precision = tp / np.maximum(tp + fp, 1)
    P = max(int((y_true == 1).sum()), 1)
    recall = tp / P

    # prepend start point (recall=0, precision=1)
    precision = np.r_[1.0, precision]
    recall    = np.r_[0.0, recall]
    thresholds= np.r_[np.inf, y_score]

    # AP nach sklearn: Präzision monoton machen und Flächeninhalt summieren
    # (Integration der präzisionshülle über recall)
    prec_mon = precision.copy()
    for i in range(len(prec_mon)-2, -1, -1):
        prec_mon[i] = max(prec_mon[i], prec_mon[i+1])
    # Summation über recall-steps
    ap = float(np.sum((recall[1:] - recall[:-1]) * prec_mon[1:]))

    return precision, recall, thresholds, ap

def _style_maps(detectors):
    cmap = plt.get_cmap("tab10")
    det2color = {det: cmap(i % 10) for i, det in enumerate(detectors)}
    tag2style = lambda tag: "-" if tag.lower()=="baseline" else "--"
    return det2color, tag2style

# ---------------------------
# Plotten
# ---------------------------
def plot_dataset(df: pd.DataFrame, dataset: str, outdir: Path, search_root: Path):
    sub = df[df["dataset"].astype(str)==dataset]
    if sub.empty: return

    detectors = list(dict.fromkeys(sub["detector"].tolist()))
    det2color, tag2style = _style_maps(detectors)

    plt.figure(figsize=(7.2, 5.2), dpi=300)
    ax = plt.gca()
    ax.set_title(f"Precision-Recall - {dataset}", fontsize=12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    # No-skill-Linie = Positivklasse-Rate (falls bestimmbar)
    pos_rate_drawn = False

    csv_rows = []

    for _, row in sub.sort_values(["detector","tag","__file__"]).iterrows():
        det = row["detector"]; tag = row["tag"]
        ytp = row.get("y_true_path"); ysp = row.get("y_score_path")

        y_true_p = Path(ytp) if ytp else None
        y_score_p= Path(ysp) if ysp else None
        if not (y_true_p and y_true_p.exists() and y_score_p and y_score_p.exists()):
            y_true_p, y_score_p = _guess_pred_files(search_root, det, dataset, tag)
            if (not y_true_p) or (not y_score_p):
                print(f"[WARN] Keine Predictions für {det} / {dataset} / {tag} gefunden -> Kurve wird übersprungen.")
                continue

        y_true, y_score = _load_preds(y_true_p, y_score_p)

        # No-skill line (nur einmal zeichnen)
        if not pos_rate_drawn:
            pos_rate = (y_true == 1).mean() if y_true.size else 0.0
            ax.hlines(pos_rate, xmin=0, xmax=1, colors="#666666", linestyles=":", linewidth=1.0, label="No-skill")
            pos_rate_drawn = True

        precision, recall, thr, ap = _precision_recall_curve(y_true, y_score)

        ax.plot(recall, precision, tag2style(tag), color=det2color[det], linewidth=2.0,
                label=f"{det}  AP={ap:.4f}%")

        # CSV sammeln
        for r, p in zip(recall, precision):
            csv_rows.append({
                "detector": det, "tag": tag, "dataset": dataset,
                "recall": float(r), "precision": float(p)
            })

    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"pr_{dataset}.png"
    pdf = outdir / f"pr_{dataset}.pdf"
    plt.tight_layout()
    plt.savefig(png, bbox_inches="tight", pad_inches=0.15)
    plt.savefig(pdf, bbox_inches="tight", pad_inches=0.15)
    plt.close()

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(outdir / f"pr_{dataset}__curvepoints.csv", index=False)
    print(f"[OK] PR gespeichert: {png} / {pdf}")

# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="analysis_outputs/metrics", help="Ordner mit JSON-Metriken (rekursiv).")
    ap.add_argument("--outdir", default="analysis_outputs/plots/pr", help="Ausgabeordner für Plots/CSVs.")
    args = ap.parse_args()

    df = load_all_json(Path(args.results_dir))
    outdir = Path(args.outdir)
    search_root = Path(args.results_dir).resolve()

    for dset in df["dataset"].cat.categories:
        plot_dataset(df, str(dset), outdir, search_root=search_root)

if __name__ == "__main__":
    main()
