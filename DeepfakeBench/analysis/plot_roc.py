#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROC-Kurven zeichnen (pro Dataset) aus DeepfakeBench-Ausgaben.

Nutzt:
- JSON-Metriken (wie in analysis_outputs/metrics erzeugt)
- zugehörige y_true.npy / y_score.npy (wird automatisch gesucht oder aus JSON gelesen)

Ausgabe:
- <outdir>/roc_<DATASET>.png / .pdf
- <outdir>/roc_<DATASET>__curvepoints.csv (FPR/TPR je Kurve)

Aufrufbeispiel:
python analysis/plot_roc.py \
  --results_dir analysis_outputs/metrics \
  --outdir analysis_outputs/plots/roc
"""
from pathlib import Path
import argparse, json, re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Hilfsfunktionen
# ---------------------------

def _lower(d): return {str(k).lower(): v for k,v in d.items()}

def _pretty_detector(name: str) -> str:
    m = {"effort":"Effort", "xception":"Xception"}
    s = (name or "").strip()
    return m.get(s.lower(), s[:1].upper()+s[1:])

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
    # Häufige Namensschemata (anpassbar)
    candidates = []
    # 1) <det>__<dset>__<tag>_y_true.npy
    candidates += list(search_root.rglob(f"*{det}*{dset}*{tag}*y_true*.npy"))
    candidates += list(search_root.rglob(f"*{det}*{dset}*{tag}*y_score*.npy"))
    # 2) <dset>_y_true.npy (z. B. Exp1 ohne Tag)
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
    # y_score kann (N,) oder (N,2) sein
    y_score = y_score.reshape(-1, ) if y_score.ndim==1 else y_score[:,1]
    return y_true, y_score

def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """ROC: FPR, TPR, Schwellen + AUC (numpy-only)."""
    # Sort by descending score
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score= y_score[order]

    P = (y_true==1).sum()
    N = (y_true==0).sum()
    if P==0 or N==0:
        return np.array([0,1]), np.array([0,1]), np.array([np.inf, -np.inf]), 0.5

    # True Positive/False Positive cumulative
    tps = np.cumsum(y_true==1)
    fps = np.cumsum(y_true==0)

    # FPR/TPR at each unique threshold
    # insert (0,0) and (1,1)
    TPR = np.concatenate(([0.0], tps / P, [1.0]))
    FPR = np.concatenate(([0.0], fps / N, [1.0]))
    thr = np.concatenate(([np.inf], y_score, [-np.inf]))

    # AUC via trapezoid
    auc = np.trapz(TPR, FPR)
    return FPR, TPR, thr, float(auc)

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
    if sub.empty:
        return

    detectors = list(dict.fromkeys(sub["detector"].tolist()))
    det2color, tag2style = _style_maps(detectors)

    plt.figure(figsize=(7.2, 5.2), dpi=300)
    ax = plt.gca()
    ax.set_title(f"ROC - {dataset}", fontsize=12)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    legend_entries = []
    csv_rows = []

    for _, row in sub.sort_values(["detector","tag","__file__"]).iterrows():
        det = row["detector"]; tag = row["tag"]; jf = Path(row["__file__"])
        ytp, ysp = row.get("y_true_path"), row.get("y_score_path")

        y_true_p = Path(ytp) if ytp else None
        y_score_p= Path(ysp) if ysp else None
        if not (y_true_p and y_true_p.exists() and y_score_p and y_score_p.exists()):
            # versuche zu raten: suche relativ zum results_dir (und ggf. dessen parent)
            y_true_p, y_score_p = _guess_pred_files(search_root, det, dataset, tag)
            if (not y_true_p) or (not y_score_p):
                print(f"[WARN] Keine Predictions für {det} / {dataset} / {tag} gefunden -> Kurve wird übersprungen.")
                continue

        y_true, y_score = _load_preds(y_true_p, y_score_p)
        FPR, TPR, thr, auc = _roc_curve(y_true, y_score)

        ax.plot(FPR, TPR, tag2style(tag), color=det2color[det], linewidth=2.0,
                label=f"{det}  AUC={auc:.4f}")

        # CSV sammeln
        for f, t in zip(FPR, TPR):
            csv_rows.append({
                "detector": det, "tag": tag, "dataset": dataset,
                "fpr": float(f), "tpr": float(t)
            })

    ax.plot([0,1],[0,1], ":", color="#666666", linewidth=1.0)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"roc_{dataset}.png"
    pdf = outdir / f"roc_{dataset}.pdf"
    plt.tight_layout()
    plt.savefig(png, bbox_inches="tight", pad_inches=0.15)
    plt.savefig(pdf, bbox_inches="tight", pad_inches=0.15)
    plt.close()

    # CSV mit Punkten
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(outdir / f"roc_{dataset}__curvepoints.csv", index=False)
    print(f"[OK] ROC gespeichert: {png} / {pdf}")

# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="analysis_outputs/metrics", help="Ordner mit JSON-Metriken (rekursiv).")
    ap.add_argument("--outdir", default="analysis_outputs/plots/roc", help="Ausgabeordner für Plots/CSVs.")
    args = ap.parse_args()

    df = load_all_json(Path(args.results_dir))
    outdir = Path(args.outdir)
    search_root = Path(args.results_dir).resolve()
    # Zusätzlich auch im übergeordneten Ordner suchen (häufig liegen npy neben metrics-Ordnern)
    parent_root = search_root.parent

    for dset in df["dataset"].cat.categories:
        # Suche zunächst unter results_dir, falls nichts gefunden wird, wird _guess_pred_files
        # automatisch auch breiter fündig (patterns sind rekursiv).
        plot_dataset(df, str(dset), outdir, search_root=search_root)
        # Falls du möchtest zusätzlich noch den Parent probieren, kannst du die folgende Zeile
        # aktivieren und vorheriges Ergebnis überschreiben, wenn mehr Kurven gefunden wurden.
        # plot_dataset(df, str(dset), outdir, search_root=parent_root)

if __name__ == "__main__":
    main()
