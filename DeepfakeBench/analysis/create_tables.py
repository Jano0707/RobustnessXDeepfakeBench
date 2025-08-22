#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erzeugt Ergebnis-Tabellen (Experiment 1 & 2) für DeepfakeBench:
- CSV + Markdown + PNG/PDF (immer)
- Multi-Header: pro Dataset die Spalten ACC/AP/AUC/EER; rechts "Avg." (AUC-Mittel)
- EER: kleinster Wert wird hervorgehoben, andere Metriken: größter Wert

Eingabe: JSON-Dateien mit Metriken je (detector, dataset, tag)
Dateinamen-Konvention (optional): <detector>__<dataset>__<tag>.json
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------
# Konstanten / Hilfen
# ---------------------------
# METRICS = ["acc", "ap", "auc", "eer"]   # Spaltenreihenfolge je Dataset
METRICS = ["acc","auc", "eer"]   # Spaltenreihenfolge je Dataset

def _lower_keys(d): return {str(k).lower(): v for k, v in d.items()}
def _pretty_detector(name: str) -> str:
    m = {"effort": "Effort", "xception": "Xception"}
    s = (name or "").strip()
    return m.get(s.lower(), s[:1].upper() + s[1:])

def _infer_from_name(fp: Path, idx: int, default=None):
    parts = fp.stem.split("__")
    return parts[idx] if 0 <= idx < len(parts) else default

def _fmt(x: float, places=4): return f"{x:.{places}f}" if pd.notna(x) else ""

# ---------------------------
# Laden & Normalisieren
# ---------------------------
def load_metric_file(fp: Path) -> Dict:
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    lo = _lower_keys(obj)
    det = (lo.get("detector") or lo.get("model") or _infer_from_name(fp, 0) or "unknown").strip()
    dset = (lo.get("dataset")  or _infer_from_name(fp, 1) or "unknown").strip()
    tag  = (lo.get("tag")      or _infer_from_name(fp, 2, "baseline")).strip()

    metrics = _lower_keys(lo.get("metrics", {}))
    if not metrics:
        metrics = {k: lo.get(k) for k in ("auc","acc","eer","ap") if lo.get(k) is not None}

    def _to_rate(v):
        if isinstance(v, str):
            v = v.strip().replace("%","")
            try: v = float(v)
            except: v = None
        if v is not None and v > 1.0: v = v / 100.0
        return v

    vals = {k: _to_rate(metrics.get(k)) for k in ("acc","ap","auc","eer")}

    # Tag-Aliase
    alias = {"bw":"grayscale","gray":"grayscale","grey":"grayscale",
             "jpeg":"jpeg_compression","jpg":"jpeg_compression",
             "text":"text_overlay","facesmooth":"face_smoothing",
             "facesmoothing":"face_smoothing"}
    tag = alias.get(tag.lower(), tag)

    return {
        "detector": _pretty_detector(det),
        "dataset": dset,
        "tag": tag,
        "acc": vals["acc"], "ap": vals["ap"], "auc": vals["auc"], "eer": vals["eer"],
        "__file__": str(fp),
    }

def load_all(results_dir: Path) -> pd.DataFrame:
    files = list(Path(results_dir).rglob("*.json"))
    if not files: raise SystemExit("Keine JSON-Dateien gefunden.")
    rows = []
    for fp in files:
        try: rows.append(load_metric_file(fp))
        except Exception as e: print(f"[WARN] {fp}: {e}")
    if not rows: raise SystemExit("Keine gültigen JSON-Dateien.")
    df = pd.DataFrame(rows)

    # Reihenfolge der Datasets so, wie sie erscheinen
    order = list(dict.fromkeys(df["dataset"].tolist()))
    df["dataset"] = pd.Categorical(df["dataset"], categories=order, ordered=True)
    return df

# ---------------------------
# Tabellenbau
# ---------------------------
def _avg_auc_across_datasets(row_blocks: Dict[str, Dict[str, float]]) -> float:
    aucs = [row_blocks[ds].get("auc") for ds in row_blocks if row_blocks[ds].get("auc") is not None]
    return float(np.mean(aucs)) if aucs else np.nan

def _collect_row_values(datasets: List[str], row_blocks: Dict[str, Dict[str, float]], as_percent=True):
    vals = []
    for ds in datasets:
        m = row_blocks.get(ds, {})
        for met in METRICS:
            v = m.get(met)
            vals.append(_fmt(v) if as_percent else v)
    avg = _avg_auc_across_datasets(row_blocks)
    vals.append(_fmt(avg) if as_percent else avg)
    return vals

def _build_markdown(idx_header: str, datasets: List[str], rows: List[List[str]]) -> str:
    h1 = [idx_header] + sum(([f"**{ds}**","","",""] for ds in datasets), []) + ["**Avg.**"]
    h2 = [""] + sum(([ "ACC","AP","AUC","EER"] for _ in datasets), []) + ["Avg."]
    align = [":--"] + ["---:" for _ in h2[1:]]
    lines = [
        "| " + " | ".join(h1) + " |",
        "| " + " | ".join(align) + " |",
        "| " + " | ".join(h2) + " |",
        "| " + " | ".join(align) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)

def _build_csv_dataframe(index_cols: List[str], datasets: List[str],
                         data_rows: List[List[str]], index_vals: List[List[str]]) -> pd.DataFrame:
    cols = [(ds, m) for ds in datasets for m in METRICS] + [("Avg.","")]
    df = pd.DataFrame(data_rows, columns=pd.MultiIndex.from_tuples(cols, names=["dataset","metric"]))
    idx = pd.MultiIndex.from_tuples([tuple(v) for v in index_vals], names=index_cols) \
            if len(index_cols)>1 else pd.Index([v[0] for v in index_vals], name=index_cols[0])
    df.index = idx
    return df

def table_experiment1(df: pd.DataFrame):
    base = df[df["tag"]=="baseline"].copy()
    if base.empty: raise ValueError("Experiment 1: keine 'baseline'-Einträge.")
    detectors = list(dict.fromkeys(base["detector"].tolist()))
    datasets  = list(dict.fromkeys(base["dataset"].astype(str).tolist()))

    rows_md, rows_csv, idx_vals = [], [], []
    for det in detectors:
        row_blocks = {}
        for ds in datasets:
            sub = base[(base["detector"]==det) & (base["dataset"]==ds)].sort_values("__file__")
            row_blocks[ds] = {m: sub.iloc[-1][m] for m in METRICS} if not sub.empty else {}
        vals = _collect_row_values(datasets, row_blocks, as_percent=True)
        rows_md.append([det] + vals); rows_csv.append(vals); idx_vals.append([det])

    md = _build_markdown("Detector", datasets, rows_md)
    df_csv = _build_csv_dataframe(["Detector"], datasets, rows_csv, idx_vals)
    return df_csv, md

def table_experiment2(df: pd.DataFrame):
    tags = [t for t in df["tag"].unique().tolist() if t != "baseline"]
    if not tags: raise ValueError("Experiment 2: keine Manipulationen gefunden.")
    detectors = list(dict.fromkeys(df["detector"].tolist()))
    datasets  = list(dict.fromkeys(df["dataset"].astype(str).tolist()))

    rows_md_abs, rows_csv_abs, idx_abs = [], [], []
    rows_md_del, rows_csv_del, idx_del = [], [], []

    for det in detectors:
        for tag in ["baseline"] + tags:
            row_abs, row_del = {}, {}
            for ds in datasets:
                sub = df[(df["detector"]==det) & (df["dataset"]==ds)]
                base = sub[sub["tag"]=="baseline"].sort_values("__file__")
                cur  = sub[sub["tag"]==tag].sort_values("__file__")

                if cur.empty:
                    row_abs[ds] = {}
                    row_del[ds] = {m: None for m in METRICS}
                else:
                    r = cur.iloc[-1]
                    row_abs[ds] = {m: r[m] for m in METRICS}
                    if base.empty or tag=="baseline":
                        row_del[ds] = {m: None for m in METRICS}
                    else:
                        rb = base.iloc[-1]
                        row_del[ds] = {m: (r[m]-rb[m] if pd.notna(r[m]) and pd.notna(rb[m]) else None) for m in METRICS}

            vals_abs = _collect_row_values(datasets, row_abs, as_percent=True)
            rows_md_abs.append([det, tag] + vals_abs); rows_csv_abs.append(vals_abs); idx_abs.append([det, tag])

            vals_del = []
            for ds in datasets:
                for m in METRICS:
                    dv = row_del[ds].get(m)
                    vals_del.append(f"{dv:+.4f}" if dv is not None else "")
            avg_del_auc = _avg_auc_across_datasets(row_del)
            vals_del.append(f"{avg_del_auc:+.4f}" if avg_del_auc is not None and not np.isnan(avg_del_auc) else "")
            rows_md_del.append([det, tag] + vals_del); rows_csv_del.append(vals_del); idx_del.append([det, tag])

    md_abs   = _build_markdown("Detector | Tag", datasets, rows_md_abs)
    md_delta = _build_markdown("Detector | Tag", datasets, rows_md_del)
    df_abs   = _build_csv_dataframe(["Detector","Tag"], datasets, rows_csv_abs, idx_abs)
    df_delta = _build_csv_dataframe(["Detector","Tag"], datasets, rows_csv_del, idx_del)
    return df_abs, md_abs, df_delta, md_delta

# ---------------------------
# Ausgabe & Plotten
# ---------------------------
def save_csv_and_md(df_csv: pd.DataFrame, md_text: str, out_csv: Path, out_md: Path, title: str = None):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_csv.to_csv(out_csv)
    with open(out_md, "w", encoding="utf-8") as f:
        if title: f.write(f"# {title}\n\n")
        f.write(md_text + "\n")

def _cells_to_float(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", "."), errors="coerce")
    return out

def _plot_table(df_csv: pd.DataFrame, title: str, out_base: Path,
                datasets_order: Optional[List[str]] = None,
                idx_name: str = "Detector"):

    # Spalten ordnen
    top_cols = [c for c in df_csv.columns.levels[0] if c != "Avg."]
    if datasets_order:
        top_cols = [d for d in datasets_order if d in top_cols]
    else:
        top_cols = sorted(top_cols)
    mets = METRICS
    cols = [(ds, m) for ds in top_cols for m in mets] + [("Avg.","")]

    df_show = df_csv.copy()
    df_show = df_show.reindex(columns=pd.MultiIndex.from_tuples(cols, names=df_csv.columns.names))
    df_num  = _cells_to_float(df_show)

    # --- unified style (darker grey + thicker lines) ---
    EDGE = "#d0d7de"   # was '#d0d7de' (too light)

    # Layout Maße
    n_rows = df_show.shape[0]
    n_ds   = len(top_cols)
    n_m    = len(mets)
    n_cols = n_ds * n_m + 1  # + Avg.
    stub_w = 2.6
    cw     = 1.15
    rh     = 0.55
    h_head = 0.75
    h_sub  = 0.55
    pad    = 0.2  # a bit more padding to avoid cropping

    W = stub_w + n_cols*cw + pad*2
    H = h_head + h_sub + n_rows*rh + pad*2 + 0.5

    fig, ax = plt.subplots(figsize=(W, H), dpi=220)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    y = H - pad

    # Titel
    ax.text(W/2, y, title, ha="center", va="top", fontsize=13, weight="bold")
    y -= 0.6

    # Header 1: Datensätze
    x = pad
    ax.add_patch(Rectangle((x, y-h_head), stub_w, h_head, facecolor="#f6f8fa",
                           edgecolor=EDGE, joinstyle="miter"))
    ax.text(x+0.15, y-h_head/2, idx_name, ha="left", va="center", fontsize=10, weight="bold")
    x += stub_w
    for ds in top_cols:
        w = n_m * cw
        ax.add_patch(Rectangle((x, y-h_head), w, h_head, facecolor="#f6f8fa",
                               edgecolor=EDGE, joinstyle="miter"))
        ax.text(x+w/2, y-h_head/2, ds, ha="center", va="center", fontsize=10, weight="bold")
        x += w
    # Avg.-Block
    ax.add_patch(Rectangle((x, y-h_head), cw, h_head, facecolor="#f6f8fa",
                           edgecolor=EDGE, joinstyle="miter"))
    ax.text(x+cw/2, y-h_head/2, "", ha="center", va="center", fontsize=10, weight="bold")
    y -= h_head

    # Header 2: ***add missing stub cell under 'Detector'***
    x = pad
    ax.add_patch(Rectangle((x, y-h_sub), stub_w, h_sub, facecolor="white",
                           edgecolor=EDGE, joinstyle="miter"))
    # (left metrics header is intentionally blank)
    x += stub_w
    for _ds in top_cols:
        for m in mets:
            ax.add_patch(Rectangle((x, y-h_sub), cw, h_sub, facecolor="white",
                                   edgecolor=EDGE, joinstyle="miter"))
            ax.text(x+cw/2, y-h_sub/2, m.upper(), ha="center", va="center", fontsize=9)
            x += cw
    ax.add_patch(Rectangle((x, y-h_sub), cw, h_sub, facecolor="white",
                           edgecolor=EDGE, joinstyle="miter"))
    ax.text(x+cw/2, y-h_sub/2, "Avg.", ha="center", va="center", fontsize=9)
    y -= h_sub

    # Bestwert je Spalte: EER = min, sonst max
    best = {}
    for c in df_num.columns:
        ds, met = c
        if met == "eer":
            best[c] = df_num[c].min(skipna=True)
        else:
            best[c] = df_num[c].max(skipna=True)

    # Wertezeilen
    for idx, (row_idx, row) in enumerate(df_show.iterrows()):
        # Stub
        x = pad
        ax.add_patch(Rectangle((x, y-rh), stub_w, rh, facecolor="white",
                               edgecolor=EDGE, joinstyle="miter"))
        label = " | ".join([str(v) for v in (row_idx if isinstance(row_idx, tuple) else (row_idx,))])
        ax.text(x+0.15, y-rh/2, label, ha="left", va="center", fontsize=9)
        x += stub_w

        # Zellen
        for c in df_show.columns:
            ax.add_patch(Rectangle((x, y-rh), cw, rh, facecolor="white",
                                   edgecolor=EDGE, joinstyle="miter"))
            txt = row[c]
            if txt not in ("", None) and not pd.isna(txt):
                try:
                    val = float(str(txt).replace(",", "."))
                except:
                    val = np.nan
                is_best = pd.notna(val) and np.isfinite(best[c]) and np.isclose(val, best[c], atol=1e-9)
                ax.text(x+cw/2, y-rh/2, f"{val:.4f}",
                        ha="center", va="center", fontsize=9,
                        color="black", weight=("bold" if is_best else "normal"))
            x += cw
        y -= rh

    # --- guarantee outer strokes (bottom + left + right) on top ---
    left_x   = pad
    right_x  = pad + stub_w + n_cols*cw
    bottom_y = y  # after last y-=rh above, y == bottom
    top_y    = H - pad

    out_base.parent.mkdir(parents=True, exist_ok=True)
    # tight bbox + pad ensures no clipping of outer strokes
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------
# CLI
# ---------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Erzeugt Tabellen/Plots aus JSON-Metriken.")
    ap.add_argument("--results_dir", type=str, default="analysis_outputs/metrics", help="Ordner mit *.json (rekursiv).")
    ap.add_argument("--outdir", type=str, default="analysis_outputs/tables", help="Zielordner für Tabellen/Plots.")
    args = ap.parse_args()

    df = load_all(Path(args.results_dir))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # --- Experiment 1 ---
    try:
        df1_csv, md1 = table_experiment1(df)
        save_csv_and_md(df1_csv, md1,
                        outdir/"tables_experiment1.csv",
                        outdir/"tables_experiment1.md",
                        title="Experiment 1 - Baseline (ACC/AP/AUC/EER) + Avg. (AUC)")
        _plot_table(df1_csv, "Experiment 1 - Baseline", outdir/"tables_experiment1", idx_name="Detector")
        print(f"[OK] Experiment 1 Tabellen gespeichert unter: {outdir}")
    except Exception as e:
        print(f"[HINWEIS] Experiment 1 übersprungen: {e}")

    # --- Experiment 2 ---
    try:
        df2_abs, md2_abs, df2_delta, md2_delta = table_experiment2(df)
        save_csv_and_md(df2_abs,   md2_abs,   outdir/"tables_experiment2.csv",        outdir/"tables_experiment2.md",
                        title="Experiment 2 - Baseline + Manipulationen (ACC/AP/AUC/EER) + Avg. (AUC)")
        save_csv_and_md(df2_delta, md2_delta, outdir/"tables_experiment2_delta.csv",  outdir/"tables_experiment2_delta.md",
                        title="Experiment 2 - Delta zu Baseline (Prozentpunkte)")
        _plot_table(df2_abs,   "Experiment 2 - Baseline + Manipulationen (ACC/AP/AUC/EER) + Avg. (AUC)", outdir/"tables_experiment2",       idx_name="Detector | Tag")
        _plot_table(df2_delta, "Experiment 2 - Delta zu Baseline (Prozentpunkte)",  outdir/"tables_experiment2_delta", idx_name="Detector | Tag")
        print(f"[OK] Experiment 1 Tabellen gespeichert unter: {outdir}")
    except Exception as e:
        print(f"[HINWEIS] Experiment 2 übersprungen: {e}")

if __name__ == "__main__":
    main()
