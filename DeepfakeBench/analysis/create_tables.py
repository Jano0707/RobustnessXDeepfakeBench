import json
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------
# Konstanten / Hilfen
# ---------------------------
# METRICS = ["acc", "ap", "auc", "eer"]   # Spaltenreihenfolge je Dataset
METRICS = ["acc","auc"]   # Spaltenreihenfolge je Dataset

# erlaubte Varianten pro Experiment
GEN_TAGS = ["Baseline", "Within-Domain", "Cross-Domain"]
ROB_TAGS = ["Baseline", "Schwarz-Weiß", "JPEG", "Face-Smoothing", "Text-Overlay", "Text-Overlay-Augen"]

HILITE_BG = "#fff7cc"  # Hintergrundfarbe für Overall-Best

def _lower_keys(d): return {str(k).lower(): v for k, v in d.items()}
def _pretty_detector(name: str) -> str:
    m = {"effort": "Effort", "xception": "Xception"}
    s = (name or "").strip()
    return m.get(s.lower(), s[:1].upper() + s[1:])

def _infer_from_name(fp: Path, idx: int, default=None):
    parts = fp.stem.split("__")
    return parts[idx] if 0 <= idx < len(parts) else default

def _fmt(x: float, places=4): return f"{x:.{places}f}" if pd.notna(x) else ""

def _norm_variant(v: str) -> str:
    if not v: return ""
    a = {
        "baseline":"Baseline",
        "schwarz-weiss":"Schwarz-Weiß",
        "jpeg":"JPEG",
        "face-smoothing":"Face-Smoothing",
        "within-domain":"Within-Domain",
        "cross-domain":"Cross-Domain",
        "text-overlay":"Text-Overlay",
        "text-overlay-augen":"Text-Overlay-Augen"
    }
    v = v.strip().lower()
    return a.get(v, v)

def _parse_tag_like(raw_tag: str):
    """
    Zerlegt tag in (exp, tag). Robust gegen zu kurze Tags.
    """
    raw = (raw_tag or "").strip()
    parts = [p.strip().lower() for p in raw.split("-") if p.strip()]
    
    if not parts:
        return "", _norm_variant(raw), raw

    exp = parts[0]
    if len(parts) >= 3:
        variant = "-".join(parts[2:])
    elif len(parts) == 2:
        variant = parts[1]
    else:
        variant = raw

    return exp, _norm_variant(variant), raw

def _display_dataset_name(dset: str) -> str:
    """
    Anzeige-Name des Datasets.
    Erkannt werden folgende (case-insensitive) Endungen – längste zuerst:
      -TEXT-Augen  -> -Text-Overlay-Augen
      -TEXT        -> -Text-Overlay
      -FACE        -> -Gesichtsglättung
      -JPEG        -> -JPEG
      -S_W         -> -Schwarz-Weiß
    Basisnamen (ohne Suffix) bleiben unverändert.
    """
    if not dset:
        return dset

    # (key -> display-suffix)
    suffix_map = {
        "-text-augen": " (Text-Overlay-Augen)",
        "-text":       " (Text-Overlay)",
        "-face":       " (Gesichtsglättung)",
        "-jpeg":       " (JPEG)",
        "-s_w":        " (Schwarz-Weiß)",
    }
    s_lower = dset.lower()

    # längste Schlüssel zuerst prüfen
    for key in sorted(suffix_map.keys(), key=len, reverse=True):
        if s_lower.endswith(key):
            base = dset[: -len(key)]
            return base + suffix_map[key]

    # kein bekanntes Suffix -> unverändert
    return dset
    
# Priorität in E1 (Gen)
_GEN_TAG_ORDER = {"Baseline": 0, "Within-Domain": 1, "Cross-Domain": 2}
# Priorität in E2 (Rob) – vorausgesetzte Reihenfolge nach Baseline:
# Schwarz-Weiß, JPEG, Gesichtsglättung, Text-Overlay, Text-Overlay-Augen
_ROB_GROUP_ORDER = {
    "Baseline": 0,
    "Schwarz-Weiß": 1,
    "JPEG": 2,
    "Gesichtsglättung": 3,
    "Text-Overlay": 4,
    "Text-Overlay-Augen": 5,
}

def _rob_base_and_group_from_raw(dset: str) -> Tuple[str, str]:
    """
    Zerlegt Roh-Datasetnamen in (base, group) für Experiment 2.
    group ∈ {"Baseline","Schwarz-Weiß","JPEG","Gesichtsglättung","Text-Overlay","Text-Overlay-Augen"}.
    """
    if not dset:
        return dset, "Baseline"

    mapping = [
        ("-text-augen", "Text-Overlay-Augen"),
        ("-text",       "Text-Overlay"),
        ("-face",       "Gesichtsglättung"),
        ("-jpeg",       "JPEG"),
        ("-s_w",        "Schwarz-Weiß"),
    ]
    s_lower = dset.lower()
    for key, group in sorted(mapping, key=lambda kv: len(kv[0]), reverse=True):
        if s_lower.endswith(key):
            base = dset[: -len(key)]
            return base, group
    return dset, "Baseline"

def _latest_value(df_sub: pd.DataFrame, col: str = "auc") -> Optional[float]:
    """Nimmt die letzte Zeile (nach __file__ sortiert) und gibt col zurück."""
    if df_sub.empty:
        return None
    r = df_sub.sort_values("__file__").iloc[-1]
    return r.get(col, None)

def _rob_groups_ordered() -> List[str]:
    return ["Schwarz-Weiß", "JPEG", "Gesichtsglättung", "Text-Overlay", "Text-Overlay-Augen"]
    
# ---------------------------
# Laden & Normalisieren
# ---------------------------
def load_metric_file(fp: Path) -> Dict:
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    lo = _lower_keys(obj)
    det = (lo.get("detector") or lo.get("model") or _infer_from_name(fp, 0) or "unknown").strip()
    dset = (lo.get("dataset")  or _infer_from_name(fp, 1) or "unknown").strip()
    tag_raw  = (lo.get("tag")      or _infer_from_name(fp, 2, "baseline")).strip()
    
    exp   = (lo.get("exp")   or "").strip().lower()
    if not exp:
        p_exp, _, _ = _parse_tag_like(tag_raw)
        exp = p_exp
    tag_norm = _norm_variant(tag_raw)
    
    # Metrics
    metrics_obj = _lower_keys(lo.get("metrics", {}))
    if not metrics_obj:
        metrics_obj = {k: lo.get(k) for k in ("auc","acc","eer","ap") if lo.get(k) is not None}

    def _to_rate(v):
        if isinstance(v, str):
            v = v.strip().replace("%","")
            try: v = float(v)
            except: v = None
        if v is not None and v > 1.0: v = v / 100.0
        return v

    vals = {k: _to_rate(metrics_obj.get(k)) for k in ("acc","ap","auc","eer")}

    return {
        "detector": _pretty_detector(det),
        "dataset": dset,
        "exp": exp,                 
        "tag": tag_norm,            
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

def _collect_row_values(datasets: List[str],
                        row_blocks: Dict[str, Dict[str, float]],
                        as_percent=True,
                        include_avg: bool = True):
    vals = []
    for ds in datasets:
        m = row_blocks.get(ds, {})
        for met in METRICS:
            v = m.get(met)
            vals.append(_fmt(v) if as_percent else v)
    if include_avg:
        avg = _avg_auc_across_datasets(row_blocks)
        vals.append(_fmt(avg) if as_percent else avg)
    return vals

def _build_markdown(idx_header: str,
                    datasets: List[str],
                    rows: List[List[str]],
                    include_avg: bool = True):
    # Oberzeile: Dataset-Blöcke
    top = [idx_header] + sum(([f"**{ds}**"] + [""]*(len(METRICS)-1) for ds in datasets), [])
    if include_avg: top += ["**Average**"]

    # Metrikzeile passend zu METRICS
    met_labels = [m.upper() for m in METRICS]
    h2 = [""] + sum(([*met_labels] for _ in datasets), [])
    if include_avg: h2 += ["Average"]

    align = [":--"] + ["---:" for _ in h2[1:]]
    lines = [
        "| " + " | ".join(top) + " |",
        "| " + " | ".join(align) + " |",
        "| " + " | ".join(h2) + " |",
        "| " + " | ".join(align) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)

def _build_markdown_by_dataset(idx_header: str,
                               detectors: List[str],
                               rows: List[List[str]]) -> str:
    # Oberzeile: Detektor-Blöcke
    top = [idx_header] + sum(([f"**{det}**"] + [""]*(len(METRICS)-1) for det in detectors), [])
    # Unterzeile: Metriken
    met_labels = [m.upper() for m in METRICS]
    h2 = [""] + sum(([*met_labels] for _ in detectors), [])
    align = [":--"] + ["---:" for _ in h2[1:]]

    lines = [
        "| " + " | ".join(top) + " |",
        "| " + " | ".join(align) + " |",
        "| " + " | ".join(h2) + " |",
        "| " + " | ".join(align) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)
    
def _build_csv_dataframe_by_dataset(index_col: str,
                                    detectors: List[str],
                                    data_rows: List[List[str]],
                                    index_vals: List[str]) -> pd.DataFrame:
    cols = [(det, m) for det in detectors for m in METRICS]
    df = pd.DataFrame(data_rows, columns=pd.MultiIndex.from_tuples(cols, names=["detector","metric"]))
    df.index = pd.Index(index_vals, name=index_col)
    return df

def _make_table_dset_rows_gen(sub: pd.DataFrame,
                              include_avg_row: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Experiment 1 (Gen): Zeilen sind (Dataset [+ Tag-Suffix]), Spalten sind (Detektor, Metrik).
    Sortierung:
      - primär Tag-Gruppen: Baseline -> Within-Domain -> Cross-Domain
      - innerhalb jeder Tag-Gruppe alphabetisch nach Dataset (Displayname)
    """
    if sub.empty:
        raise ValueError("Keine Daten nach Filterung (Experiment 1).")

    detectors = list(dict.fromkeys(sub["detector"].tolist()))

    # alle (dataset, tag) Paare, die existieren -> mit Displaynamen (inkl. Tag-Suffix bei Nicht-Baseline)
    pairs = []
    for (d_raw, t), g in sub.groupby(["dataset", "tag"]):
        if t not in GEN_TAGS:
            continue
        d_disp_base = _display_dataset_name(str(d_raw))
        row_label = d_disp_base if t == "Baseline" else f"{d_disp_base} ({t})"
        pairs.append((d_raw, t, d_disp_base, row_label))

    # Sortieren nach Tag-Priorität, dann alphabetisch nach Basis-Display
    pairs.sort(key=lambda x: (_GEN_TAG_ORDER.get(x[1], 99), x[2].lower()))

    rows_csv: List[List[str]] = []
    rows_md : List[List[str]] = []
    idx_vals: List[str] = []

    for d_raw, t, d_disp_base, row_label in pairs:
        row_vals: List[str] = []
        for det in detectors:
            cur = sub[(sub["dataset"]==d_raw) & (sub["tag"]==t) & (sub["detector"]==det)].sort_values("__file__")
            if cur.empty:
                row_vals += ["" for _ in METRICS]
            else:
                r = cur.iloc[-1]
                for m in METRICS:
                    row_vals.append(_fmt(r[m]))
        rows_csv.append(row_vals)
        rows_md.append([row_label] + row_vals)
        idx_vals.append(row_label)

    # Avg.-Zeile (Durchschnitt je (Detektor, Metrik))
    if include_avg_row and rows_csv:
        tmp = _build_csv_dataframe_by_dataset("Datensatz", detectors, rows_csv, idx_vals)
        tmp_num = _cells_to_float(tmp)
        avg_vals: List[str] = []
        for det in detectors:
            for m in METRICS:
                col = tmp_num[(det, m)]
                mean_val = float(np.nanmean(col)) if not col.isna().all() else np.nan
                avg_vals.append(_fmt(mean_val))
        rows_csv.append(avg_vals)
        rows_md.append(["Average"] + avg_vals)
        idx_vals.append("Average")

    df_csv = _build_csv_dataframe_by_dataset("Datensatz", detectors, rows_csv, idx_vals)
    md_text = _build_markdown_by_dataset("Datensatz", detectors, rows_md)
    return df_csv, md_text


def _make_table_dset_rows(sub: pd.DataFrame,
                          include_avg_row: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Experiment 2: Zeilenanzeige
      - Baseline: <Base>
      - Varianten: <Base> (<Gruppe>)
    Sortierung:
      1) alphabetisch nach <Base>
      2) innerhalb der Basis nach _ROB_GROUP_ORDER (Baseline, Schwarz-Weiß, JPEG, Gesichtsglättung, Text-Overlay, Text-Overlay-Augen)
    """
    if sub.empty:
        raise ValueError("Keine Daten nach Filterung vorhanden.")

    detectors = list(dict.fromkeys(sub["detector"].tolist()))

    all_raw = list(dict.fromkeys(sub["dataset"].astype(str).tolist()))
    items = []
    for d_raw in all_raw:
        base, group = _rob_base_and_group_from_raw(d_raw)
        disp = base if group == "Baseline" else f"{base} ({group})"
        prio = _ROB_GROUP_ORDER.get(group, 99)
        items.append((d_raw, base, group, disp, prio))

    # NEU: primär nach Base, sekundär nach Gruppen-Priorität
    items.sort(key=lambda x: (x[1].lower(), x[4]))

    rows_csv, rows_md, idx_vals = [], [], []
    for d_raw, base, group, disp, _prio in items:
        row_vals = []
        for det in detectors:
            cur = sub[(sub["dataset"] == d_raw) & (sub["detector"] == det)].sort_values("__file__")
            if cur.empty:
                row_vals += ["" for _ in METRICS]
            else:
                r = cur.iloc[-1]
                for m in METRICS:
                    row_vals.append(_fmt(r[m]))
        rows_csv.append(row_vals)
        rows_md.append([disp] + row_vals)
        idx_vals.append(disp)

    if include_avg_row and rows_csv:
        tmp = _build_csv_dataframe_by_dataset("Datensatz", detectors, rows_csv, idx_vals)
        tmp_num = _cells_to_float(tmp)
        avg_vals = []
        for det in detectors:
            for m in METRICS:
                col = tmp_num[(det, m)]
                mean_val = float(np.nanmean(col)) if not col.isna().all() else np.nan
                avg_vals.append(_fmt(mean_val))
        rows_csv.append(avg_vals)
        rows_md.append(["Average"] + avg_vals)
        idx_vals.append("Average")

    df_csv = _build_csv_dataframe_by_dataset("Datensatz", detectors, rows_csv, idx_vals)
    md_text = _build_markdown_by_dataset("Datensatz", detectors, rows_md)
    return df_csv, md_text

def _build_csv_dataframe(index_cols: List[str],
                         datasets: List[str],
                         data_rows: List[List[str]],
                         index_vals: List[List[str]],
                         include_avg: bool = True) -> pd.DataFrame:
    cols = [(ds, m) for ds in datasets for m in METRICS]
    if include_avg:
        cols += [("Average","")]
    df = pd.DataFrame(data_rows, columns=pd.MultiIndex.from_tuples(cols, names=["dataset","metric"]))
    idx = pd.MultiIndex.from_tuples([tuple(v) for v in index_vals], names=index_cols) \
            if len(index_cols)>1 else pd.Index([v[0] for v in index_vals], name=index_cols[0])
    df.index = idx
    return df

# ---------------------------
# Experiment 1: Generalisierung
# ---------------------------

def table_experiment1(df: pd.DataFrame, include_avg: bool = True):
    sub = df[(df["exp"].str.lower()=="gen") & (df["tag"].isin(GEN_TAGS))].copy()
    if sub.empty:
        raise ValueError("Experiment 1: keine Einträge mit exp='gen' gefunden.")
    df_csv, md = _make_table_dset_rows_gen(sub, include_avg_row=include_avg)
    return df_csv, md

def table_experiment1_baseline(df: pd.DataFrame):
    sub = df[(df["exp"].str.lower()=="gen") & (df["tag"]=="Baseline")].copy()
    if sub.empty: raise ValueError("Experiment 1 (Baseline): keine Einträge gefunden.")
    df_csv, md = _make_table_dset_rows(sub, include_avg_row=False)
    return df_csv, md

def table_experiment1_within_domain(df: pd.DataFrame):
    allowed = ["Baseline", "Within-Domain"]
    sub = df[(df["exp"].str.lower()=="gen") & (df["tag"].isin(allowed))].copy()
    if sub.empty:
        raise ValueError("Experiment 1 (Baseline+Within-Domain): keine Einträge gefunden.")
    df_csv, md = _make_table_dset_rows_gen(sub, include_avg_row=False)
    return df_csv, md


# ---------------------------
# Experiment 2: Robustheit
# ---------------------------

def table_experiment2(df: pd.DataFrame):
    sub = df[(df["exp"].str.lower()=="rob") & (df["tag"].isin(ROB_TAGS))].copy()
    if sub.empty: raise ValueError("Experiment 2: keine Einträge mit exp='rob' gefunden.")
    df_abs, md_abs = _make_table_dset_rows(sub, include_avg_row=True)

    return df_abs, md_abs, None, None

def table_experiment2_subset(df: pd.DataFrame, extra_tags: List[str], include_avg: bool = True):
    allowed = ["Baseline"] + list(dict.fromkeys(extra_tags))
    sub = df[(df["exp"].str.lower() == "rob") & (df["tag"].isin(allowed))].copy()
    if sub.empty:
        raise ValueError(f"Experiment 2 (Subset: {allowed}): keine Einträge gefunden.")

    # Map Tag -> Display-Gruppenname
    display_name_map = {"Face-Smoothing": "Gesichtsglättung"}
    target_groups = ["Baseline"] + [display_name_map.get(t, t) for t in extra_tags]
    order_map = {g: i for i, g in enumerate(target_groups)}

    detectors = list(dict.fromkeys(sub["detector"].tolist()))
    all_raw = list(dict.fromkeys(sub["dataset"].astype(str).tolist()))

    items = []
    for d_raw in all_raw:
        base, group = _rob_base_and_group_from_raw(d_raw)
        if group not in order_map:  # nur Gruppen, die in diesem Subset vorkommen sollen
            continue
        disp = base if group == "Baseline" else f"{base} ({group})"
        items.append((d_raw, base, group, disp, order_map[group]))

    # NEU: primär Base, sekundär gewünschte Gruppenreihenfolge
    items.sort(key=lambda x: (x[1].lower(), x[4]))

    rows_csv, rows_md, idx_vals = [], [], []
    for d_raw, base, group, disp, _prio in items:
        row_vals = []
        for det in detectors:
            cur = sub[(sub["dataset"] == d_raw) & (sub["detector"] == det)].sort_values("__file__")
            if cur.empty:
                row_vals += ["" for _ in METRICS]
            else:
                r = cur.iloc[-1]
                for m in METRICS:
                    row_vals.append(_fmt(r[m]))
        rows_csv.append(row_vals)
        rows_md.append([disp] + row_vals)
        idx_vals.append(disp)

    if include_avg and rows_csv:
        tmp = _build_csv_dataframe_by_dataset("Datensatz", detectors, rows_csv, idx_vals)
        tmp_num = _cells_to_float(tmp)
        avg_vals = []
        for det in detectors:
            for m in METRICS:
                col = tmp_num[(det, m)]
                mean_val = float(np.nanmean(col)) if not col.isna().all() else np.nan
                avg_vals.append(_fmt(mean_val))
        rows_csv.append(avg_vals)
        rows_md.append(["Average"] + avg_vals)
        idx_vals.append("Average")

    df_abs = _build_csv_dataframe_by_dataset("Datensatz", detectors, rows_csv, idx_vals)
    md_abs = _build_markdown_by_dataset("Datensatz", detectors, rows_md)
    return df_abs, md_abs



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
                detectors_order: Optional[List[str]] = None,
                idx_name: str = "Datensatz"):
    # Erwartet: columns MultiIndex ("detector","metric"), index="Datensatz"
    if not isinstance(df_csv.columns, pd.MultiIndex) or df_csv.columns.names != ["detector","metric"]:
        cols = pd.MultiIndex.from_tuples(df_csv.columns, names=["detector","metric"])
        df_csv = df_csv.copy()
        df_csv.columns = cols

    dets = list(df_csv.columns.levels[0])
    if detectors_order:
        dets = [d for d in detectors_order if d in dets]

    mets = METRICS
    cols = [(det, m) for det in dets for m in mets]
    df_show = df_csv.copy().reindex(columns=pd.MultiIndex.from_tuples(cols, names=df_csv.columns.names))
    df_num  = _cells_to_float(df_show)

    EDGE = "#d0d7de"
    n_rows = df_show.shape[0]
    n_det  = len(dets)
    n_m    = len(mets)

    long_labels = {"DeepFakeDetection", "DeepFakeDetection (Schwarz-Weiß)", "DeepFakeDetection (Cross-Domain)", "FaceForensics++ (Within-Domain)", "DeepFakeDetection (Text-Overlay)"}
    xlong_labels = {"DeepFakeDetection (Gesichtsglättung)", "DeepFakeDetection (Text-Overlay-Augen)"}
    has_long = any(str(idx) in long_labels for idx in df_show.index.tolist())
    has_xlong = any(str(idx) in xlong_labels for idx in df_show.index.tolist())

    stub_w = 3.1 if has_xlong else (2.8 if has_long else 2.2)
    cw     = 1.15
    rh     = 0.55
    h_head = 0.75
    h_sub  = 0.55
    pad    = 0.0

    W = stub_w + len(dets)*len(mets)*cw + pad*2
    H = h_head + h_sub + n_rows*rh + pad*2  

    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    fig.subplots_adjust(0, 0, 1, 1)           
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    y = H - pad

    # Header 1: Detektoren (ohne Titelzeile)
    x = pad
    ax.add_patch(Rectangle((x, y-h_head), stub_w, h_head, facecolor="#f6f8fa",
                           edgecolor=EDGE, joinstyle="miter"))
    ax.text(x+0.15, y-h_head/2, idx_name, ha="left", va="center", fontsize=10, weight="bold")
    x += stub_w
    for det in dets:
        w = len(mets) * cw
        ax.add_patch(Rectangle((x, y-h_head), w, h_head, facecolor="#f6f8fa",
                               edgecolor=EDGE, joinstyle="miter"))
        ax.text(x+w/2, y-h_head/2, det, ha="center", va="center", fontsize=10, weight="bold")
        x += w
    y -= h_head

    # Header 2: Metriken
    x = pad
    ax.add_patch(Rectangle((x, y-h_sub), stub_w, h_sub, facecolor="white", edgecolor=EDGE))
    x += stub_w
    for _ in dets:
        for m in mets:
            ax.add_patch(Rectangle((x, y-h_sub), cw, h_sub, facecolor="white", edgecolor=EDGE))
            ax.text(x+cw/2, y-h_sub/2, m.upper(), ha="center", va="center", fontsize=10)
            x += cw
    y -= h_sub

    # Zellen
    for row_idx, row in df_show.iterrows():
        x = pad
        ax.add_patch(Rectangle((x, y-rh), stub_w, rh, facecolor="white", edgecolor=EDGE))
        ax.text(x+0.15, y-rh/2, str(row_idx), ha="left", va="center", fontsize=10)
        x += stub_w

        highlight_this_row = (str(row_idx).strip().lower() != "average")
        best_for_metric = {}
        if highlight_this_row:
            for m in mets:
                values = [pd.to_numeric(str(row[(det, m)]).replace(",", "."), errors="coerce") for det in dets]
                arr = np.array(values, dtype=float)
                best_for_metric[m] = np.nanmin(arr) if m == "eer" else np.nanmax(arr) if not np.all(np.isnan(arr)) else np.nan

        for det in dets:
            for m in mets:
                txt = row[(det, m)]
                val = pd.to_numeric(str(txt).replace(",", "."), errors="coerce")
                is_best = False
                if highlight_this_row:
                    b = best_for_metric.get(m, np.nan)
                    is_best = (pd.notna(val) and np.isfinite(val) and np.isfinite(b) and np.isclose(val, b, atol=1e-12))
                face = HILITE_BG if is_best else "white"
                ax.add_patch(Rectangle((x, y-rh), cw, rh, facecolor=face, edgecolor=EDGE))
                if pd.notna(val):
                    ax.text(x+cw/2, y-rh/2, f"{val:.4f}", ha="center", va="center", fontsize=9,
                            color="black", weight=("bold" if is_best else "normal"))
                x += cw
        y -= rh

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def plot_robustness_deltas(df: pd.DataFrame, outdir: Path):
    sub = df[(df["exp"].str.lower() == "rob")].copy()
    if sub.empty:
        return

    outdir.mkdir(parents=True, exist_ok=True)

    detectors = list(dict.fromkeys(sub["detector"].tolist()))
    bases = sorted({ _rob_base_and_group_from_raw(str(ds))[0] for ds in sub["dataset"].astype(str) })

    for det in detectors:
        for base in bases:
            # Baseline-Zeile für (det, base)
            base_rows = sub[(sub["detector"] == det) &
                            (sub["dataset"].astype(str) == base)]
            auc_base = _latest_value(base_rows, "auc")
            if auc_base is None or pd.isna(auc_base):
                continue  # ohne Baseline keine sinnvollen Deltas

            groups = _rob_groups_ordered()
            y_labels, deltas = [], []
            for g in groups:
                cand = sub[(sub["detector"] == det)]
                mask = cand["dataset"].astype(str).apply(
                    lambda s: _rob_base_and_group_from_raw(s) == (base, g)
                )
                var_rows = cand[mask]
                auc_var = _latest_value(var_rows, "auc")
                if auc_var is None or pd.isna(auc_var):
                    continue
                delta = float(auc_var) - float(auc_base)
                y_labels.append(g)
                deltas.append(delta)

            if not deltas:
                continue

            # Plot
            fig, ax = plt.subplots(figsize=(4.3, 3.0), dpi=220)
            y_pos = np.arange(len(deltas))
            bars = ax.barh(
                y_pos, deltas,
                color="#E6EBFF",        
                edgecolor="#C8D2FF",    
                linewidth=1.0
            )

            # Null-Linie gestrichelt
            ax.axvline(0.0, linestyle="dashed", linewidth=1.0, color="black", alpha=0.8)

            # Y-Achse: Manipulationsnamen
            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_labels)
            ax.invert_yaxis()

            ax.set_xlabel("Differenz zur Baseline (AUC)")
            ax.set_title(f"Detektor: {det} | Datensatz: {base}", fontsize=10)

            max_abs = max(abs(v) for v in deltas) if deltas else 0.0
            xpad = max(0.002, 0.02 * max_abs)  

            for bar, v in zip(bars, deltas):
                w = bar.get_width()
                y = bar.get_y() + bar.get_height() / 2.0
                if w >= 0:
                    x, ha = w - xpad, "right"
                else:
                    x, ha = w + xpad, "left"
                ax.text(x, y, f"{v:+.3f}", va="center", ha=ha, fontsize=9, color="black")

            fig.tight_layout()


            safe_base = base.replace("/", "_")
            out_path = outdir / f"rob_delta_{det}_{safe_base}"
            fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05)
            fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
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

    # --- Experiment 1 (voll) ---
    try:
        df1_csv, md1 = table_experiment1(df, include_avg=True)
        save_csv_and_md(df1_csv, md1,
                        outdir/"table_experiment1.csv",
                        outdir/"table_experiment1.md",
                        title="Experiment 1 - Generalisierung")
        _plot_table(df1_csv, "Experiment 1 - Generalisierung", outdir/"table_experiment1",
                    idx_name="Datensatz")
        print(f"[OK] Experiment 1 (voll) gespeichert unter: {outdir}")
    except Exception as e:
        print(f"[HINWEIS] Experiment 1 (voll) übersprungen: {e}")

    # --- Experiment 1 (nur Baseline) ---
    try:
        df1b_csv, md1b = table_experiment1_baseline(df)
        save_csv_and_md(df1b_csv, md1b,
                        outdir/"table_experiment1_baseline.csv",
                        outdir/"table_experiment1_baseline.md",
                        title="Experiment 1 - Generalisierung (Baseline)")
        _plot_table(df1b_csv, "Experiment 1 - Generalisierung (Baseline)",
                    outdir/"table_experiment1_baseline",
                    idx_name="Datensatz")
        print(f"[OK] Experiment 1 (Baseline) gespeichert unter: {outdir}")
    except Exception as e:
        print(f"[HINWEIS] Experiment 1 (Baseline) übersprungen: {e}")

    # --- Experiment 1 (Baseline + Within-Domain) ---
    try:
        df1w_csv, md1w = table_experiment1_within_domain(df)
        save_csv_and_md(df1w_csv, md1w,
                        outdir/"table_experiment1_within-domain.csv",
                        outdir/"table_experiment1_within-domain.md",
                        title="Experiment 1 - Generalisierung (Baseline + Within-Domain)")
        _plot_table(df1w_csv, "Experiment 1 - Generalisierung (Baseline + Within-Domain)",
                    outdir/"table_experiment1_within-domain",
                    idx_name="Datensatz")
        print(f"[OK] Experiment 1 (Baseline+Within) gespeichert unter: {outdir}")
    except Exception as e:
        print(f"[HINWEIS] Experiment 1 (Baseline+Within) übersprungen: {e}")

    # --- Experiment 2 ---
    try:
        df2_abs, md2_abs, df2_delta, md2_delta = table_experiment2(df)
        save_csv_and_md(df2_abs, md2_abs,
                        outdir/"table_experiment2.csv",
                        outdir/"table_experiment2.md",
                        title="Experiment 2 - Robustheit")
        _plot_table(df2_abs, "Experiment 2 - Robustheit", outdir/"table_experiment2",
                    idx_name="Datensatz")
        print(f"[OK] Experiment 2 Tabellen gespeichert unter: {outdir}")
    except Exception as e:
        print(f"[HINWEIS] Experiment 2 übersprungen: {e}")
    
    # --- Experiment 2: Baseline + einzelne/kombinierte Tags ---
    try:
        subsets = [
            ["Schwarz-Weiß"],
            ["JPEG"],
            ["Face-Smoothing"],
            ["Text-Overlay"],
            ["Text-Overlay-Augen"],
            ["Text-Overlay", "Text-Overlay-Augen"],  # Kombi
        ]

        for extra in subsets:
            df_sub, md_sub = table_experiment2_subset(df, extra_tags=extra, include_avg=True)
            base = f"table_experiment2_baseline-{'+'.join(extra)}"
            save_csv_and_md(df_sub, md_sub,
                            outdir / f"{base}.csv",
                            outdir / f"{base}.md",
                            title=f"Experiment 2 - Robustheit (Baseline + {' + '.join(extra)})")
            _plot_table(df_sub,
                        f"Experiment 2 - Robustheit (Baseline + {' + '.join(extra)})",
                        outdir / base,
                        idx_name="Datensatz")

        print(f"[OK] Experiment 2 (Baseline+Tag-Teiltabellen) gespeichert unter: {outdir}")
    except Exception as e:
        print(f"[HINWEIS] Experiment 2 (Baseline+Tag-Teiltabellen) übersprungen: {e}")
    
    # --- Experiment 2: Balkendiagramme Delta AUC je Detektor+Datensatz ---
    try:
        plot_robustness_deltas(df, outdir / "plots_robustheit_deltas")
        print(f"[OK] Experiment 2 ΔAUC-Plots gespeichert unter: {outdir/'plots_robustheit_deltas'}")
    except Exception as e:
        print(f"[HINWEIS] Delta AUC-Plots (Experiment 2) übersprungen: {e}")

if __name__ == "__main__":
    main()
