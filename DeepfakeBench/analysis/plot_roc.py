# analysis/plot_roc.py
"""
ROC-Kurven zeichnen (robust gegen gleichnamige Datasets mit verschiedenen Tags).

- Liest JSON-Metriken unter --results_dir (rekursiv)
- Nutzt y_true/y_score aus JSON; fällt andernfalls auf Dateisuche zurück
- Erzeugt:
  1) Pro (Dataset, Experiment/Tag-Gruppe) einen Plot: roc_<DISPLAY>.png/.pdf
  2) Sammelplot Experiment 1 (gen): roc_GEN_All.png/.pdf (Titel: "ROC - Generalisierung")
  3) Sammelplot Experiment 2 (rob) je Basis-Dataset:
     roc_ROB_Celeb-DF-v2.png/.pdf und roc_ROB_DeepFakeDetection.png/.pdf
"""

from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILL_ALPHA = 0.18

ROB_GROUPS_WITH_BASELINE = {
    "JPEG", "Gesichtsglättung", "Schwarz-Weiß", "Text-Overlay", "Text-Overlay-Augen"
}

# ---------------------------
# Helpers
# ---------------------------

def _lower(d): return {str(k).lower(): v for k,v in d.items()}

def _pretty_detector(name: str) -> str:
    m = {"effort":"Effort", "xception":"Xception"}
    s = (name or "").strip()
    return m.get(s.lower(), s[:1].upper()+s[1:])

def _infer_from_name(stem: str, idx: int, default=None):
    parts = stem.split("__")
    return parts[idx] if 0 <= idx < len(parts) else default

def _norm_variant(v: str) -> str:
    """
    Normalisiert Schreibweisen und Kurzformen auf die kanonischen Varianten.
    Akzeptiert auch 'domain', 'cross domain', 'weiss', 'smoothing', 'augen', 'text' etc.
    """
    if not v:
        return ""
    s = str(v).strip().lower().replace("_", "-").replace("  ", " ")

    # Häufige Kurzformen/Abkürzungen auffangen
    aliases = {
        "baseline": "Baseline",
        "gen": "Baseline",  # falls mal nur 'gen' steht
        "within": "Within-Domain",
        "within-domain": "Within-Domain",
        "within domain": "Within-Domain",
        "domain": "Within-Domain",  # <- aus deinem Dump
        "cross": "Cross-Domain",
        "cross-domain": "Cross-Domain",
        "cross domain": "Cross-Domain",

        "jpeg": "JPEG",
        "face": "Face-Smoothing",
        "smoothing": "Face-Smoothing",
        "face-smoothing": "Face-Smoothing",

        "schwarz-weiss": "Schwarz-Weiß",
        "schwarz-weiß": "Schwarz-Weiß",
        "schwarz weiss": "Schwarz-Weiß",
        "weiss": "Schwarz-Weiß",
        "weiß": "Schwarz-Weiß",
        "s-w": "Schwarz-Weiß",
        "s-w.": "Schwarz-Weiß",

        "text": "Text-Overlay",
        "text-overlay": "Text-Overlay",
        "text overlay": "Text-Overlay",

        "text-augen": "Text-Overlay-Augen",
        "text-augens": "Text-Overlay-Augen",
        "text overlay augen": "Text-Overlay-Augen",
        "augen": "Text-Overlay-Augen",
    }

    # direkter Treffer?
    if s in aliases:
        return aliases[s]

    # letzte Rettung: Bindestriche vereinheitlichen und nochmal prüfen
    s = s.replace("--", "-")
    return aliases.get(s, v if v and v[0].isupper() else v.title())

def _parse_tag_like(raw_tag: str):
    """
    Zerlegt 'raw_tag' in (exp, tag_norm, raw).
    WICHTIG: Varianten mit Bindestrich korrekt zusammensetzen:
      'gen-Within-Domain'  -> exp='gen',  variant='Within-Domain'
      'rob-Face-Smoothing' -> exp='rob',  variant='Face-Smoothing'
      'Baseline'           -> exp='',     variant='Baseline'
    """
    raw = (raw_tag or "").strip()
    if not raw:
        return "", _norm_variant(raw), raw

    parts = [p.strip() for p in raw.split("-") if p.strip()]
    parts_low = [p.lower() for p in parts]

    # Falls erstes Teil 'gen' oder 'rob' ist, ist ab Index 1 die Variante
    if parts_low and parts_low[0] in {"gen", "rob"}:
        exp = parts_low[0]
        variant = "-".join(parts[1:]) if len(parts) > 1 else "Baseline"
    else:
        exp = ""
        # ganze Zeichenkette als Variante interpretieren (z. B. 'Baseline')
        variant = raw

    return exp, _norm_variant(variant), raw


def _rob_base_and_group_from_raw(dset: str):
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

def _display_dataset_name(dset: str, exp: str = "", tag_norm: str = "") -> str:
    if (exp or "").lower() == "rob":
        base, group = _rob_base_and_group_from_raw(dset)
        return base if group == "Baseline" else f"{base} ({group})"
    if (exp or "").lower() == "gen":
        if tag_norm in ("Within-Domain", "Cross-Domain"):
            return f"{dset} ({tag_norm})"
        return dset
    base, group = _rob_base_and_group_from_raw(dset)
    return base if group == "Baseline" else f"{base} ({group})"

_slug_rx = re.compile(r"[^A-Za-z0-9._+-]+")
def _slugify(s: str) -> str:
    s = s.replace("(", "_").replace(")", "_").replace("/", "_")
    s = s.replace("__", "_")
    return _slug_rx.sub("_", s).strip("_")

def _augment_with_baseline_for_rob(df_all: pd.DataFrame, df_sub: pd.DataFrame) -> pd.DataFrame:
    if df_sub.empty:
        return df_sub

    exp0 = str(df_sub["exp"].iloc[0] if "exp" in df_sub.columns else "").lower()
    dataset0 = str(df_sub["dataset"].iloc[0] if "dataset" in df_sub.columns else "")
    base0, group0 = _rob_base_and_group_from_raw(dataset0)

    if exp0 != "rob" or group0 not in ROB_GROUPS_WITH_BASELINE:
        return df_sub

    baseline = df_all[
        (df_all["exp"].astype(str).str.lower() == "rob") &
        (df_all["dataset"].astype(str) == base0)
    ]
    if baseline.empty:
        return df_sub  # nichts zu ergänzen

    cols = list(df_sub.columns)
    aug = pd.concat([baseline[cols], df_sub[cols]], axis=0, ignore_index=True)
    return aug
# ---------------------------
# Laden
# ---------------------------

def load_metric_file(fp: Path):
    obj = json.loads(Path(fp).read_text(encoding="utf-8"))
    lo  = _lower(obj)
    det = (lo.get("detector") or lo.get("model") or _infer_from_name(fp.stem,0,"unknown")).strip()
    dset= (lo.get("dataset")  or _infer_from_name(fp.stem,1,"unknown")).strip()
    tag_raw = (lo.get("tag")  or _infer_from_name(fp.stem,2,"baseline")).strip()
    exp = (lo.get("exp") or "").strip().lower()
    if not exp:
        exp, tag_norm, _ = _parse_tag_like(tag_raw)
    else:
        _, tag_norm, _ = _parse_tag_like(tag_raw)
    # Pfade
    ytrue = lo.get("y_true_path") or lo.get("ytrue_path")
    yscore= lo.get("y_score_path") or lo.get("yscore_path")
    return {
        "detector": _pretty_detector(det),
        "dataset": dset,
        "tag_raw": tag_raw,
        "tag_norm": tag_norm,
        "exp": (exp or "").lower(),     # 'gen' | 'rob' | ''
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
    return pd.DataFrame(rows)

# ---------------------------
# Vorhersagen & ROC
# ---------------------------

def _guess_pred_files(search_root: Path, det: str, dset: str, tag_raw: str):
    # robuste Suche; der Tag wird bewusst mit in die Muster aufgenommen
    cands_true  = list(search_root.rglob(f"*{det}*{dset}*{tag_raw}*y_true*.npy")) + \
                  list(search_root.rglob(f"*{dset}*{tag_raw}*y_true*.npy")) + \
                  list(search_root.rglob(f"*{dset}*y_true*.npy"))
    cands_score = list(search_root.rglob(f"*{det}*{dset}*{tag_raw}*y_score*.npy")) + \
                  list(search_root.rglob(f"*{dset}*{tag_raw}*y_score*.npy")) + \
                  list(search_root.rglob(f"*{dset}*y_score*.npy"))
    y_true = cands_true[0]  if cands_true  else None
    y_score= cands_score[0] if cands_score else None
    return y_true, y_score

def _load_preds(y_true_path: Path, y_score_path: Path):
    y_true  = np.load(y_true_path)
    y_score = np.load(y_score_path)
    y_true  = y_true.astype(np.int64).reshape(-1)
    y_score = y_score.reshape(-1, ) if y_score.ndim==1 else y_score[:,1]
    return y_true, y_score

def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y_true = y_true[order]; y_score = y_score[order]
    P = (y_true==1).sum(); N = (y_true==0).sum()
    if P==0 or N==0:
        return np.array([0,1]), np.array([0,1]), 0.5
    tps = np.cumsum(y_true==1); fps = np.cumsum(y_true==0)
    TPR = np.concatenate(([0.0], tps / max(P,1), [1.0]))
    FPR = np.concatenate(([0.0], fps / max(N,1), [1.0]))
    auc = float(np.trapz(TPR, FPR))
    return FPR, TPR, auc

def _curve_colors(n_curves: int):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n_curves)]

# ---------------------------
# Einzel-Plot (pro Dataset+Gruppe)
# ---------------------------

def plot_one(df_sub: pd.DataFrame,
             display_name: str,
             outdir: Path,
             search_root: Path,
             legend_formatter,
             title_prefix: str = "ROC - ",
             forced_fname: str = None):

    if df_sub.empty:
        return
    context = {"mode": "single"}

    # Farben je Kurve (nicht je Detektor)
    rows_sorted = df_sub.sort_values(["detector","__file__"]).to_dict("records")
    colors = _curve_colors(len(rows_sorted))

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=300)
    ax.set_title(f"{title_prefix}{display_name}", fontsize=12)
    ax.set_xlabel("Falsch-positiv-Rate (FPR)")
    ax.set_ylabel("Richtig-positiv-Rate (RPR)")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    legend_items = []  # (auc, handle, label)

    for i, r in enumerate(rows_sorted):
        det = r["detector"]

        # Predictions laden (unter Berücksichtigung von Tag/Raw!)
        ytp = r.get("y_true_path"); ysp = r.get("y_score_path")
        if ytp and ysp and Path(ytp).exists() and Path(ysp).exists():
            yt, ys = Path(ytp), Path(ysp)
        else:
            yt, ys = _guess_pred_files(search_root, det, r["dataset"], r["tag_raw"])
        if (yt is None) or (ys is None):
            print(f"[WARN] Keine Predictions für {det} / {r['dataset']} / {r['tag_raw']}")
            continue

        y_true, y_score = _load_preds(yt, ys)
        FPR, TPR, auc = _roc_curve(y_true, y_score)

        line, = ax.plot(FPR, TPR, color=colors[i], linewidth=1.8)
        ax.fill_between(FPR, TPR, 0.0, color=colors[i], alpha=FILL_ALPHA, linewidth=0, zorder=-1)

        label = legend_formatter(det, r, auc, context)
        legend_items.append((auc, line, label))

    # Zufalls-Klassifikator
    rand_line, = ax.plot([0,1],[0,1], ":", color="#666666", linewidth=1.4)
    rand_label = "zufällige Klassifizierung"

    ax.set_xlim(0,1); ax.set_ylim(0,1)

    # Legende: nach AUC absteigend sortieren, Zufall zuletzt
    legend_items.sort(key=lambda t: (t[0] if t[0] is not None else -np.inf), reverse=True)
    handles = [h for _, h, _ in legend_items] + [rand_line]
    labels  = [lb for _, _, lb in legend_items] + [rand_label]
    ax.legend(handles, labels, loc="lower right", frameon=True, framealpha=0.9)

    outdir.mkdir(parents=True, exist_ok=True)

    exp0      = str(df_sub["exp"].iloc[0] if "exp" in df_sub.columns else "").lower()
    tag0      = str(df_sub["tag_norm"].iloc[0] if "tag_norm" in df_sub.columns and pd.notna(df_sub["tag_norm"].iloc[0]) else "")
    dataset0  = str(df_sub["dataset"].iloc[0] if "dataset" in df_sub.columns else "")
    disp0     = display_name

    if forced_fname:
        base = outdir / forced_fname
    else:
        if exp0 == "gen":
            if tag0 in ("Within-Domain", "Cross-Domain"):
                fname = f"roc_{_slugify(dataset0)}_{_slugify(tag0)}"
            else:
                fname = f"roc_{_slugify(dataset0)}"
        elif exp0 == "rob":
            fname = f"roc_{_slugify(disp0)}"
        else:
            fname = f"roc_{_slugify(disp0)}"
        base = outdir / fname

    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[OK] ROC: {base.with_suffix('.png')}")


def plot_ffpp_within_domain(dff: pd.DataFrame, outdir: Path, search_root: Path):
    if dff.empty:
        print("[WARN] plot_ffpp_within_domain: DataFrame leer.")
        return

    # Hilfsnormalisierer
    def _norm(s):
        return (str(s or "").strip())

    def _is_within_domain(x):
        xs = str(x or "").strip().lower()
        return xs in {"within-domain", "within domain", "within_domain"}
    cand = dff[dff["dataset"].astype(str).str.strip().eq("FaceForensics++") |
               dff["display"].astype(str).str.strip().eq("FaceForensics++ (Within-Domain)")].copy()
)
    cand = cand[cand["exp"].astype(str).str.strip().str.lower().eq("gen")]

    mask_within = cand["tag_norm"].apply(_is_within_domain) | \
                  cand["display"].astype(str).str.contains(r"\(Within-Domain\)\s*$", regex=True)
    sub = cand[mask_within].copy()

    if sub.empty:
        print("[WARN] Keine Einträge für FaceForensics++ (Within-Domain) gefunden.")
        print("       Verfügbare Kombinationen (erste 10):")
        dbg = dff[["dataset","display","exp","tag_norm","tag_raw"]].head(10)
        print(dbg.to_string(index=False))
        return

    def legend_fmt(det, _row, auc, _ctx):
        return f"{det} AUC = {auc:.4f}"

    forced_fname = f"roc_{_slugify('FaceForensics++_Within-Domain')}"

    plot_one(
        sub,
        display_name="FaceForensics++ (Within-Domain)",
        outdir=outdir,
        search_root=search_root,
        legend_formatter=legend_fmt,
        title_prefix="ROC - ",
        forced_fname=forced_fname  # <— nutzt deinen erweiterten plot_one
    )


# ---------------------------
# Sammelplots
# ---------------------------

def plot_experiment_gen(df: pd.DataFrame, outdir: Path, search_root: Path):
    sub = df[df["exp"]=="gen"].copy()
    if sub.empty: return

    def legend_fmt(det, r, auc, context):
        disp = _display_dataset_name(r["dataset"], exp="gen", tag_norm=r["tag_norm"])
        return f"{det} ({disp}) AUC = {auc:.4f}"

    display = "Generalisierung"
    context = {"mode": "combined"}
    rows = sub.sort_values(["dataset","detector","__file__"]).to_dict("records")
    colors = _curve_colors(len(rows))

    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=300)
    ax.set_title("ROC - Generalisierung", fontsize=12)
    ax.set_xlabel("Falsch-positiv-Rate (FPR)")
    ax.set_ylabel("Richtig-positiv-Rate (RPR)")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    legend_items = []
    for i, r in enumerate(rows):
        det = r["detector"]
        ytp = r.get("y_true_path"); ysp = r.get("y_score_path")
        if ytp and ysp and Path(ytp).exists() and Path(ysp).exists():
            yt, ys = Path(ytp), Path(ysp)
        else:
            yt, ys = _guess_pred_files(search_root, det, r["dataset"], r["tag_raw"])
        if (yt is None) or (ys is None): 
            continue
        y_true, y_score = _load_preds(yt, ys)
        FPR, TPR, auc = _roc_curve(y_true, y_score)
        line, = ax.plot(FPR, TPR, color=colors[i], linewidth=1.8)
        #ax.fill_between(FPR, TPR, 0.0, color=colors[i], alpha=FILL_ALPHA, linewidth=0, zorder=-1)
        legend_items.append((auc, line, legend_fmt(det, r, auc, context)))

    rand_line, = ax.plot([0,1],[0,1], ":", color="#666666", linewidth=1.4)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    legend_items.sort(key=lambda t: (t[0] if t[0] is not None else -np.inf), reverse=True)
    handles = [h for _, h, _ in legend_items] + [rand_line]
    labels  = [lb for _, _, lb in legend_items] + ["zufällige Klassifizierung"]
    ax.legend(handles, labels, loc="lower right", frameon=True, framealpha=0.9)

    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / "roc_GEN_All"
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[OK] ROC: {base.with_suffix('.png')}")

def plot_experiment_rob(df: pd.DataFrame, outdir: Path, search_root: Path):
    sub = df[df["exp"]=="rob"].copy()
    if sub.empty: return
    bases = sorted({ _rob_base_and_group_from_raw(s)[0] for s in sub["dataset"].astype(str) })
    for base in bases:
        sub_b = sub[sub["dataset"].astype(str).apply(lambda s: _rob_base_and_group_from_raw(s)[0] == base)]
        if sub_b.empty: continue

        def legend_fmt(det, r, auc, context):
            _base, group = _rob_base_and_group_from_raw(r["dataset"])
            return f"{det} AUC = {auc:.4f}" if group=="Baseline" else f"{det} ({group}) AUC = {auc:.4f}"

        rows = sub_b.sort_values(["dataset","detector","__file__"]).to_dict("records")
        colors = _curve_colors(len(rows))

        fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=300)
        ax.set_title(f"ROC - Robustheit ({base})", fontsize=12)
        ax.set_xlabel("Falsch-positiv-Rate (FPR)")
        ax.set_ylabel("Richtig-positiv-Rate (RPR)")
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

        legend_items = []
        for i, r in enumerate(rows):
            det = r["detector"]
            ytp = r.get("y_true_path"); ysp = r.get("y_score_path")
            if ytp and ysp and Path(ytp).exists() and Path(ysp).exists():
                yt, ys = Path(ytp), Path(ysp)
            else:
                yt, ys = _guess_pred_files(search_root, det, r["dataset"], r["tag_raw"])
            if (yt is None) or (ys is None):
                continue
            y_true, y_score = _load_preds(yt, ys)
            FPR, TPR, auc = _roc_curve(y_true, y_score)
            line, = ax.plot(FPR, TPR, color=colors[i], linewidth=1.8)
            #ax.fill_between(FPR, TPR, 0.0, color=colors[i], alpha=FILL_ALPHA, linewidth=0, zorder=-1)
            legend_items.append((auc, line, legend_fmt(det, r, auc, {"mode":"combined"})))

        rand_line, = ax.plot([0,1],[0,1], ":", color="#666666", linewidth=1.4)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        legend_items.sort(key=lambda t: (t[0] if t[0] is not None else -np.inf), reverse=True)
        handles = [h for _, h, _ in legend_items] + [rand_line]
        labels  = [lb for _, _, lb in legend_items] + ["zufällige Klassifizierung"]
        ax.legend(handles, labels, loc="lower right", frameon=True, framealpha=0.9)

        outdir.mkdir(parents=True, exist_ok=True)
        base_path = outdir / f"roc_ROB_{_slugify(base)}"
        fig.tight_layout()
        fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12)
        fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        print(f"[OK] ROC: {base_path.with_suffix('.png')}")

# ---------------------------
# Main:
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="analysis_outputs/metrics", help="Ordner mit JSON-Metriken (rekursiv).")
    ap.add_argument("--outdir", default="analysis_outputs/plots/roc", help="Ausgabeordner für Plots.")
    args = ap.parse_args()

    df = load_all_json(Path(args.results_dir))

    records = []
    for _, r in df.iterrows():
        records.append({
            "detector": r["detector"],
            "dataset": r["dataset"],
            "display": _display_dataset_name(r["dataset"], exp=r["exp"], tag_norm=r["tag_norm"]),
            "exp": r["exp"], "tag_norm": r["tag_norm"], "tag_raw": r["tag_raw"],
            "__file__": r["__file__"],
            "y_true_path": r["y_true_path"], "y_score_path": r["y_score_path"],
        })
    dff = pd.DataFrame(records)

    outdir = Path(args.outdir)
    search_root = Path(args.results_dir).resolve()

    plot_ffpp_within_domain(dff, outdir, search_root)
    for (exp, dataset, tag_norm), sub in dff.groupby(["exp","dataset","tag_norm"], dropna=False):
        if sub.empty:
            continue
        display = _display_dataset_name(dataset, exp=exp, tag_norm=tag_norm)

        base_name, group_name = _rob_base_and_group_from_raw(dataset)

        def legend_fmt(det, row, auc, _ctx):
            if (exp or "").lower() == "rob" and group_name in ROB_GROUPS_WITH_BASELINE:
                _b, _g = _rob_base_and_group_from_raw(str(row["dataset"]))
                if _g == "Baseline":
                    return f"{det} (Baseline) AUC = {auc:.4f}"
                return f"{det} AUC = {auc:.4f}"
            # Standard (alle anderen Fälle)
            return f"{det} AUC = {auc:.4f}"

        sub_aug = _augment_with_baseline_for_rob(dff, sub)
        plot_one(sub_aug, display, outdir, search_root, legend_fmt, title_prefix="ROC - ")

    plot_experiment_gen(dff, outdir, search_root)
    plot_experiment_rob(dff, outdir, search_root)

if __name__ == "__main__":
    main()

