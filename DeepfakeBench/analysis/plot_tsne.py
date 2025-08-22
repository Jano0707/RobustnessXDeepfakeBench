#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t-SNE-Visualisierung (pro Dataset) aus DeepfakeBench-Ergebnissen.

Nutzt:
- JSON-Metriken in --results_dir (wie von test.py erzeugt)
- zugehörige *_feat.npy sowie *_y_true.npy (Pfade im JSON; ansonsten wird rekursiv gesucht)

Ausgabe pro Dataset:
- <outdir>/tsne_<DATASET>.png / .pdf  (Subplots je (Detector, Tag))
- <outdir>/tsne_<DATASET>__points.csv (alle 2D-Punkte mit Meta-Infos)

Aufrufbeispiel:
python analysis/plot_tsne.py \
  --results_dir analysis_outputs/metrics \
  --outdir analysis_outputs/plots/tsne
"""
from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ---------------------------
# Helfer
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
    obj = json.loads(fp.read_text(encoding="utf-8"))
    lo  = _lower(obj)
    det = (lo.get("detector") or lo.get("model") or _infer_from_name(fp.stem,0,"unknown")).strip()
    dset= (lo.get("dataset")  or _infer_from_name(fp.stem,1,"unknown")).strip()
    tag = (lo.get("tag")      or _infer_from_name(fp.stem,2,"baseline")).strip()
    return {
        "detector": _pretty_detector(det),
        "dataset": dset,
        "tag": tag,
        "__file__": str(fp),
        "feat_path": lo.get("feat_path"),
        "y_true_path": lo.get("y_true_path") or lo.get("ytrue_path"),
    }

def load_all_json(results_dir: Path) -> pd.DataFrame:
    files = list(results_dir.rglob("*.json"))
    if not files: raise SystemExit("Keine JSON-Dateien gefunden.")
    rows = []
    for f in files:
        try: rows.append(load_metric_file(f))
        except Exception as e:
            print(f"[WARN] Überspringe {f}: {e}")
    if not rows: raise SystemExit("Keine verwertbaren JSON-Dateien.")
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["dataset"].tolist()))  # Reihenfolge wie gefunden
    df["dataset"] = pd.Categorical(df["dataset"], categories=order, ordered=True)
    return df

def _guess_feat_and_labels(search_root: Path, det: str, dset: str, tag: str):
    """Finde *_feat.npy und *_y_true.npy robust anhand von Namen."""
    feat_cands = list(search_root.rglob(f"*{det}*{dset}*{tag}*feat*.npy"))
    if not feat_cands:
        feat_cands = list(search_root.rglob(f"*{dset}*feat*.npy"))
    ytrue_cands = list(search_root.rglob(f"*{det}*{dset}*{tag}*y_true*.npy"))
    if not ytrue_cands:
        ytrue_cands = list(search_root.rglob(f"*{dset}*y_true*.npy"))

    feat_path = feat_cands[0] if feat_cands else None
    ytrue_path= ytrue_cands[0] if ytrue_cands else None
    return feat_path, ytrue_path

def _load_feats_labels(feat_path: Path, y_true_path: Path):
    feats = np.load(feat_path)
    if feats.ndim > 2:
        feats = feats.reshape(feats.shape[0], -1)
    feats = feats.astype(np.float32)
    y_true = np.load(y_true_path).astype(np.int64).reshape(-1)
    # Längen angleichen, falls minimal unterschiedlich
    n = min(len(feats), len(y_true))
    return feats[:n], y_true[:n]

def _balanced_subsample(feats: np.ndarray, y: np.ndarray, max_points: int, seed: int):
    """Ausgewogenes Sampling über Klassen 0/1; fällt auf unbalanciert zurück, falls nötig."""
    if max_points is None or len(feats) <= max_points:
        return feats, y
    rng = np.random.default_rng(seed)
    idx0 = np.where(y==0)[0]
    idx1 = np.where(y==1)[0]
    if len(idx0)==0 or len(idx1)==0:
        sel = rng.choice(len(y), size=max_points, replace=False)
        return feats[sel], y[sel]
    k = max_points//2
    k0 = min(k, len(idx0)); k1 = min(k, len(idx1))
    s0 = rng.choice(idx0, size=k0, replace=False)
    s1 = rng.choice(idx1, size=k1, replace=False)
    sel = np.concatenate([s0,s1])
    rng.shuffle(sel)
    return feats[sel], y[sel]

def _safe_perplexity(n_samples: int, desired: float = 30.0):
    # Faustregel: perplexity < (n_samples-1)/3
    upper = max((n_samples-1)/3.0, 5.0)
    return float(min(desired, upper))

# ---------------------------
# t-SNE je Dataset plotten
# ---------------------------
def plot_dataset_tsne(df: pd.DataFrame, dataset: str, outdir: Path,
                      search_root: Path,
                      max_points: int = 3000,
                      perplexity: float = 30.0,
                      learning_rate: float = 200.0,
                      seed: int = 1024,
                      pca_dim: int = 50):
    sub = df[df["dataset"].astype(str)==dataset]
    if sub.empty: return

    groups = []  # (det, tag, feats, y)
    for _, row in sub.sort_values(["detector","tag","__file__"]).iterrows():
        det, tag = row["detector"], row["tag"]
        feat_p = Path(row["feat_path"]) if row.get("feat_path") else None
        y_p    = Path(row["y_true_path"]) if row.get("y_true_path") else None
        if not (feat_p and feat_p.exists() and y_p and y_p.exists()):
            feat_p, y_p = _guess_feat_and_labels(search_root, det, dataset, tag)
        if not (feat_p and y_p and Path(feat_p).exists() and Path(y_p).exists()):
            print(f"[WARN] Keine Features/Labels für {det} / {dataset} / {tag} -> übersprungen.")
            continue
        try:
            feats, y = _load_feats_labels(feat_p, y_p)
            feats, y = _balanced_subsample(feats, y, max_points=max_points, seed=seed)
        except Exception as e:
            print(f"[WARN] Laden/Sampling fehlgeschlagen für {feat_p}: {e}")
            continue
        groups.append((det, tag, feats, y))

    if not groups:
        print(f"[HINWEIS] Dataset '{dataset}': keine t-SNE-Gruppen gefunden.")
        return

    # Figure-Layout: Grid für alle (det, tag)
    G = len(groups)
    ncols = 2 if G>=2 else 1
    nrows = math.ceil(G / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0*ncols, 4.8*nrows), dpi=300)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    csv_rows = []
    for i, (det, tag, feats, y) in enumerate(groups):
        ax = axes[i]
        # Perplexity sicher wählen
        perf = _safe_perplexity(len(feats), desired=perplexity)
        # Optional: Vor-Reduktion via PCA (nur wenn Dimension sehr groß)
        X = feats
        # z-Score pro Dimension (stabilisiert t-SNE)
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
        if pca_dim and X.shape[1] > pca_dim:
            # einfache PCA mit np.linalg.svd (ohne sklearn-Abhängigkeit)
            Xc = X - X.mean(axis=0, keepdims=True)
            # ökonomischer SVD-Weg: auf max pca_dim beschränken
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            X = (U[:, :pca_dim] * S[:pca_dim])

        tsne = TSNE(n_components=2, perplexity=perf, random_state=seed,
                    learning_rate=learning_rate, init="pca", n_iter=1000, verbose=0)
        emb = tsne.fit_transform(X)

        # Plot: 0=Real blau (o), 1=Fake orange (x)
        real = (y==0); fake = (y==1)
        ax.scatter(emb[real,0], emb[real,1], s=8, c="tab:blue", marker="o", alpha=0.7, label=f"Real (n={real.sum()})")
        ax.scatter(emb[fake,0], emb[fake,1], s=8, c="tab:orange", marker="o", alpha=0.7, label=f"Fake (n={fake.sum()})")

        ax.set_title(f"{det} - {dataset}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)

        # CSV-Zeilen sammeln
        for (x,y2), lab in zip(emb, y):
            csv_rows.append({
                "dataset": dataset, "detector": det, "tag": tag,
                "x": float(x), "y": float(y2), "label": int(lab)
            })

    # leere Achsen ausblenden (falls Grid größer als Gruppen)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"tsne_{dataset}.png"
    pdf = outdir / f"tsne_{dataset}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(outdir / f"tsne_{dataset}__points.csv", index=False)
    print(f"[OK] t-SNE gespeichert: {png} / {pdf}")

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="analysis_outputs/metrics", help="Ordner mit JSON-Metriken (rekursiv).")
    ap.add_argument("--outdir", default="analysis_outputs/plots/tsne", help="Ausgabeordner.")
    ap.add_argument("--max_points", type=int, default=3000, help="Max. Punkte pro (det, tag) nach balanciertem Sampling.")
    ap.add_argument("--perplexity", type=float, default=20, help="t-SNE Perplexity (wird automatisch auf zulässiges Maximum gekappt).")
    ap.add_argument("--learning_rate", type=float, default=250, help="t-SNE learning rate.")
    ap.add_argument("--seed", type=int, default=1024, help="Random-Seed.")
    ap.add_argument("--pca_dim", type=int, default=50, help="Vor-Reduktion per PCA (0 = aus).")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    outdir      = Path(args.outdir).resolve()
    df = load_all_json(results_dir)

    for dset in df["dataset"].cat.categories:
        plot_dataset_tsne(df, str(dset), outdir,
                          search_root=results_dir,
                          max_points=args.max_points,
                          perplexity=args.perplexity,
                          learning_rate=args.learning_rate,
                          seed=args.seed,
                          pca_dim=args.pca_dim)

if __name__ == "__main__":
    main()
