import pandas as pd
import glob
import os

# NEU:
import argparse
from pathlib import Path
import csv
import numpy as np

# NEU: von hier bis replace_name alles neu
try:
    from sklearn.metrics import roc_auc_score
    SKL_OK = True
except Exception:
    SKL_OK = False
    
def get_rec_dir(records_dir: str, exp_tag: str, detector: str) -> Path:
    return Path(records_dir) / f"{exp_tag}_{detector}"
    
def read_auc_from_csv(csv_path: Path):
    # Erwartete Formate:
    # 1) metric,value  \n auc,0.91234
    # 2) auc,0.91234
    if not csv_path.exists():
        return None
    val = None
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        # Header überspringen, wenn vorhanden
        for row in rows:
            if len(row) >= 2 and row[0].strip().lower() == "auc":
                try:
                    val = float(row[1])
                except Exception:
                    pass
    return val

def compute_auc_from_npy(rec_dir: Path, dataset: str):
    y_true_path = rec_dir / f"{dataset}_y_true.npy"
    y_score_path = rec_dir / f"{dataset}_y_score.npy"
    if not (y_true_path.exists() and y_score_path.exists()):
        return None
    y_true = np.load(y_true_path, allow_pickle=True)
    y_score = np.load(y_score_path, allow_pickle=True)
    # Falls y_score 2D (N,2) ist, positiv-Klasse annehmen = Spalte 1
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]
    if not SKL_OK:
        raise RuntimeError("scikit-learn nicht verfügbar. Installiere sklearn, oder liefere test_<DATASET>_auc.csv.")
    return float(roc_auc_score(y_true, y_score))

def main():
    ap = argparse.ArgumentParser(description="Erzeuge AUC-Tabelle aus test.py-Records.")
    ap.add_argument("--records_dir", default="./analysis", help="Wurzelverzeichnis für Analysis-Records")
    ap.add_argument("--exp_tag", default="exp1_records", help="Präfix des Experimentordners")
    ap.add_argument("--detectors", nargs="+", required=True, help="Liste der Detektoren (z.B. effort xception)")
    ap.add_argument("--datasets", nargs="+", required=True, help="Liste der Datensätze (z.B. UADFV Celeb-DF-v2)")
    ap.add_argument("--out_csv", default="./analysis/auc_table1.csv", help="Pfad der Ergebnis-CSV")
    ap.add_argument("--out_md", default=None, help="Optional: Pfad für Markdown-Tabelle")
    args = ap.parse_args()

    rows = []
    header = ["dataset"] + args.detectors

    for ds in args.datasets:
        row = {"dataset": ds}
        for det in args.detectors:
            rec_dir = get_rec_dir(args.records_dir, args.exp_tag, det)
            csv_path = rec_dir / f"test_{ds}_auc.csv"
            auc = read_auc_from_csv(csv_path)
            if auc is None:
                auc = compute_auc_from_npy(rec_dir, ds)
            row[det] = auc if auc is not None else ""
        rows.append(row)

    # Schreibe CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r["dataset"]] + [r.get(det, "") for det in args.detectors])

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        # Markdown-Tabelle
        with out_md.open("w") as f:
            # Kopf
            f.write("| Dataset | " + " | ".join(args.detectors) + " |\n")
            f.write("|---" * (len(args.detectors)+1) + "|\n")
            for r in rows:
                vals = []
                for det in args.detectors:
                    v = r.get(det, "")
                    if isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                f.write(f"| {r['dataset']} | " + " | ".join(vals) + " |\n")

    print(f"? AUC-Tabelle geschrieben: {out_csv}")
    if args.out_md:
        print(f"? Markdown-Tabelle: {args.out_md}")

if __name__ == "__main__":
    main()

def replace_name(csv_name: str):
    if csv_name == 'test_Celeb-DF-v1_auc':
        return_name = 'CelebDF-v1'
    elif csv_name == 'test_Celeb-DF-v2_auc':
        return_name = 'CelebDF-v2'
    elif csv_name == 'test_DeeperForensics-1.0_auc':
        return_name = 'DF-1.0'
    elif csv_name == 'test_FaceShifter_auc':
        return_name = 'FaceShifter'
    elif csv_name == 'test_DeepFakeDetection_auc':
        return_name = 'DFD'
    elif csv_name == 'test_DFDC_auc':
        return_name = 'DFDC'
    elif csv_name == 'test_DFDCP_auc':
        return_name = 'DFDCP'
    elif csv_name == 'test_FaceForensics++_auc':
        return_name = 'FF++_c23'
    elif csv_name == 'test_FaceForensics++_c40_auc':
        return_name = 'FF++_c40'
    elif csv_name == 'test_FF-DF_auc':
        return_name = 'FF-DF'
    elif csv_name == 'test_FF-F2F_auc':
        return_name = 'FF-F2F'
    elif csv_name == 'test_FF-FS_auc':
        return_name = 'FF-FS'
    elif csv_name == 'test_FF-NT_auc':
        return_name = 'FF-NT'
    elif csv_name == 'test_UADFV_auc':
        return_name = 'UADFV'
    else:
        raise ValueError(f'Unknown csv name: {csv_name}')
    return return_name

detectors = glob.glob(os.path.join('exp1_record/*'))  # Assuming the script is running in the parent directory
results = []

for detector in detectors:
    csv_files = glob.glob(f'{detector}/*.csv')

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        top_3_auc = df['Value'].nlargest(3).mean()  # Get mean of top 3 AUC
        
        test_data = os.path.basename(csv_file).replace('.csv','')  # Assuming test_data is the file name
        results.append({'Detector': detector.replace('exp1_record/', ''), 'Test Data': replace_name(test_data), 'Top-3 AUC': top_3_auc})

# Convert list of dicts to DataFrame
df_results = pd.DataFrame(results)

# Pivot the dataframe to have detectors as rows and test_data as columns
final_df = df_results.pivot(index='Detector', columns='Test Data', values='Top-3 AUC')

# Add the 'avg' column as the mean of other columns
final_df['Avg.'] = final_df.mean(axis=1)

print(final_df)

# Save the dataframe to excel
final_df.to_excel('auc_table1_fromrecord.xlsx')
