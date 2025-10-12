# DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection (NeurIPS 2023 D&B)
## 🔧 Nutzung im Rahmen der Bachelorarbeit


Dieses Repository enthält eine angepasste Version von **DeepfakeBench**, die im Rahmen einer Bachelorarbeit erweitert wurde. Ziel ist es, die Durchführung und Nachvollziehbarkeit von Experimenten zu erleichtern.  
Neben den allgemeinen Informationen aus der Original-README sind hier:

1. **Eine Schritt-für-Schritt-Anleitung** zur Reproduktion der Experimente enthalten.  
2. **Alle vorgenommenen Änderungen** am Framework dokumentiert.  

---

## 🚀 Schritt-für-Schritt-Anleitung

### 1. Modelle & Gewichte herunterladen
- Lade das **Effort-Backbone** von Hugging Face:  
  [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14)  
  und speichere es unter `DeepfakeBench/training/networks`.  
  > Hinweis: Passe anschließend den Pfad in `training/detectors/effort_detector.py` in der Funktion `build_backbone()` an.

- Lade die vortrainierten Gewichte:  
  - [Xception (DeepfakeBench Releases)](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1)  
  - [Effort (FaceForensics++ Checkpoints)](https://github.com/YZY-stack/Effort-AIGI-Detection)  

  Beide Gewichte müssen unter `DeepfakeBench/training/weights/` abgelegt werden.  
  Da die Effort-Gewichte ein Prefix enthalten, das vom Testskript nicht verarbeitet werden kann, nutze:  

  ```
  cd DeepfakeBench
  python convert_effort_ckpt.py training/weights/effort_clip_L14_trainOn_FaceForensic.pth training/weights/effort_clip_L14_trainOn_FaceForensic_stripped.pth
  ```

### 2. Datensätze herunterladen
  Alle Datensätze müssen axakt so abgelegt werden, wie es im folgenden beschrieben wird.
  Speicherort: `DeepfakeBench/datasets/rgb/`
  

#### FaceForensics++  
$\rightarrow$ Basis für *Baseline-Generalisierung* und *Within-Domain-Tests*  

1. Lade den Datensatz über das [Google-Formular](https://github.com/ondyari/FaceForensics/tree/master/dataset) herunter.  
2. Anschließend erhalten Sie per Mail ein Download-Skript, dass als download-FaceForensics.py abgelegt werden muss.
3. Führen Sie die Downloads mit folgendem Skript aus:  

```
python download-FaceForensics.py <Pfad:.../datasets/rgb/FaceForensics++> -d original -c c23 -t videos --server EU2
python download-FaceForensics.py <Pfad:.../datasets/rgb/FaceForensics++> -d Deepfakes -c c23 -t videos --server EU2
python download-FaceForensics.py <Pfad:.../datasets/rgb/FaceForensics++> -d Face2Face -c c23 -t videos --server EU2
python download-FaceForensics.py <Pfad:.../datasets/rgb/FaceForensics++> -d FaceSwap -c c23 -t videos --server EU2
python download-FaceForensics.py <Pfad:.../datasets/rgb/FaceForensics++> -d FaceShifter -c c23 -t videos --server EU2
python download-FaceForensics.py <Pfad:.../datasets/rgb/FaceForensics++> -d NeuralTextures -c c23 -t videos --server EU2
```
4. Kopiere zusätzlich die Splits (`train.json`, `val.json`, `test.json`) in den Ordner:
`DeepfakeBench/datasets/rgb/FaceForensics++/`

#### DeepFakeDetection
$\rightarrow$ Für *Cross-Domain-Evaluierung* und *Robustheitstests*

```
python download-FaceForensics.py <Pfad:.../datasets/rgb/DeepFakeDetection> -d DeepFakeDetection_original -c c23 -t videos --server EU2
python download-FaceForensics.py <Pfad:.../datasets/rgb/DeepFakeDetection> -d DeepFakeDetection -c c23 -t videos --server EU2
```

#### Celeb-DF-v2
$\rightarrow$ Ebenfalls für *Cross-Domain-Evaluierung* und *Robustheitstests*

1. Lade den Datensatz über das [Google-Formular](https://github.com/yuezunli/celeb-deepfakeforensics) herunter.
2. Entpacke den Inhalt nach:
`DeepfakeBench/datasets/rgb/Celeb-DF-v2`

---

Die Datensatzstruktur sollte danach wie folgt aussehen:

```
datasets
├── lmdb
|   ├── FaceForensics++_lmdb
|   |   ├── data.mdb
|   |   ├── lock.mdb
|   ├── Celeb-DF-v2_lmdb
|   |   ├── data.mdb
|   |   ├── lock.mdb
|   ├── DeepfakeDetection_lmdb
|   |   ├── data.mdb
|   |   ├── lock.mdb
├── rgb
|   ├── FaceForensics++
|   │   ├── original_sequences
|   │   │   ├── youtube
|   │   │   │   ├── c23
|   │   │   │   │   ├── videos
|   │   │   │   │   │   └── *.mp4
|   │   ├── manipulated_sequences
|   │   │   ├── Deepfakes
|   │   │   │   ├── c23
|   │   │   │   │   └── videos
|   │   │   │   │   │   └── *.mp4
│   │   |   ├── Face2Face
│   |   │   │   ├── ...
|   │   │   ├── FaceSwap
|   │   │   │   ├── ...
|   │   │   ├── NeuralTextures
|   │   │   │   ├── ...
|   │   │   ├── FaceShifter
|   │   │   │   ├── ...
|   ├── Celeb-DF-v2
|   │   ├── Celeb-real
|   │   │   ├── videos
|   │   │   │   ├── *.mp4
|   │   ├── Celeb-synthesis
|   │   │   ├── videos
|   │   │   │   ├── *.mp4
|   │   ├── YouTube-real
|   │   │   ├── videos
|   │   │   │   ├── *.mp4
|   ├── DeepFakeDetection
|   │   ├── original_sequences
|   │   │   ├── actors
|   │   │   │   ├── c23
|   │   │   │   │   ├── videos
|   │   │   │   │   │   └── *.mp4
|   │   ├── manipulated_sequences
|   │   │   ├── DeepFakeDetection
|   │   │   │   ├── c23
|   │   │   │   │   └── videos
|   │   │   │   │   │   └── *.mp4
```
---

### 3. Preprocessing

Die Datensätze werden so vorbereitet, dass sie für Tests mit DeepfakeBench geeignet sind (Frame-Extraktion, Face-Cropping/Alignment, Split-Erzeugung, LMDB).

> **Konfiguration:**  
> Passe in `preprocessing/config.yaml` den gewünschten Datensatz und die Pfade an (für Preprocessing, Rearrangement und LMDB-Erstellung).

1) **Frames extrahieren & Gesichter ausrichten**

```
cd preprocessing
python preprocess.py
```
2) **Rearrangement (Train/Val/Test erzeugen)**  
- FF++: verwendet die heruntergeladenen `train.json`, `val.json`, `test.json`  
- DFD: keine öffentlichen Splits $\rightarrow$ es werden alle Bilder genutzt  
- Celeb-DF-v2: nutzt die bereitgestellte TXT-Splitdatei

```
python rearrange.py
```
3) **LMDB erstellen**  
- `dataset_size` = 20 für FF++  
- `dataset_size` = 12 für DFD  
- `dataset_size` = 20 für Celeb-DF-v2

```
cd ..
python preprocessing/dataset2lmdb_test.py --dataset_size <SIZE>
```

### 4. Bildmanipulationen anwenden

Zur Simulation sozialmedientypischer Veränderungen werden Duplikate der Datensätze erzeugt und **Bildmanipulationen** auf die jeweiligen `frames/`-Ordner angewandt.

> **Namenskonvention:**  
> Erzeuge Kopien von DeepFakeDetection und Celeb-DF-v2 mit Suffixen wie `-S_W` (Schwarz/Weiß), `-JPEG`, `-TEXT`, `-TEXT-Augen`, `-FACE` (Face Smoothing).  
> Beispiel: `Celeb-DF-v2-S_W`, `DeepFakeDetection-JPEG`, …

**Beispielanwendungen der Bildmanipulationen:**
```
cd Bildmanipulationen
python face-manipulations.py --function=black_white
--input=.../datasets/rgb/Celeb-DF-v2/YouTube-real/frames
--output=.../datasets/rgb/Celeb-DF-v2-S_W/YouTube-real/frames
```
**Gesichtsglättung:**
```
cd face-smoothing
python infer.py
--input '.../datasets/rgb/DeepFakeDetection/manipulated_sequences/DeepFakeDetection/c23/frames'
--output '.../datasets/rgb/DeepFakeDetection-FACE/manipulated_sequences/DeepFakeDetection/c23/frames'
```

> **Wichtig:** Für jeden neu erzeugten (manipulierten) Datensatz **Rearrangement** (Kap. 3, Schritt 2) und **LMDB-Erstellung** (Kap. 3, Schritt 3) erneut durchführen.

---

### 5. Tests ausführen

Passe in `training/config/test_config.yaml` die Pfade zu **LMDB** und **dataset_json** an.

**Generisches Testkommando:**
```
python3 training/test.py
--detector_path ./training/config/detector/<Detektor>.yaml
--test_dataset "<Datensatzname>"
--weights_path ./training/weights/<Detektor>.pth
--exp <Experiment>
--tag <Test>
--metric_outdir analysis_output/metrics
```

**Parameter:**
- `<Detektor>`: `xception` | `effort`  
- `<Experiment>`: `gen` (Generalisierung) | `rob` (Robustheit)  
- `<Test>`: `Baseline` | `Within-Domain` | `Cross-Domain` | `JPEG` | `Schwarz-Weiss` | `Gesichtsglaettung` | `Text-Overlay` | `Text-Overlay-Augen`

**Beispiele:**

Baseline-Generalisierung mit Xception auf der Trainingsmenge von FF++:
Ändere in `training/test.py` `use_train_data = True`.

```
python3 training/test.py
--detector_path ./training/config/detector/xception.yaml
--test_dataset "FaceForensics++"
--weights_path ./training/weights/xception_best.pth
--exp gen
--tag Baseline
--metric_outdir analysis_output/metrics
```

Cross-Domain mit Effort auf Celeb-DF-v2:
Ändere wieder `training/test.py` `use_train_data = False`.

```
python3 training/test.py
--detector_path ./training/config/detector/effort.yaml
--test_dataset "Celeb-DF-v2"
--weights_path ./training/weights/effort_clip_L14_trainOn_FaceForensic_stripped.pth
--exp gen
--tag Cross-Domain
--metric_outdir analysis_output/metrics
```

Robustheitstest mit Effort auf DeepFakeDetection-JPEG:
Weiterhin `training/test.py` `use_train_data = False`.

```
python3 training/test.py
--detector_path ./training/config/detector/effort.yaml
--test_dataset "DeepFakeDetection-JPEG"
--weights_path ./training/weights/effort_clip_L14_trainOn_FaceForensic_stripped.pth
--exp rob
--tag JPEG
--metric_outdir analysis_output/metrics
```

---

### 6. Ergebnisse analysieren

Metriken werden in der Konsole ausgegeben und unter `analysis_output/metrics/` gespeichert. Die nachfolgenden Skripte erzeugen Tabellen, ROC-Kurven und t-SNE-Plots. Die Skripte filtern ahand der in `training/test.py` gesetzten Argumente --exp und --tag und erzeugen somit automatisch die gewünschten Abbildungen.

**Tabellen:**
Erstellt pro Experiment (Generalisierung, Robustheit) eine zusammenfassende Tabelle mit allen verwendeten Datensätzen und Detektoren (AUC, ACC). Die Ergebnisse landen in `analysis_output/tables`. 

```
python analysis/create_tables.py
```

**ROC-Kurven:**
Erzeugt ROC-Kurven je Datensatz und Detektor.

```
python analysis/plot_roc.py
```


### ⚙️ Änderungen am Framework

Dieser Abschnitt dokumentiert alle Anpassungen am DeepfakeBench-Framework, die für die Reproduktion der Experimente im Rahmen der Bachelorarbeit erforderlich sind. Ziel der Änderungen ist eine robuste Auswertung (AUC, ACC), klare Nachvollziehbarkeit über `--exp`/`--tag`, die Unterstützung manipulierter Datensätze sowie reproduzierbare Visualisierungen.

---

#### `training/test.py`
- Speichert sämtliche Metriken (AUC, ACC) unter `analysis_output/metrics/`, sodass Analyse-Skripte diese automatisiert einlesen können.
- Neue CLI-Argumente:
  - `--exp` für das Experiment (`gen` = Generalisierung, `rob` = Robustheit).
  - `--tag` für den konkreten Test (z. B. `Baseline`, `Within-Domain`, `Cross-Domain`, `JPEG`, `Schwarz-Weiss`, `Text-Overlay`, `Text-Overlay-Augen`, `Gesichtsglaettung`).
- Gibt die Anzahl der verwendeten Bilder/Frames aus (Transparenz bei Train/Test-Zuschnitt).
- Optionale Nutzung der Trainingsmenge als Testmenge (für Baseline-Generalisierung).
- Kompatibilität mit Effort-Gewichten in Kombination mit dem Hilfsskript `convert_effort_ckpt.py`.

---

#### Detektor-Konfigurationen (`training/config/detector/*.yaml`)
- Standardisierte Frames/Video:
  - 8 Frames für FaceForensics++.
  - 4 Frames für DeepFakeDetection.
- Bereinigung/Vereinheitlichung der Test-Configs für reproduzierbare Läufe (keine trainingsspezifischen Optionen in Test-Configs).

---

#### `preprocessing/config.yaml`
- Erweiterung zur Unterstützung manipulierter Datensätze mit Suffixen:
  - `-S_W` (Schwarz/Weiß), `-JPEG`, `-TEXT`, `-FACE` (Face Smoothing).

---

#### `dataset/abstract_dataset.py`
- Übergibt die Bild-Menge passend zum Testlauf an `test.py` (inkl. Support, die Trainingsmenge als Test zu verwenden).
- Behandelt DeepFakeDetection als eigenständigen Datensatz (nicht mehr an FF++ gekoppelt).
- Bugfix: korrekte Übersetzung der JSON-Pfade in LMDB-Schlüssel, um Lookup-Fehler zu vermeiden.

---

#### Analyse-Skripte (`analysis/`)
- `create_tables.py` – Metriken pro Experiment/Test zu Tabellen (CSV) nach `analysis_output/tables/` und bildet zusätzlich den AUC-Durchschnitt pro Detektor ab. Danach wird, falls vorhanden für die Robustheitstests ein Balkendiagramm erstellt, in dem der Effekt zur Baseline dargestellt wird (jeweils für jede Detektor-Datensatz-Kombination).
- `plot_roc.py` – Erzeugt ROC-Kurven (inkl. AUC) je Datensatz/Detektor und schreibt Abbildungen nach `analysis_output/plots/`.
---

#### Hilfsskript `convert_effort_ckpt.py`
- Entfernt inkompatible Prefixe in Effort-Checkpoints, damit diese mit dem Testskript geladen werden können.
- Beispielaufruf (4-Space-Codeblock):
    
    ```
    cd DeepfakeBench
    python convert_effort_ckpt.py training/weights/effort_clip_L14_trainOn_FaceForensic.pth training/weights/effort_clip_L14_trainOn_FaceForensic_stripped.pth
    ```
---

#### Effort-Backbone-Einbindung (`training/detectors/effort_detector.py`)
- In `build_backbone()` wurde die Backbone-Pfadübergabe so angepasst, dass lokale Kopien des CLIP ViT-L/14 (z. B. unter `training/networks/`) geladen werden können.
- README-Hinweis ergänzt, den Pfad zum lokal gespeicherten Backbone nach dem Download anzupassen.

---

#### Namens- und Strukturkonventionen
- Einheitliche Tags für Tests (z. B. `Baseline`, `Within-Domain`, `Cross-Domain`, `JPEG`, `S_W`, `Text`, `Glättung`) zur automatischen Filterung in den Analyse-Skripten.
- Einheitliche Ablagepfade:
  - Metriken: `analysis_output/metrics/`
  - Tabellen: `analysis_output/tables/`
  - Abbildungen: `analysis_output/plots/`

---

#### Kompatibilität & Reproduzierbarkeit
- Konsistente Nutzung von `--exp` und `--tag` in allen Testläufen.
- Einheitliche Frame-Anzahl pro Datensatz (siehe Detektor-Configs).
- Getrennte Verarbeitung und LMDB-Erzeugung für manipulierte Datensätze mit Suffixen, sodass Original- und Manipulationsvarianten parallel ausgewertet werden können.

---

Nun hier ein paar allgemeine Informationen zu DeepfakeBench, übernommen aus der originalen README:

# DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection (NeurIPS 2023 D&B)

<b> Authors: <a href='https://yzy-stack.github.io/'>Zhiyuan Yan</a>, <a href='https://yzhang2016.github.io/'>Yong Zhang</a>, Xinhang Yuan, <a href='https://cse.buffalo.edu/~siweilyu/'>Siwei Lyu</a>, <a href='https://sites.google.com/site/baoyuanwu2015/'>Baoyuan Wu* </a>  </b>

[[paper](https://arxiv.org/abs/2307.01426)] [[pre-trained weights](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1)]

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="figures/archi.png" style="max-width:60%;">
</div>

DeepfakeBench has the following features:

⭐️  **Detectors** (**36** detectors):
  - 5 Naive Detectors: [Xception](./training/detectors/xception_detector.py), [MesoNet](./training/detectors/meso4_detector.py), [MesoInception](./training/detectors/meso4Inception_detector.py), [CNN-Aug](./training/detectors/resnet34_detector.py), [EfficientNet-B4](./training/detectors/efficientnetb4_detector.py)
  - 20 Spatial Detectors: [Capsule](./training/detectors/capsule_net_detector.py), [DSP-FWA](./training/detectors/fwa_detector.py), [Face X-ray](./training/detectors/facexray_detector.py), [FFD](./training/detectors/ffd_detector.py), [CORE](./training/detectors/core_detector.py), [RECCE](./training/detectors/recce_detector.py), [UCF](./training/detectors/ucf_detector.py), [Local-relation](./training/detectors/lrl_detector.py), [IID](./training/detectors/lrl_detector.py), [RFM](./training/detectors/rfm_detector.py), [SIA](./training/detectors/sia_detector.py), [SLADD](./training/detectors/sladd_detector.py), [UIA-ViT](./training/detectors/uia_vit_detector.py), [CLIP](./training/detectors/clip_detector.py), [SBI](./training/detectors/sbi_detector.py), [PCL-I2G](./training/detectors/pcl_xception_detector.py), [Multi-Attention](./training/detectors/multi_attention_detector.py), [LSDA](./training/detectors/lsda_detector.py), [Effort](./training/detectors/effort_detector.py)
  - 3 Frequency Detectors: [F3Net](./training/detectors/f3net_detector.py), [SPSL](./training/detectors/spsl_detector.py), [SRM](./training/detectors/srm_detector.py)
  - 8 Video Detectors: [TALL](./training/detectors/tall_detector.py), [I3D](./training/detectors/i3d_detector.py), [STIL](./training/detectors/stil_detector.py), [FTCN](./training/detectors/ftcn_detector.py), [X-CLIP](./training/detectors/xclip_detector.py), [TimeTransformer](./training/detectors/timesformer_detector.py), [VideoMAE](./training/detectors/videomae_detector.py)


⭐️ **Datasets** (9 datasets): [FaceForensics++](https://github.com/ondyari/FaceForensics), [FaceShifter](https://github.com/ondyari/FaceForensics/tree/master/dataset), [DeepfakeDetection](https://github.com/ondyari/FaceForensics/tree/master/dataset), [Deepfake Detection Challenge (Preview)](https://ai.facebook.com/datasets/dfdc/), [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data), [Celeb-DF-v1](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v1), [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics), [DeepForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master/dataset), [UADFV](https://docs.google.com/forms/d/e/1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/viewform)


### Data

<a href="#top">[Back to top]</a>

All datasets used in DeepfakeBench can be downloaded from their own websites or repositories and preprocessed accordingly.

Other detailed information about the datasets used in DeepfakeBench is summarized below:


| Dataset | Real Videos | Fake Videos | Total Videos | Rights Cleared | Total Subjects | Synthesis Methods | Perturbations | Original Repository |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FaceForensics++ | 1000 | 4000 | 5000 | NO | N/A | 4 | 2 | [Hyper-link](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
| FaceShifter | 1000 | 1000 | 2000 | NO | N/A | 1 | - | [Hyper-link](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
| DeepfakeDetection | 363 | 3000 | 3363 | YES | 28 | 5 | - | [Hyper-link](https://github.com/ondyari/FaceForensics/tree/master/dataset) |
| Deepfake Detection Challenge (Preview) | 1131 | 4119 | 5250 | YES | 66 | 2 | 3 | [Hyper-link](https://ai.facebook.com/datasets/dfdc/) |
| Deepfake Detection Challenge | 23654 | 104500 | 128154 | YES | 960 | 8 | 19 | [Hyper-link](https://www.kaggle.com/c/deepfake-detection-challenge/data) |
| CelebDF-v1 | 408 | 795 | 1203 | NO | N/A | 1 | - | [Hyper-link](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v1) |
| CelebDF-v2 | 590 | 5639 | 6229 | NO | 59 | 1 | - | [Hyper-link](https://github.com/yuezunli/celeb-deepfakeforensics) |
| DeepForensics-1.0 | 50000 | 10000 | 60000 | YES | 100 | 1 | 7 | [Hyper-link](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master/dataset) |
| UADFV | 49 | 49 | 98 | NO | 49 | 1 | - | [Hyper-link](https://docs.google.com/forms/d/e/1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/viewform) |

## 📝 Citation

<a href="#top">[Back to top]</a>

If you find our benchmark useful to your research, please cite it as follows:

```
@inproceedings{DeepfakeBench_YAN_NEURIPS2023,
 author = {Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {4534--4565},
 publisher = {Curran Associates, Inc.},
 title = {DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf},
 volume = {36},
 year = {2023}
}
```

If interested, you can read our recent works about deepfake detection, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).
```
@inproceedings{UCF_YAN_ICCV2023,
 title={Ucf: Uncovering common features for generalizable deepfake detection},
 author={Yan, Zhiyuan and Zhang, Yong and Fan, Yanbo and Wu, Baoyuan},
 booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
 pages={22412--22423},
 year={2023}
}

@inproceedings{LSDA_YAN_CVPR2024,
  title={Transcending forgery specificity with latent space augmentation for generalizable deepfake detection},
  author={Yan, Zhiyuan and Luo, Yuhao and Lyu, Siwei and Liu, Qingshan and Wu, Baoyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@inproceedings{cheng2024can,
  title={Can We Leave Deepfake Data Behind in Training Deepfake Detector?},
  author={Cheng, Jikang and Yan, Zhiyuan and Zhang, Ying and Luo, Yuhao and Wang, Zhongyuan and Li, Chen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}

@article{chen2024textit,
  title={X^2-DFD: A framework for eXplainable and eXtendable Deepfake Detection},
  author={Chen, Yize and Yan, Zhiyuan and Lyu, Siwei and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2410.06126},
  year={2024}
}

@article{cheng2024stacking,
  title={Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection},
  author={Cheng, Jikang and Yan, Zhiyuan and Zhang, Ying and Hao, Li and Ai, Jiaxin and Zou, Qin and Li, Chen and Wang, Zhongyuan},
  journal={arXiv preprint arXiv:2411.11396},
  year={2024}
}

@article{yan2024effort,
  title={Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection},
  author={Yan, Zhiyuan and Wang, Jiangming and Wang, Zhendong and Jin, Peng and Zhang, Ke-Yue and Chen, Shen and Yao, Taiping and Ding, Shouhong and Wu, Baoyuan and Yuan, Li},
  journal={arXiv preprint arXiv:2411.15633},
  year={2024}
}

```


## 🛡️ License

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data (SCLBD) at The School of Data Science (SDS) of The Chinese University of Hong Kong, Shenzhen, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on the research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If you have any suggestions, comments, or wish to contribute code or propose methods, we warmly welcome your input. Please contact us at wubaoyuan@cuhk.edu.cn or yanzhiyuan1114@gmail.com. We look forward to collaborating with you in pushing the boundaries of deepfake detection.



