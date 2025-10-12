# RobustnessXDeepfakeBench

**RobustnessXDeepfakeBench** ist eine Erweiterung von [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) zur Untersuchung der Robustheit moderner Deepfake-Detektionsmodelle gegenüber realitätsnahen Bildmanipulationen.  
Das Projekt ist Teil der Bachelorarbeit *„Deepfake-Detektion im digitalen Zeitalter: Eine Analyse der Robustheit gegenüber realitätsnahen Bildmanipulationen“*.

## Aufbau des Projekts

Das Repository ist in zwei Hauptordner unterteilt:

- **`/DeepfakeBench`**  
  Enthält das modifizierte DeepfakeBench-Projekt.  
  Im Unterordner [`/DeepfakeBench/README.md`](DeepfakeBench/README.md) sind alle vorgenommenen Änderungen sowie Installations- und Nutzungshinweise dokumentiert.

- **`/Bildmanipulationen`**  
  Enthält die Implementierungen der Bildmanipulationen, die auf Kopien der verwendeten Datensätze angewendet werden.  
  Dazu zählen unter anderem:
  - JPEG-Kompression
  - Schwarz-Weiß-Umwandlung
  - Gesichtsglättung (basierend auf dem face-smoothing Projekt)
  - Hinzufügen von Text-Overlays   
Details zu den einzelnen Manipulationen und zu den Änderungen am Originalprojekt *face-smoothing* finden sich in [`/Bildmanipulationen/README.md`](Bildmanipulationen/README.md).

## 🚀 Inbetriebnahme

1. Anaconda-Download: https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh 
2. git clone https://github.com/Jano0707/RobustnessXDeepfakeBench.git
3. cd DeepfakeBench
4. ./install.sh 
$\rightarrow$ falls Probleme mit dlib auftauchen: conda install -c conda-forge dlib=19.24.0=py38*
5. conda create -n DFBench
6. conda activate DFBench (muss sowohl für alle Schritte in DeepfakeBench, als auch für die Bildmanipulationen aktiv sein)

## Drittanbieter-Software und Lizenzen

Dieses Projekt basiert auf und nutzt unter anderem folgende Open-Source-Software:

- **DeepfakeBench** – [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/)  
  Copyright (c) 2023, CUHK(SZ)  
  Modifiziert im Rahmen dieses Projekts, siehe `/DeepfakeBench/README.md`.

- **face-smoothing**  
  Originalprojekt: GitHub [5starkarma/face-smoothing](https://github.com/5starkarma/face-smoothing)  
  Modifiziert im Rahmen dieses Projekts für RGB-Ausgabe und Erhalt der Original-Bildmaße, siehe `/Bildmanipulationen/README.md`.

- **OpenCV** – [Apache License](https://github.com/opencv/opencv/blob/master/LICENSE)  
  Copyright (c) 2014-2023, OpenCV team

- **Albumentations** – [MIT License](https://github.com/albumentations-team/albumentations/blob/main/LICENSE)  
  Copyright (c) 2018, Alexandr Buslaev, Vladimir Iglovikov, Alexander Parinov

- **NumPy** – [BSD License](https://github.com/numpy/numpy/blob/main/LICENSE.txt)  
  Copyright (c) 2005-2023, NumPy Developers

- **Python Standard Library** – [Python Software Foundation License](https://docs.python.org/3/license.html)  
  Copyright (c) Python Software Foundation

Alle oben genannten Bibliotheken und Projekte werden ausschließlich zu nicht-kommerziellen Forschungszwecken im Rahmen der Bachelorarbeit *„Deepfake-Detektion im digitalen Zeitalter: Eine Analyse der Robustheit gegenüber realitätsnahen Bildmanipulationen“* verwendet.
