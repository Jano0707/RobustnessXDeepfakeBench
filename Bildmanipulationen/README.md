# Bildmanipulationen

Dieser Ordner enthält alle Implementierungen, die zur Erzeugung realitätsnaher Bildmanipulationen auf Kopien der verwendeten Datensätze eingesetzt werden. Ziel ist es, die Robustheit moderner Deepfake-Detektionsmodelle gegenüber typischen, in sozialen Medien vorkommenden Bildveränderungen zu analysieren.

Der Ordner gliedert sich in:

- **`face-manipulations.py`** – Hauptskript für verschiedene Bildmanipulationen.
- **`face-smoothing/`** – Modifiziertes GitHub-Projekt [face-smoothing](https://github.com/5starkarma/face-smoothing) für gezielte Weichzeichnung im Gesicht.

---

## 1. `face-manipulations.py` – Bildmanipulationen mit Albumentations

Das Skript bietet eine Sammlung von Manipulationsfunktionen, die einzeln per CLI ausgeführt werden können. Die Funktionen arbeiten wahlweise auf Einzelbildern oder auf allen Bildern in einem Verzeichnis.

### Implementierte Techniken

- **`black_white`**  
  Konvertiert das Bild in eine Schwarz-Weiß-Version unter Beibehaltung der Dimensionen. Weiterhin bleiben die drei Kanäle bestehen, alle 3 Kanäle enthalten die gleichen Graustufenwerte ([Albumentations-Doku](https://explore.albumentations.ai/transform/ToGray)).

- **`jpeg_compress`**  
  Simuliert Qualitätsverluste durch JPEG-Kompression unter Beibehaltung der Dimensionen(einstellbarer Qualitätsfaktor, default ist 40) ([Albumentations-Doku](https://explore.albumentations.ai/transform/ImageCompression)).

- **`add_text`**  
  Fügt zentrierten Text im unteren Bildbereich hinzu.  
  - Automatischer Zeilenumbruch basierend auf Bildbreite.  
  - Dynamischer Text und Farbe (schwarz/weiß) in Abhängigkeit von der Hintergrundhelligkeit.
  Außerdem kann der Text zentriert auf den Augen, sowie den Augenbrauen hinzugefügt werden. Für die Umsetzung wird DLIB und `shape_predictor_81_face_landmarks.dat` genutzt, um die notwendigen Gesichtsmerkmale zu finden. Durch Tests fiel auf, dass es zum Teil bei kleinen Pixelmaßen Probleme gibt, da die Texte in diesen Fällen nicht den gesamten Bereich abgedeckt haben. Dies würde nicht den gewünschten Effekt erzielen, dass die Detektoren schlechter Artefakte aus diesem Bereich entnehmen können. Entsprechend wird in diesem Fall das Bild auf 1600 x 1600 Pixel vergrößert, der Text wird hinzugefügt und dann auf die Ausgangsmaße zurückgeführt.
  `add_text` behält die Dimensionen bei.

(Optionale, aber aktuell auskommentierte Funktionen für Skalierung und Größenänderung sind ebenfalls enthalten. Zunächst waren sie ebenfalls im Projekt geplant, jedoch erwarten die Detektoren feste Pixelmaße, die durch das Preprocessing erzeugt werden. Diese werden durch die auskommentierten Funktionen geändert.)

---

## 2. `face-smoothing/` – Gesichtsweichzeichnung

Das Unterprojekt basiert auf dem Open-Source-Projekt [face-smoothing](https://github.com/5starkarma/face-smoothing). Es verwendet ein vortrainiertes TensorFlow-Modell zur Gesichtserkennung und wendet anschließend einen **Bilateralfilter** gezielt auf den Gesichtsbereich an. Auch `face-smoothing` arbeitet sowohl auf Einzelbildern, als auch auf Ordnern als Eingabe.

### Änderungen gegenüber dem Originalprojekt

- **Aktualisierung der verwendeten Abhängigkeiten**  
  Veraltete Abhängigkeiten wurden in `face-smoothing/requirements.txt` auf aktuelle Versionen gebracht.
- **Einheitlicher Bild-Output**  
  Ausgaberoutinen so angepasst, dass Bilder unter dem gleichen Dateinamen im angegebenen Zielordner gespeichert werden.
- **RGB-Output**  
  Anpassungen, um sicherzustellen, dass Ausgaben im **RGB-Format** gespeichert werden, da OpenCV standardmäßig im BGR-Format arbeitet. Außerdem arbeitet face-smoothing nun rekursiv, sodass ein Ordner mit Ordnern als Input funktioniert. Zusätzlich wurden Anpassungen vorgenommen, sodass negative Koordinaten des erkannten Gesichts nicht in einem Abbruch enden.
  Änderungen vorgenommen in:
  - `face-smoothing/utils/image.py`: Funktionen `load_image()` und `save_image()`  
  - `face-smoothing/infer.py`: Funktion `save_image()`

---

## 3. Verwendung

Für die Anwendung der Bildmanipulationen wird die Nutzung der erstellten Conda-Umgebung empfohlen, da bereits für die Conda-Umgebung, DeepfakeBench, alle verwendeten Imports vorhanden sind.

### Schwarz-Weiß-Umwandlung
```
python face-manipulations.py --function=black_white --input=./Beispiel-Bilder --output=./Beispiel-Outputs
```

### JPEG-Kompression
```
python face-manipulations.py --function=jpeg --input=./Beispiel-Bilder --output=./Beispiel-Outputs --quality=50
```

### Text-Overlay
```
python face-manipulations.py --function=add_text --input=./Beispiel-Bilder --output=./Beispiel-Outputs --text="Hallo, ich bin ein Untertitel"
```

### Text-Overlay Augen
```
python face-manipulations.py --function=add_text --input=./Beispiel-Bilder --output=./Beispiel-Outputs --text="Hallo, ich bin ein Untertitel" --place="eyes"
```

### Face-Smoothing
```
cd face-smoothing
python infer.py --input './Beispiel-Bilder' --output './Beispiel-Outputs'
```
