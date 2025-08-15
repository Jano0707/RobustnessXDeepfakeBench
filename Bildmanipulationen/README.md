# Bildmanipulationen

Dieser Ordner enthält alle Implementierungen, die zur Erzeugung realitätsnaher Bildmanipulationen auf Kopien der verwendeten Datensätze eingesetzt werden. Ziel ist es, die Robustheit moderner Deepfake-Detektionsmodelle gegenüber typischen, in sozialen Medien vorkommenden Bildveränderungen zu analysieren.

Der Ordner gliedert sich in:

- **`manipulations.py`** – Hauptskript für verschiedene Bildmanipulationen.
- **`face-smoothing/`** – Modifiziertes GitHub-Projekt [face-smoothing](https://github.com/5starkarma/face-smoothing) für gezielte Weichzeichnung im Gesicht.

---

## 1. `manipulations.py` – Bildmanipulationen mit Albumentations

Das Skript bietet eine Sammlung von Manipulationsfunktionen, die einzeln per CLI ausgeführt werden können. Die Funktionen arbeiten wahlweise auf Einzelbildern oder auf allen Bildern in einem Verzeichnis.

### Implementierte Techniken

- **`black_white`**  
  Konvertiert das Bild in eine Schwarz-Weiß-Version unter Beibehaltung der Dimensionen. Weiterhin bleiben die drei Kanäle bestehen, alle 3 Kanäle enthalten die gleichen Graustufenwerte ([Albumentations-Doku](https://explore.albumentations.ai/transform/ToGray)).

- **`rotate_90_left`**  
  Dreht das Bild um exakt 90° gegen den Uhrzeigersinn unter Beibehaltung der Dimensionen([Albumentations-Doku](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/rotate/#Rotate)).

- **`jpeg_compress`**  
  Simuliert Qualitätsverluste durch JPEG-Kompression unter Beibehaltung der Dimensionen(einstellbarer Qualitätsfaktor, default ist 40) ([Albumentations-Doku](https://explore.albumentations.ai/transform/ImageCompression)).

- **`add_text`**  
  Fügt zentrierten Text im unteren Bildbereich hinzu.  
  - Automatischer Zeilenumbruch basierend auf Bildbreite.  
  - Dynamischer Text und Farbe (schwarz/weiß) in Abhängigkeit von der Hintergrundhelligkeit.
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
  Anpassungen, um sicherzustellen, dass Ausgaben im **RGB-Format** gespeichert werden, da OpenCV standardmäßig im BGR-Format arbeitet.  
  Änderungen vorgenommen in:
  - `face-smoothing/utils/image.py`: Funktionen `load_image()` und `save_image()`  
  - `face-smoothing/infer.py`: Funktion `save_image()`

---

## 3. Verwendung

### Schwarz-Weiß-Umwandlung
```
python face-manipulations.py --function=black_white --input=./Beispiel-Bilder --output=./Beispiel-Outputs
```

### Rotation um 90 Grad gegen den Uhrzeigersinn
```
python face-manipulations.py --function=black_white --input=./Beispiel-Bilder --output=./Beispiel-Outputs
```

### JPEG-Kompression
```
python face-manipulations.py --function=black_white --input=./Beispiel-Bilder --output=./Beispiel-Outputs --quality=40
```

### Text-Overlay
```
python face-manipulations.py --function=add_text --input=./Beispiel-Bilder --output=./Beispiel-Outputs --text="Hallo ich bin ein Untertitel"
```

### Face-Smoothing
```
python face-smoothing/infer.py --input './Beispiel-Bilder' --output './Beispiel-Outputs'
```