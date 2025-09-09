import cv2
import numpy as np

"""
def get_roi(detected_img, bboxes, box_num):
"""
"""
    Crop detected image to size of detection

    Parameters
    ----------
    detected_img : np.array [H,W,3]
        BGR image
    bboxes : 
"""
"""
    return detected_img[bboxes[box_num][1]:bboxes[box_num][3], 
                        bboxes[box_num][0]:bboxes[box_num][2]]
"""

def _to_xyxy(box):
    """Erkenne (x1,y1,x2,y2) vs. (x,y,w,h) automatisch und gib (x1,y1,x2,y2) zurück."""
    x, y, a, b = map(int, box)
    # Falls a<=x oder b<=y, interpretieren wir (x,y,w,h)
    if a <= x or b <= y:
        return x, y, x + a, y + b
    return x, y, a, b

def _clip_xyxy(x1, y1, x2, y2, w, h):
    """Schneide BBox an Bildgrenzen und verwerfe leere Flächen."""
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def get_roi(detected_img, bboxes, box_num):
    """
    Liefert ein sicheres ROI und die geclippten Koordinaten.
    Rückgabe:
      roi_img (np.ndarray oder None), (x1,y1,x2,y2) oder None
    """
    h, w = detected_img.shape[:2]
    box = bboxes[box_num]
    x1, y1, x2, y2 = _to_xyxy(box)
    clipped = _clip_xyxy(x1, y1, x2, y2, w, h)
    if clipped is None:
        return None, None
    x1, y1, x2, y2 = clipped
    roi = detected_img[y1:y2, x1:x2]
    if roi.size == 0 or roi.ndim != 3 or roi.shape[2] != 3:
        return None, None
    return roi, (x1, y1, x2, y2)

def smooth_face(cfg, detected_img, bboxes):
    """
    Smooth faces in an image using bilateral filtering.

    Parameters
    ----------
    cfg : dict
        Dictionary of configurations
    box_face : np.array [H,W,3]
        BGR image
    bboxes : list
        List of detected bounding boxes

    Returns
    -------
    detected_img : np.array [H,W,3]
        BGR image with face detections
    roi : np.array [H,W,3]
        BGR image
    full_mask : np.array [H,W,3]
        BGR image
    full_img : np.array [H,W,3]
        BGR image
    """
    output_img = detected_img.copy()

    last_roi = None
    last_mask = None
    last_smoothed = None

    for i in range(len(bboxes)):
        print(f'Face detected: {bboxes[i]}')
        roi_img, coords = get_roi(detected_img, bboxes, i)
        if roi_img is None or coords is None:
            print(f"[skip] Ungültiges ROI für Box {bboxes[i]}")
            continue

        x1, y1, x2, y2 = coords
        temp_img = roi_img.copy()

        # Robuste Farbraumumwandlung (nur auf nicht-leeres ROI)
        try:
            hsv_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        except cv2.error as e:
            print(f"[skip] cvtColor-Fehler bei Box {bboxes[i]}: {e}")
            continue

        # HSV-Maske
        low  = np.array(cfg['image']['hsv_low'],  dtype=np.uint8)
        high = np.array(cfg['image']['hsv_high'], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv_img, low, high)
        full_mask = cv2.merge([hsv_mask, hsv_mask, hsv_mask])

        # Bilateral-Filter
        d  = int(cfg['filter']['diameter'])
        s1 = float(cfg['filter']['sigma_1'])
        s2 = float(cfg['filter']['sigma_2'])
        blurred_img = cv2.bilateralFilter(roi_img, d, s1, s2)

        # Zusammensetzen: (Original außerhalb Maske) + (Blur innerhalb Maske)
        masked_blur = cv2.bitwise_and(blurred_img, full_mask)
        inv_mask    = cv2.bitwise_not(full_mask)
        masked_orig = cv2.bitwise_and(temp_img, inv_mask)
        smoothed_roi = cv2.add(masked_orig, masked_blur)

        # Zurück ins Gesamtbild
        output_img[y1:y2, x1:x2] = smoothed_roi

        last_roi = roi_img
        last_mask = full_mask
        last_smoothed = smoothed_roi

        #print(f"-> verwendet: Box {bboxes[i]} -> geclippt {coords}")

    # Falls kein Gesicht erfolgreich bearbeitet wurde, gib None für die letzten Artefakte zurück
    if last_roi is None:
        return output_img, None, None, None
    return output_img, last_roi, last_mask, last_smoothed
