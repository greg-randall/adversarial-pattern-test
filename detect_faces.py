#!/usr/bin/env python3
"""Check whether a face is still detectable after an adversarial pattern, makeup,
or camouflage has been applied.

Runs four independent, modern face detectors on an image (or a folder of images)
and draws a semi-transparent, colour-coded overlay showing what each one found:

    - MediaPipe    (Google, on-device, tuned for precision)
    - RetinaFace   (CVPR 2020, detection + landmarks + 3D)
    - SCRFD        (InsightFace, top of the WIDER Face benchmark)
    - YOLOv11-face (community-trained, tuned for recall)

Why four detectors instead of one: an adversarial pattern that defeats a single
model (or a single vendor's product) tells you almost nothing about whether it
defeats face detection in general. Different detectors use different
architectures and training data, so agreement across all four is much stronger
evidence than a pass against any one of them.

IMPORTANT: this checks DETECTION only (does a bounding box get drawn around the
face at all). It does not check RECOGNITION (would a face-matching system
still identify *whose* face it is). A pattern that survives detection could
still fail against recognition, and vice versa -- they are different attacks.

Usage:
    python3 detect_faces.py                  # process every image in the current folder
    python3 detect_faces.py photo.jpg         # process a single image
    python3 detect_faces.py path/to/folder    # process every image in a folder

Output: ``{stem}_overlay{ext}`` written next to each input image.

First run downloads model weights for YOLOv11-face, RetinaFace, and InsightFace
(a few hundred MB total) -- this only happens once, weights are cached locally.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np


# --- Configuration -----------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
OUTPUT_SUFFIX = "_overlay"

# (name, colour-BGR) -- colour order chosen for contrast: blue, orange, green, red
DETECTOR_DEFS = [
    ("mediapipe",  (0, 165, 255)),
    ("retinaface", (255, 100, 0)),
    ("scrfd",      (0, 220, 0)),
    ("yolo",       (0, 0, 255)),
]

ALPHA = 0.45        # transparency of filled boxes (0 = invisible, 1 = solid)
LINE_ALPHA = 0.85   # box outline is more opaque for visibility
LEGEND_HEIGHT = 50
FONT = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# Detector builders -- each returns detect(img_bgr) -> [(x1, y1, x2, y2, conf), ...]
# ---------------------------------------------------------------------------

def build_mediapipe():
    import mediapipe as mp
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    def detect(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = fd.process(rgb)
        faces = []
        if results.detections:
            h, w = img_bgr.shape[:2]
            for d in results.detections:
                bb = d.location_data.relative_bounding_box
                faces.append((
                    max(0, int(bb.xmin * w)),
                    max(0, int(bb.ymin * h)),
                    min(w - 1, int((bb.xmin + bb.width) * w)),
                    min(h - 1, int((bb.ymin + bb.height) * h)),
                    float(d.score[0]),
                ))
        return faces
    return detect


def build_retinaface():
    """Must be initialised before MediaPipe -- its Keras model conflicts with
    MediaPipe's TF Lite backend otherwise."""
    from retinaface import RetinaFace

    def detect(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        result = RetinaFace.detect_faces(rgb)
        faces = []
        if isinstance(result, dict):
            for _key, val in result.items():
                if isinstance(val, dict) and "facial_area" in val:
                    x1, y1, x2, y2 = val["facial_area"]
                    score = float(val.get("score", 0.999))
                    faces.append((
                        max(0, int(x1)), max(0, int(y1)),
                        min(w - 1, int(x2)), min(h - 1, int(y2)),
                        score,
                    ))
        return faces
    return detect


def build_scrfd():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=-1)  # -1 = CPU, 0+ = GPU

    def detect(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        detections = app.get(rgb)
        faces = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox.astype(int).tolist()
            conf = float(d.det_score)
            faces.append((max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2), conf))
        return faces
    return detect


def build_yolo():
    from ultralytics import YOLO

    model_url = (
        "https://github.com/akanametov/yolo-face/releases/download/1.0.0/"
        "yolov11n-face.pt"
    )
    model_path = Path.home() / ".cache" / "face-detect" / "yolov11n-face.pt"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading YOLOv11-face model to {model_path} ...")
        urllib.request.urlretrieve(model_url, str(model_path))
        print("  Download complete.")

    model = YOLO(str(model_path))

    def detect(img_bgr):
        h, w = img_bgr.shape[:2]
        results = model(img_bgr, verbose=False)
        faces = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                faces.append((max(0, int(x1)), max(0, int(y1)),
                              min(w - 1, int(x2)), min(h - 1, int(y2)), conf))
        return faces
    return detect


BUILDERS = {
    "mediapipe": build_mediapipe,
    "retinaface": build_retinaface,
    "scrfd": build_scrfd,
    "yolo": build_yolo,
}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_overlay(img_bgr, all_faces, legend_entries):
    """Draw semi-transparent boxes for every model + a legend bar at the bottom.

    *all_faces* is a list of (name, colour, [(x1,y1,x2,y2,conf), ...]) tuples.
    Returns the composite BGR image (same size as input + legend).
    """
    h, w = img_bgr.shape[:2]

    overlay_fill = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_line = np.zeros((h, w, 3), dtype=np.uint8)

    for _name, colour, faces in all_faces:
        for (x1, y1, x2, y2, _conf) in faces:
            cv2.rectangle(overlay_fill, (x1, y1), (x2, y2), colour, -1)
            cv2.rectangle(overlay_line, (x1, y1), (x2, y2), colour, 2)

    img = img_bgr.copy().astype(np.float32)

    fill_mask = overlay_fill.any(axis=2)
    if fill_mask.any():
        blended = img[fill_mask] * (1 - ALPHA) + overlay_fill[fill_mask].astype(np.float32) * ALPHA
        img[fill_mask] = blended

    line_mask = overlay_line.any(axis=2)
    if line_mask.any():
        blended = img[line_mask] * (1 - LINE_ALPHA) + overlay_line[line_mask].astype(np.float32) * LINE_ALPHA
        img[line_mask] = blended

    img = img.clip(0, 255).astype(np.uint8)

    legend = np.full((LEGEND_HEIGHT, w, 3), 30, dtype=np.uint8)  # dark grey

    n = len(legend_entries)
    cell_w = w // max(n, 1)
    text_x0 = 46
    max_text_w = max(cell_w - text_x0 - 4, 10)  # stay clear of the next swatch
    for i, (name, colour) in enumerate(legend_entries):
        x0 = i * cell_w
        cv2.rectangle(legend, (x0 + 8, 8), (x0 + 36, LEGEND_HEIGHT - 8), colour, -1)
        cv2.rectangle(legend, (x0 + 8, 8), (x0 + 36, LEGEND_HEIGHT - 8), (255, 255, 255), 1)

        font_scale = 0.55
        (text_w, _text_h), _ = cv2.getTextSize(name, FONT, font_scale, 1)
        while text_w > max_text_w and font_scale > 0.3:
            font_scale -= 0.05
            (text_w, _text_h), _ = cv2.getTextSize(name, FONT, font_scale, 1)

        cv2.putText(legend, name, (x0 + text_x0, LEGEND_HEIGHT // 2 + 6),
                    FONT, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([img, legend])


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def is_own_output(path: Path) -> bool:
    return OUTPUT_SUFFIX in path.stem


def find_image_files(target: Path) -> list[Path]:
    """Return image files to process: a single file, or every image in a folder
    (skipping this script's own ``_overlay`` outputs)."""
    if target.is_file():
        return [target] if target.suffix.lower() in IMAGE_EXTENSIONS else []

    images = []
    for entry in sorted(target.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if is_own_output(entry):
            continue
        images.append(entry)
    return images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "path", nargs="?", default=".",
        help="Image file or folder of images (default: current folder)")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: {target} does not exist.")
        sys.exit(1)

    images = find_image_files(target)
    if not images:
        print("No image files found.")
        sys.exit(0)

    print(f"Found {len(images)} image(s).\n")

    # RetinaFace must load first -- its Keras model conflicts with MediaPipe's
    # TF Lite backend if MediaPipe loads first.
    init_sequence = sorted(
        DETECTOR_DEFS, key=lambda x: (0 if x[0] == "retinaface" else 1, x[0]))

    print("Initialising detectors (first run downloads model weights) ...")
    detectors = []       # (name, colour, detect_fn)
    legend_entries = []  # (name, colour) -- only for models that loaded
    for name, colour in init_sequence:
        try:
            fn = BUILDERS[name]()
            detectors.append((name, colour, fn))
            legend_entries.append((name, colour))
            print(f"  {name} ready.")
        except Exception as exc:
            print(f"  [SKIP] {name}: {exc}")

    if not detectors:
        print("No detectors available. Check your installation (see requirements.txt).")
        sys.exit(1)

    print(f"\nColours: {'  |  '.join(n for n, _ in legend_entries)}")
    print()

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[SKIP] Could not read: {img_path.name}")
            continue
        if img_bgr.ndim >= 3 and img_bgr.shape[2] == 4:
            img_bgr = img_bgr[:, :, :3]

        print(f"Processing: {img_path.name}  ({img_bgr.shape[1]}x{img_bgr.shape[0]})")

        all_faces = []
        for name, colour, detect_fn in detectors:
            try:
                faces = detect_fn(img_bgr)
                all_faces.append((name, colour, faces))
                print(f"  {name:14s}: {len(faces)} face(s)")
            except Exception as exc:
                print(f"  {name:14s}: ERROR - {exc}")

        face_counts = {name: len(faces) for name, _colour, faces in all_faces}
        counted_legend = [
            (f"{name}: {face_counts[name]}" if name in face_counts else name, colour)
            for name, colour in legend_entries
        ]

        out_img = draw_overlay(img_bgr, all_faces, counted_legend)
        out_path = img_path.parent / f"{img_path.stem}{OUTPUT_SUFFIX}{img_path.suffix}"
        cv2.imwrite(str(out_path), out_img)
        print(f"  -> {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
