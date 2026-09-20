# 🚉 Train Platform Crowd Monitor

Automatically detect train platform sections, count people on each one in real time, and export a fully annotated video — all from a single Python script.

Built with **OpenCV** + **Ultralytics YOLOv8**.

---

## ✨ Features

- **Automatic platform detection** — uses Canny edge detection + Hough line transform to find the platform's straight edges and auto-generate 3 zone polygons (Left / Center / Right Section), no manual coordinates needed.
- **Freehand fallback** — if the automatic detection doesn't match your camera angle, reject it with one keypress and sketch your own zone outlines with the mouse. OpenCV auto-simplifies your hand-drawn line into a clean polygon.
- **Reusable calibration** — zones are cached to `zones_config.json`, so you only calibrate once per camera setup.
- **Person detection & counting** — YOLOv8 (`yolov8n.pt`) detects people frame-by-frame; each person is assigned to a zone using their foot position for accuracy.
- **Live annotated output** — bounding boxes, translucent zone overlays, and a real-time per-zone + total headcount panel, saved to a new video file.

---

## 📦 Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy

Install dependencies:

```bash
pip install ultralytics opencv-python numpy
```

> The first run will automatically download `yolov8n.pt` (~6 MB) if it isn't already present.

---

## 📁 Project Structure

```
.
├── main.py              # Everything: calibration, detection, counting, video export
├── platform_video.mp4   # Your input video (you provide this)
├── zones_config.json    # Auto-generated after your first calibration
└── crowd_output.mp4     # Auto-generated annotated output video
```

---

## 🚀 Usage

1. Place your source video in the project folder and name it `platform_video.mp4`
   (or edit `INPUT_VIDEO_PATH` in `main.py`).
2. Run the script:

   ```bash
   python main.py
   ```

3. **Calibrate your zones** (first run only):
   - A preview window shows platform edges detected automatically, with the resulting Left/Center/Right zones drawn on top.
   - Press **`y`** to accept the automatic zones, or **`n`** to draw your own instead.
4. If you chose to draw manually:
   - **Click and drag** to sketch the outline of the current zone.
   - **`ENTER`** or **`c`** — confirm the zone and move to the next one.
   - **`r`** — clear and redo the current zone.
   - **`q`** — stop early (any un-drawn zones fall back to a simple default rectangle).
5. Sit back — the script processes the full video and writes `crowd_output.mp4` with:
   - Bounding boxes around every detected person
   - Semi-transparent colored zone overlays
   - A live headcount panel (per zone + total) in the top-left corner

On your **next run**, you'll be asked whether to reuse the saved zones from `zones_config.json` or recalibrate from scratch.

---

## ⚙️ Configuration

All key settings live at the top of `main.py`:

| Variable | Description | Default |
|---|---|---|
| `INPUT_VIDEO_PATH` | Source video file | `"platform_video.mp4"` |
| `OUTPUT_VIDEO_PATH` | Annotated output video | `"crowd_output.mp4"` |
| `ZONES_CONFIG_PATH` | Cached zone calibration file | `"zones_config.json"` |
| `MODEL_WEIGHTS` | YOLOv8 model weights | `"yolov8n.pt"` |
| `PERSON_CLASS_ID` | COCO class ID to detect | `0` (person) |
| `CONFIDENCE_THRESHOLD` | Minimum detection confidence | `0.4` |

To force a fresh calibration, delete `zones_config.json` and rerun the script.

---

## 🧠 How It Works

1. **Zone calibration** — The first frame is run through Canny edge detection and a probabilistic Hough transform to find long, mostly-diagonal lines (platform edges, track rails). Nearby line fragments are merged, the strongest boundary lines are selected, and the space between them is converted into 3 trapezoid zone polygons that follow the platform's real perspective.
2. **Detection** — Each frame is passed to YOLOv8, filtered to the `person` class only.
3. **Zone assignment** — Each detected person's *foot point* (bottom-center of their bounding box) is tested against each zone polygon with `cv2.pointPolygonTest`, since feet on the platform floor are a more accurate signal than a box's center.
4. **Annotation & export** — Boxes, zone overlays, and the headcount panel are drawn on every frame, which is written to the output video with matching resolution and FPS.

---

## ⚠️ Limitations

- There is no pretrained model that recognizes "train platform" as an object class — automatic detection is a **geometric heuristic** (straight-line detection), not a learned platform segmentation model. It works best when platform edges are clearly visible, well-lit, straight lines.
- Detection accuracy depends on camera angle, lighting, and occlusion — use the freehand fallback for tricky setups.
- Designed for a fixed camera; zones are calibrated once and assumed static for the whole video.

---

## 📄 License

Add your preferred license here (e.g. MIT).
