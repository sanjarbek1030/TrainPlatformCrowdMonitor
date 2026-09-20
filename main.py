"""
Train Platform Crowd Monitor - main.py
========================================
ONE file, does everything:

  1) AUTOMATIC PLATFORM DETECTION: on the video's first frame, uses classical
     computer vision (Canny edge detection + Hough line transform) to find
     the platform's straight edges and automatically build 3 zone polygons
     from them. Note: there's no dedicated pretrained "train platform"
     object detector, so this is a solid geometric heuristic, not a learned
     model -- it works well when platform edges/tracks are clearly visible
     as straight lines, and may need a manual touch-up on unusual angles.
  2) A preview window shows you the auto-detected zones. Press 'y' to
     accept them, or 'n' to instead freehand-draw (click and drag the
     mouse) your own zone outlines -- OpenCV will detect and simplify your
     drawn line into a clean polygon automatically either way.
  3) Saves whichever zones you end up with to 'zones_config.json' so you
     don't have to repeat calibration on your next run (you'll be asked
     whether to reuse them or redo it).
  4) Runs YOLOv8 person detection on the full video, counts people inside
     each zone frame-by-frame, and saves a fully annotated output video.

Controls in the automatic-detection preview:
    'y'  -> accept the automatically detected zones
    'n'  -> switch to manual freehand drawing instead

Controls while freehand drawing (only if you pressed 'n' above):
    Left-click + DRAG  -> sketch the outline of the current zone
    ENTER or 'c'       -> confirm this zone's shape, move to the next zone
    'r'                -> clear/redo the current zone's shape
    'q'                -> stop drawing early (any un-drawn zones fall back
                          to a simple default rectangle so the script can
                          still run)

Requirements:
    pip install ultralytics opencv-python numpy
"""

import os
import sys
import json
import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------
INPUT_VIDEO_PATH = "platform_video.mp4"      # source video to analyze
OUTPUT_VIDEO_PATH = "crowd_output.mp4"       # annotated video that gets saved
ZONES_CONFIG_PATH = "zones_config.json"      # where your drawn zones are cached
MODEL_WEIGHTS = "yolov8n.pt"                 # lightweight YOLOv8 model
PERSON_CLASS_ID = 0                          # COCO class 0 = "person"
CONFIDENCE_THRESHOLD = 0.4                   # minimum detection confidence

ZONE_NAMES = ["Left Section", "Center Section", "Right Section"]
ZONE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red (BGR)


def default_zones(frame_width, frame_height):
    """
    A simple fallback (equal vertical thirds) used ONLY if you press 'q'
    and skip drawing a zone. This just keeps the script from crashing --
    drawing your own zones is the whole point of this tool.
    """
    third = frame_width // 3
    return {
        "Left Section": [(0, 0), (third, 0), (third, frame_height), (0, frame_height)],
        "Center Section": [(third, 0), (2 * third, 0), (2 * third, frame_height), (third, frame_height)],
        "Right Section": [(2 * third, 0), (frame_width, 0), (frame_width, frame_height), (2 * third, frame_height)],
    }


# ------------------------------------------------------------------------
# 2. FREEHAND DRAWING + AUTOMATIC LINE DETECTION
# ------------------------------------------------------------------------
class ZoneDrawer:
    """
    Lets you sketch each zone's outline with your mouse. Every point along
    your drag is recorded as you draw. When you confirm a zone (ENTER/'c'),
    cv2.approxPolyDP automatically "detects" the shape of your freehand
    stroke and simplifies it into a clean polygon with just a handful of
    corner points -- this is the automatic line detection step.
    """

    def __init__(self, base_frame):
        self.base_frame = base_frame
        self.frame_height, self.frame_width = base_frame.shape[:2]
        self.current_zone_index = 0
        self.raw_points = []       # points recorded along the current stroke
        self.finished_zones = {}   # zone_name -> detected polygon
        self.drawing = False

    def mouse_callback(self, event, x, y, flags, param):
        """Records mouse movement while the left button is held down."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.raw_points = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Only keep a new point if the mouse moved a bit, so we don't
            # store thousands of near-identical points
            last = self.raw_points[-1] if self.raw_points else None
            if last is None or np.hypot(x - last[0], y - last[1]) > 2:
                self.raw_points.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def detect_polygon_from_stroke(self):
        """
        Automatically detects a clean polygon from the raw freehand stroke
        using contour approximation (cv2.approxPolyDP). This is the "line
        detection" step: it takes your messy hand-drawn line and simplifies
        it down to its essential corner points.
        """
        if len(self.raw_points) < 3:
            return None

        contour = np.array(self.raw_points, dtype=np.int32).reshape(-1, 1, 2)
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.01 * perimeter  # smaller = closer to your exact line
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return [tuple(point[0]) for point in approx]

    def render(self):
        """Draws everything confirmed so far, plus the stroke in progress."""
        display = self.base_frame.copy()

        # Draw zones that have already been confirmed
        for i, name in enumerate(ZONE_NAMES):
            if name in self.finished_zones:
                pts = np.array(self.finished_zones[name], dtype=np.int32)
                cv2.polylines(display, [pts], True, ZONE_COLORS[i], 2)

        # Draw the stroke currently being sketched, in the active zone's color
        if len(self.raw_points) > 1:
            pts = np.array(self.raw_points, dtype=np.int32)
            cv2.polylines(display, [pts], False, ZONE_COLORS[self.current_zone_index], 2)

        # Instruction bar at the bottom of the window
        active_name = ZONE_NAMES[self.current_zone_index] if self.current_zone_index < len(ZONE_NAMES) else "Done"
        instructions = f"Draw '{active_name}'  |  ENTER/c = confirm   r = redo   q = skip/quit"
        cv2.rectangle(display, (0, self.frame_height - 40), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        cv2.putText(display, instructions, (10, self.frame_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return display

    def run(self):
        """Main interactive loop: shows the window and reacts to key presses."""
        window_name = "Draw your platform zones (freehand) - see instructions at bottom"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while self.current_zone_index < len(ZONE_NAMES):
            cv2.imshow(window_name, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key in (13, ord('c')):  # ENTER or 'c' -> confirm this zone
                polygon = self.detect_polygon_from_stroke()
                if polygon is None:
                    print("[WARN] Draw a shape with at least 3 points before confirming.")
                    continue
                zone_name = ZONE_NAMES[self.current_zone_index]
                self.finished_zones[zone_name] = polygon
                print(f"[INFO] Detected polygon for '{zone_name}': {polygon}")
                self.current_zone_index += 1
                self.raw_points = []

            elif key == ord('r'):  # redo the current zone
                self.raw_points = []

            elif key == ord('q'):  # quit drawing early
                print("[INFO] Drawing stopped early by user.")
                break

        cv2.destroyWindow(window_name)
        return self.finished_zones


def extrapolate_line(x1, y1, x2, y2, y_target):
    """
    Given a line segment (x1,y1)-(x2,y2), extends it and returns the x
    position where it would cross a given y (height) value. This lets us
    stretch short detected edge fragments into full platform-length lines.
    """
    if y1 == y2:
        return x1  # perfectly horizontal, no meaningful extrapolation
    t = (y_target - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def find_line_candidates(frame):
    """
    AUTOMATIC PLATFORM EDGE DETECTION - step 1: classical edge + line finding.

    There's no off-the-shelf AI model trained specifically to recognize
    "train platform edges", so instead we use standard computer vision:
      1. Canny edge detection to find all sharp edges in the image.
      2. A probabilistic Hough Transform to find long straight line segments
         among those edges (platform edges, curbs, and track rails are all
         long straight lines in a station photo).
      3. Filter out near-horizontal lines (roof beams, ceiling trusses,
         floor tile grout) since platform boundaries are usually diagonal
         due to camera perspective.
    Returns a list of candidate line dicts, each extrapolated to where it
    would cross the top and bottom of the frame.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Ignore the very top of the frame (ceiling/roof/sky) -- platform edges
    # won't be found there, and it cuts down on false positives
    top_ignore_y = int(h * 0.2)
    mask = np.zeros_like(edges)
    mask[top_ignore_y:h, :] = 255
    edges = cv2.bitwise_and(edges, mask)

    min_line_length = int(w * 0.15)
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                 minLineLength=min_line_length, maxLineGap=25)
    if raw_lines is None:
        return []

    top_y = h * 0.3      # the "far" reference line, near the vanishing point
    bottom_y = h - 1      # the "near" reference line, at the bottom of the frame

    candidates = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Skip near-horizontal lines -- platform edges appear diagonal
        # because of the camera's perspective, not flat/horizontal
        if abs(angle) < 12 or abs(180 - abs(angle)) < 12:
            continue

        x_top = extrapolate_line(x1, y1, x2, y2, top_y)
        x_bottom = extrapolate_line(x1, y1, x2, y2, bottom_y)

        # Discard wildly extrapolated lines that shoot far outside the frame
        if -0.5 * w <= x_top <= 1.5 * w and -0.5 * w <= x_bottom <= 1.5 * w:
            candidates.append({"x_top": x_top, "x_bottom": x_bottom, "length": length})

    return candidates


def cluster_lines(candidates, frame_width, cluster_gap_fraction=0.06):
    """
    AUTOMATIC PLATFORM EDGE DETECTION - step 2: merge fragmented lines.

    A single real platform edge is often detected as several broken line
    fragments. This groups candidate lines that land close to each other
    (based on their position at the bottom of the frame) into one merged
    "boundary", weighted by each fragment's length so longer, more
    confident detections count more.
    """
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["x_bottom"])
    gap = frame_width * cluster_gap_fraction

    groups = []
    current_group = [candidates[0]]
    for c in candidates[1:]:
        if c["x_bottom"] - current_group[-1]["x_bottom"] <= gap:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]
    groups.append(current_group)

    merged = []
    for group in groups:
        total_length = sum(g["length"] for g in group)
        x_top = sum(g["x_top"] * g["length"] for g in group) / total_length
        x_bottom = sum(g["x_bottom"] * g["length"] for g in group) / total_length
        merged.append({"x_top": x_top, "x_bottom": x_bottom, "support": total_length})

    return merged


def build_boundaries(clusters, frame_width):
    """
    AUTOMATIC PLATFORM EDGE DETECTION - step 3: pick exactly 4 boundary
    lines (which define 3 zones between them: Left | Center | Right).

    Uses the leftmost and rightmost detected clusters as the outer platform
    edges, and the two strongest interior clusters (most supporting line
    length) as the dividers between sections. If not enough real edges were
    detected, missing boundaries are filled in evenly so the script always
    has something reasonable to work with.
    """
    def clamp(x):
        return max(0.0, min(float(frame_width), x))

    if len(clusters) == 0:
        # No usable lines detected at all -- fall back to even thirds
        return [
            {"x_top": 0, "x_bottom": 0},
            {"x_top": frame_width / 3, "x_bottom": frame_width / 3},
            {"x_top": 2 * frame_width / 3, "x_bottom": 2 * frame_width / 3},
            {"x_top": frame_width, "x_bottom": frame_width},
        ]

    left_edge = clusters[0]
    right_edge = clusters[-1]
    interior = clusters[1:-1]

    chosen_interior = sorted(interior, key=lambda c: c["support"], reverse=True)[:2]
    chosen_interior.sort(key=lambda c: c["x_bottom"])

    # If we don't have 2 real interior dividers, synthesize evenly-spaced ones
    missing = 2 - len(chosen_interior)
    if missing > 0:
        fractions = [1 / 3, 2 / 3][:missing] if missing == 2 else [0.5]
        for frac in fractions:
            x_top = left_edge["x_top"] + frac * (right_edge["x_top"] - left_edge["x_top"])
            x_bottom = left_edge["x_bottom"] + frac * (right_edge["x_bottom"] - left_edge["x_bottom"])
            chosen_interior.append({"x_top": x_top, "x_bottom": x_bottom, "support": 0})
        chosen_interior.sort(key=lambda c: c["x_bottom"])

    boundaries = [left_edge] + chosen_interior + [right_edge]
    for b in boundaries:
        b["x_top"] = clamp(b["x_top"])
        b["x_bottom"] = clamp(b["x_bottom"])
    return boundaries


def zones_from_boundaries(boundaries, top_y, bottom_y):
    """Turns 4 boundary lines into 3 trapezoid zone polygons between them."""
    zones = {}
    for i, name in enumerate(ZONE_NAMES):
        left = boundaries[i]
        right = boundaries[i + 1]
        polygon = [
            (int(left["x_top"]), int(top_y)),
            (int(right["x_top"]), int(top_y)),
            (int(right["x_bottom"]), int(bottom_y)),
            (int(left["x_bottom"]), int(bottom_y)),
        ]
        zones[name] = polygon
    return zones


def auto_detect_zones(base_frame):
    """
    Runs the full automatic platform-edge-detection pipeline on a frame and
    returns the resulting 3 zone polygons, plus the merged boundary lines
    (for optional visualization/debugging).
    """
    frame_height, frame_width = base_frame.shape[:2]
    candidates = find_line_candidates(base_frame)
    clusters = cluster_lines(candidates, frame_width)
    boundaries = build_boundaries(clusters, frame_width)

    top_y = frame_height * 0.3
    bottom_y = frame_height - 1
    zones = zones_from_boundaries(boundaries, top_y, bottom_y)
    return zones, boundaries


def run_auto_or_manual_calibration(base_frame):
    """
    Tries automatic platform detection first and shows you a preview. Since
    classical line-detection is a heuristic (not a trained "platform"
    model), it won't be perfect on every camera angle -- so you get the
    final say: accept the automatic zones, or fall back to freehand drawing.
    """
    frame_height, frame_width = base_frame.shape[:2]
    colors_by_name = {name: ZONE_COLORS[i] for i, name in enumerate(ZONE_NAMES)}

    print("[INFO] Running automatic platform edge detection...")
    zones, boundaries = auto_detect_zones(base_frame)

    preview = base_frame.copy()
    # Show the detected boundary lines in yellow so you can see what was found
    for b in boundaries:
        pt_top = (int(b["x_top"]), int(frame_height * 0.3))
        pt_bottom = (int(b["x_bottom"]), frame_height - 1)
        cv2.line(preview, pt_top, pt_bottom, (0, 255, 255), 2)

    preview = draw_zones(preview, zones, colors_by_name)
    cv2.rectangle(preview, (0, frame_height - 40), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.putText(preview,
                "Auto-detected zones from platform edges.  y = accept   n = draw manually instead",
                (10, frame_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    window_name = "Automatic Platform Detection - review result"
    cv2.imshow(window_name, preview)
    print("[INFO] Preview window opened. Press 'y' to accept, or 'n' to draw zones by hand instead.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            cv2.destroyWindow(window_name)
            print("[INFO] Automatic zones accepted.")
            return zones
        elif key == ord('n'):
            cv2.destroyWindow(window_name)
            print("[INFO] Switching to manual freehand drawing...")
            drawer = ZoneDrawer(base_frame)
            manual_zones = drawer.run()
            fallback = default_zones(frame_width, frame_height)
            for name in ZONE_NAMES:
                if name not in manual_zones:
                    print(f"[WARN] '{name}' was not drawn -- using a default rectangle instead.")
                    manual_zones[name] = fallback[name]
            return manual_zones


def load_or_create_zones(base_frame):
    """
    Reuses zones saved from a previous run if you want, otherwise launches
    the freehand drawing tool above and saves the result for next time.
    """
    frame_height, frame_width = base_frame.shape[:2]

    if os.path.isfile(ZONES_CONFIG_PATH):
        choice = input(f"Found saved zones in '{ZONES_CONFIG_PATH}'. Reuse them? (y/n): ").strip().lower()
        if choice != "n":
            with open(ZONES_CONFIG_PATH, "r") as f:
                saved = json.load(f)
            # JSON stores plain lists -- convert the coordinate pairs back to tuples
            return {name: [tuple(point) for point in pts] for name, pts in saved.items()}

    zones = run_auto_or_manual_calibration(base_frame)

    with open(ZONES_CONFIG_PATH, "w") as f:
        json.dump(zones, f, indent=2)
    print(f"[INFO] Zones saved to '{ZONES_CONFIG_PATH}' for future runs.")

    return zones


# ------------------------------------------------------------------------
# 3. ZONE COUNTING + DRAWING HELPERS (used during full video processing)
# ------------------------------------------------------------------------
def point_in_zone(point, polygon):
    """
    Checks whether a point sits inside a zone polygon using
    cv2.pointPolygonTest (>= 0 means inside or on the edge).
    """
    polygon_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(polygon_np, point, False) >= 0


def draw_zones(frame, zones, colors_by_name, alpha=0.25):
    """Draws a translucent color fill + solid outline for every zone."""
    overlay = frame.copy()
    for name, polygon in zones.items():
        pts = np.array(polygon, dtype=np.int32)
        color = colors_by_name[name]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_headcount_panel(frame, zone_counts, colors_by_name):
    """Draws a live headcount panel in the top-left corner of the frame."""
    panel_x, panel_y = 15, 30
    line_height = 28
    total = sum(zone_counts.values())

    box_h = line_height * (len(zone_counts) + 1) + 15
    box_w = 260
    sub_img = frame[10:10 + box_h, 10:10 + box_w]
    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    frame[10:10 + box_h, 10:10 + box_w] = cv2.addWeighted(sub_img, 0.4, black_rect, 0.6, 0)

    for i, (name, count) in enumerate(zone_counts.items()):
        color = colors_by_name[name]
        y = panel_y + i * line_height
        cv2.putText(frame, f"{name}: {count}", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    y_total = panel_y + len(zone_counts) * line_height
    cv2.putText(frame, f"Total People: {total}", (panel_x, y_total),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ------------------------------------------------------------------------
# 4. MAIN PIPELINE
# ------------------------------------------------------------------------
def main():
    # --- Step 1: Make sure the input video actually exists ---
    if not os.path.isfile(INPUT_VIDEO_PATH):
        print(f"[ERROR] Input video not found: '{INPUT_VIDEO_PATH}'")
        sys.exit(1)

    # --- Step 2: Open the video ---
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: '{INPUT_VIDEO_PATH}'. "
              f"It may be corrupted or use an unsupported codec.")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback if the file's metadata is missing FPS
    print(f"[INFO] Video: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    # --- Step 3: Grab the first frame so you can draw zones on a real image ---
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read the first frame of the video.")
        cap.release()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind so we still process frame 1 later

    # --- Step 4: Get your zones (freehand drawing, or reused from last time) ---
    zones = load_or_create_zones(first_frame)
    colors_by_name = {name: ZONE_COLORS[i] for i, name in enumerate(ZONE_NAMES)}

    # --- Step 5: Load the YOLOv8 model ---
    try:
        model = YOLO(MODEL_WEIGHTS)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model '{MODEL_WEIGHTS}': {e}")
        cap.release()
        sys.exit(1)

    # --- Step 6: Set up the output video writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"[ERROR] Failed to create output video writer for '{OUTPUT_VIDEO_PATH}'.")
        cap.release()
        sys.exit(1)

    frame_index = 0

    try:
        # --- Step 7: Process the video frame-by-frame ---
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video
            frame_index += 1

            zone_counts = {name: 0 for name in zones.keys()}

            # Run YOLOv8, restricted to "person" only, with a clean console
            results = model.predict(source=frame, classes=[PERSON_CLASS_ID],
                                     conf=CONFIDENCE_THRESHOLD, verbose=False)

            frame = draw_zones(frame, zones, colors_by_name)

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # Use the bottom-center ("foot") point for zone assignment --
                    # more accurate than the box center since feet touch the floor
                    foot_point = (int((x1 + x2) / 2), int(y2))

                    for name, polygon in zones.items():
                        if point_in_zone(foot_point, polygon):
                            zone_counts[name] += 1
                            break

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"person {conf:.2f}", (x1, max(y1 - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, foot_point, 3, (0, 0, 255), -1)

            frame = draw_headcount_panel(frame, zone_counts, colors_by_name)
            out.write(frame)

            if frame_index % 50 == 0:
                print(f"[INFO] Processed {frame_index} frames...")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Finishing up and releasing resources...")

    finally:
        # --- Step 8: Always release resources, even if something went wrong ---
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Done. Processed {frame_index} total frames.")
        print(f"[INFO] Annotated video saved to: '{OUTPUT_VIDEO_PATH}'")


if __name__ == "__main__":
    main()
