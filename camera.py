import cv2
import time
import numpy as np
import logging
from threading import Thread, Lock
from PIL import Image
from config import FRAME_SKIP, RECOG_INTERVAL
from database import load_database, add_person, delete_person, list_people, get_people
from embeddings import INPUT_SIZE, mtcnn, get_embedding_from_crop, recognize
from preprocessing import align_face, crop_face

# ─── Scale factor for detection (lower = faster, less accurate) ───
DETECT_SCALE = 0.5        # process detection at half resolution
USE_FAST_TRACKER = True    # use MOSSE instead of KCF (much faster)
MIN_DETECTION_CONFIDENCE = 0.9

logger = logging.getLogger(__name__)


def create_tracker():
    """Create the fastest available OpenCV tracker for the installed build."""
    tracker_factories = (
        ("legacy.TrackerMOSSE_create", lambda: cv2.legacy.TrackerMOSSE_create()),
        ("legacy.TrackerKCF_create", lambda: cv2.legacy.TrackerKCF_create()),
        ("TrackerKCF_create", lambda: cv2.TrackerKCF_create()),
    )
    if not USE_FAST_TRACKER:
        tracker_factories = tracker_factories[1:]

    for factory_name, factory in tracker_factories:
        try:
            return factory()
        except AttributeError:
            logger.debug("OpenCV tracker factory unavailable: %s", factory_name)

    logger.error("No supported OpenCV tracker factory is available. Install opencv-contrib-python.")
    return None


def enroll_face(pil_img, database):
    name = input("Enter name to enroll: ").strip()
    if not name:
        print("Cancelled.")
        return

    boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
    if boxes is None:
        print("No face detected!")
        return

    best_idx = int(np.argmax(probs))
    if probs[best_idx] < 0.9:
        print(f"Confidence too low: {probs[best_idx]:.2f}")
        return

    x1, y1, x2, y2 = map(int, boxes[best_idx])
    frame_rgb = np.asarray(pil_img)
    aligned = align_face(frame_rgb, landmarks[best_idx], INPUT_SIZE)
    face_crop = aligned.image if aligned is not None else pil_img.crop((x1, y1, x2, y2))
    emb = get_embedding_from_crop(face_crop)

    if emb is not None:
        add_person(database, name, emb)
    else:
        print("Could not compute embedding.")


def remove_face(database):
    """Interactive deletion of a person from the database."""
    list_people(database)
    if not get_people(database):
        return
    name = input("Enter name to delete: ").strip()
    if not name:
        print("Cancelled.")
        return
    delete_person(database, name)


class AsyncDetector:
    """Run MTCNN detection + embedding in a background thread."""

    def __init__(self):
        self._lock = Lock()
        self._frame_rgb = None
        self._frame_bgr = None
        self._ready = False
        self._result_boxes = []   # list of (x1,y1,x2,y2)
        self._result_labels = []  # list of (label_str, color)
        self._busy = False

    def submit(self, frame_bgr, frame_rgb):
        """Submit a new frame for detection (non-blocking)."""
        with self._lock:
            if self._busy:
                return  # skip if still processing
            self._busy = True
            self._frame_bgr = frame_bgr.copy()
            self._frame_rgb = frame_rgb.copy()
        Thread(target=self._detect, daemon=True).start()

    def _detect(self):
        """Background detection thread."""
        try:
            rgb = self._frame_rgb
            h, w = self._frame_bgr.shape[:2]

            # Downscale for faster detection
            small_w = int(w * DETECT_SCALE)
            small_h = int(h * DETECT_SCALE)
            small_rgb = cv2.resize(rgb, (small_w, small_h))
            pil_small = Image.fromarray(small_rgb)

            boxes, probs, landmarks = mtcnn.detect(pil_small, landmarks=True)

            det_boxes = []
            det_labels = []

            if boxes is not None:
                # Scale boxes back to original resolution
                scale_x = w / small_w
                scale_y = h / small_h
                pil_full = Image.fromarray(rgb)

                for box, prob, points in zip(boxes, probs, landmarks):
                    if prob < MIN_DETECTION_CONFIDENCE:
                        continue
                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)

                    # clamp
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue

                    det_boxes.append((x1, y1, x2, y2))

                    scaled_points = np.asarray(points, dtype=np.float32).copy()
                    scaled_points[:, 0] *= scale_x
                    scaled_points[:, 1] *= scale_y
                    aligned = align_face(rgb, scaled_points, INPUT_SIZE)
                    face_crop = aligned.image if aligned is not None else pil_full.crop((x1, y1, x2, y2))
                    emb = get_embedding_from_crop(face_crop)
                    if emb is not None:
                        det_labels.append(emb)
                    else:
                        det_labels.append(None)

            with self._lock:
                self._result_boxes = det_boxes
                self._result_labels = det_labels
                self._ready = True

        finally:
            self._busy = False

    def get_results(self):
        """Fetch the latest results (thread-safe). Returns None if not ready."""
        with self._lock:
            if not self._ready:
                return None
            self._ready = False
            return self._result_boxes, self._result_labels


def run():
    database    = load_database()
    cap         = cv2.VideoCapture(0)
    frame_count = 0
    smoothed_fps = 0.0
    fps_alpha    = 0.1
    prev_time    = time.perf_counter()

    # Current tracked state
    trackers    = []
    labels      = []

    detector = AsyncDetector()

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print("Controls:  E = enroll  |  D = delete  |  L = list DB  |  Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.perf_counter()
        frame_time = now - prev_time
        prev_time = now

        if frame_time > 0:
            instant_fps = 1.0 / frame_time
            if smoothed_fps == 0.0:
                smoothed_fps = instant_fps
            else:
                smoothed_fps = (fps_alpha * instant_fps) + ((1.0 - fps_alpha) * smoothed_fps)

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Submit detection job every N frames ─────────────────
        if frame_count % FRAME_SKIP == 0:
            detector.submit(frame, rgb)

        # ── Check for detection results (non-blocking) ──────────
        result = detector.get_results()
        if result is not None:
            det_boxes, det_embs = result
            trackers = []
            labels = []

            for (x1, y1, x2, y2), emb in zip(det_boxes, det_embs):
                w, h = x2 - x1, y2 - y1

                tracker = create_tracker()
                if tracker is None:
                    continue

                tracker.init(frame, (x1, y1, w, h))
                trackers.append(tracker)

                if emb is not None and database:
                    name, score = recognize(emb, database)
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    labels.append((f"{name}  {score:.2f}", color))
                else:
                    labels.append(("Detecting...", (0, 255, 255)))

        # ── Tracking (very fast) ────────────────────────────────
        valid_trackers, valid_labels = [], []

        for i, tracker in enumerate(trackers):
            success, box = tracker.update(frame)
            if not success or i >= len(labels):
                continue

            x, y, w, h = map(int, box)

            # Re-recognize periodically (in main thread, but only on crop)
            if frame_count % RECOG_INTERVAL == 0 and database:
                fallback_crop = crop_face(rgb, (x, y, x + w, y + h), INPUT_SIZE)
                pil_crop = fallback_crop.image if fallback_crop is not None else None
                if pil_crop is not None:
                    emb = get_embedding_from_crop(pil_crop)
                    if emb is not None:
                        name, score = recognize(emb, database)
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        labels[i] = (f"{name}  {score:.2f}", color)

            label, color = labels[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            valid_trackers.append(tracker)
            valid_labels.append(labels[i])

        trackers, labels = valid_trackers, valid_labels

        # ── FPS ─────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {smoothed_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"People in DB: {len(get_people(database))}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Energy Efficient Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):
            pil_img = Image.fromarray(rgb)
            enroll_face(pil_img, database)
        elif key == ord('d'):
            remove_face(database)
        elif key == ord('l'):
            list_people(database)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
