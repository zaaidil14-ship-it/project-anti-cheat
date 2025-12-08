# ============================================================
# server.py — FINAL (1 tombol ON/OFF + PATH FIX + list evidence)
# ============================================================

import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, abort, Response

import cv2 as cv
import mediapipe as mp
import numpy as np
import re
from difflib import SequenceMatcher
from ultralytics import YOLO
import torch.serialization
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

# -----------------------
# CONFIG
# -----------------------
CHEAT_COOLDOWN = 6
FORBIDDEN_COOLDOWN = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
CHEAT_IMG_DIR = os.path.join(EVIDENCE_DIR, "cheat", "images")
FORBIDDEN_IMG_DIR = os.path.join(EVIDENCE_DIR, "forbidden", "images")
FORBIDDEN_JSON_DIR = os.path.join(EVIDENCE_DIR, "forbidden", "json")
LOG_FILE = os.path.join(EVIDENCE_DIR, "cheat_log.json")

FORBIDDEN_WORDS = ["tolong", "jawab", "bocor", "jawaban", "cheat", "bantu", "google"]

yolo_model = YOLO("yolov8n.pt")
target_classes = ["book", "cell phone"]

# Ensure folders exist
for d in [EVIDENCE_DIR, CHEAT_IMG_DIR, FORBIDDEN_IMG_DIR, FORBIDDEN_JSON_DIR]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

# -----------------------
# GLOBALS
# -----------------------
app = Flask(__name__, static_folder=BASE_DIR)

log_lock = threading.Lock()
cheat_capture_lock = threading.Lock()
forbidden_capture_lock = threading.Lock()
metadata_lock = threading.Lock()

last_frame = None
cheat_cooldown_until = 0.0
forbidden_cooldown_until = 0.0

latest_metadata = {
    "timestamp": None,
    "alerts": [],
    "gaze_status": "-",
    "face_boxes": [],
    "yolo_boxes": []
}

system_enabled = True

book_detected_since = None
phone_detected_since = None

# -----------------------
# MEDIAPIPE
# -----------------------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_detection = mp_face_detection.FaceDetection(0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
pose = mp_pose.Pose(model_complexity=0)

# -----------------------
# UTILITIES
# -----------------------
def normalize_text(t):
    t = t.lower()
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def detect_forbidden(text):
    text = normalize_text(text)
    if not text:
        return []
    words = text.split()
    detected = []
    for fw in FORBIDDEN_WORDS:
        if fw in words or fw in text:
            detected.append(fw)
            continue
        for w in words:
            if similar(w, fw) >= 0.75:
                detected.append(fw)
                break
    return list(set(detected))

def append_log(entry):
    with log_lock:
        logs = json.load(open(LOG_FILE, "r", encoding="utf-8"))
        logs.append(entry)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

def save_screenshot(frame, forbidden=False):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if forbidden:
        path = os.path.join(FORBIDDEN_IMG_DIR, f"fb_{ts}.jpg")
    else:
        path = os.path.join(CHEAT_IMG_DIR, f"ch_{ts}.jpg")
    cv.imwrite(path, frame, [cv.IMWRITE_JPEG_QUALITY, 55])
    return path

# -----------------------
# YOLO DETECTION
# -----------------------
def detect_objects(frame):
    results = yolo_model(frame, verbose=False)[0]
    objects = []
    for box in results.boxes:
        cls_name = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if conf < 0.15 or cls_name not in target_classes:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        objects.append({"class": cls_name, "conf": conf, "box": [x1, y1, x2, y2]})
    return objects

# -----------------------
# EVIDENCE
# -----------------------
def capture_cheat_evidence(event, frame, gaze, faces, pose_detected, alerts, boxes, yolo_objects):
    global cheat_cooldown_until
    with cheat_capture_lock:
        img = save_screenshot(frame)
        rel = os.path.relpath(img, BASE_DIR).replace("\\", "/")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "cheat",
            "event": event,
            "alerts": alerts,
            "gaze_status": gaze,
            "face_count": faces,
            "pose_detected": pose_detected,
            "face_boxes": boxes,
            "yolo_objects": yolo_objects,
            "image_file": rel
        }
        append_log(entry)
        cheat_cooldown_until = time.time() + CHEAT_COOLDOWN

def capture_forbidden_evidence(words, transcript, frame):
    global forbidden_cooldown_until
    with forbidden_capture_lock:
        img = save_screenshot(frame, forbidden=True)
        rel = os.path.relpath(img, BASE_DIR).replace("\\", "/")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "forbidden",
            "event": "Forbidden Speech",
            "forbidden_words": words,
            "transcript": transcript,
            "image_file": rel
        }
        append_log(entry)
        json_path = os.path.join(FORBIDDEN_JSON_DIR, f"forbidden_{int(time.time())}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        forbidden_cooldown_until = time.time() + FORBIDDEN_COOLDOWN

# -----------------------
# GAZE
# -----------------------
def estimasi_pandangan(face_landmarks, w, h):
    try:
        nose = face_landmarks.landmark[1]
        left = face_landmarks.landmark[33]
        right = face_landmarks.landmark[263]
        nx = nose.x * w
        lx = left.x * w
        rx = right.x * w
        center = (lx + rx) / 2
        diff = nx - center
        if diff > w*0.02: return "Menoleh kiri"
        if diff < -w*0.02: return "Menoleh kanan"
        return "Menghadap Depan"
    except:
        return "Unknown"

# -----------------------
# CAMERA LOOP
# -----------------------
def open_camera():
    for backend in [cv.CAP_DSHOW, cv.CAP_MSMF, cv.CAP_ANY]:
        cap = cv.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
    return None

def anti_cheat_loop():
    global last_frame, latest_metadata
    global book_detected_since, phone_detected_since
    global system_enabled

    cap = open_camera()
    face_miss_start = None
    gaze_away_start = None
    frame_count = 0

    while True:
        if cap is None or not cap.isOpened():
            cap = open_camera()
            time.sleep(1)
            continue

        ok, frame = cap.read()
        if not ok:
            continue

        last_frame = frame.copy()
        h,w,_ = frame.shape
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # ✅ SYSTEM OFF
        if not system_enabled:
            with metadata_lock:
                latest_metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "alerts": ["SYSTEM OFF"],
                    "gaze_status": "-",
                    "face_boxes": [],
                    "yolo_boxes": []
                }
            time.sleep(0.2)
            continue

        # FACE DETECTION
        fd = face_detection.process(rgb)
        faces = len(fd.detections) if fd.detections else 0

        boxes = []
        if fd and fd.detections:
            for det in fd.detections:
                box = det.location_data.relative_bounding_box
                boxes.append({
                    "xmin": box.xmin,
                    "ymin": box.ymin,
                    "width": box.width,
                    "height": box.height
                })

        fm = face_mesh.process(rgb) if faces > 0 else None

        frame_count += 1
        pr = pose.process(rgb) if frame_count % 3 == 0 else None

        alerts = []
        now = time.monotonic()

        # Face missing
        if faces == 0:
            if face_miss_start is None:
                face_miss_start = now
            elif now - face_miss_start > 2:
                alerts.append("Face Missing >2s")
        else:
            face_miss_start = None

        # Gaze
        gaze = "Unknown"
        if fm and fm.multi_face_landmarks:
            gaze = estimasi_pandangan(fm.multi_face_landmarks[0], w, h)
            if gaze != "Menghadap Depan":
                if gaze_away_start is None:
                    gaze_away_start = now
                elif now - gaze_away_start > 3:
                    alerts.append("Looking Away >3s")
            else:
                gaze_away_start = None

        # YOLO
        objects = detect_objects(frame)

        # BOOK
        if any(o["class"] == "book" for o in objects):
            if book_detected_since is None:
                book_detected_since = now
            elif now - book_detected_since > 3:
                alerts.append("Book Detected >3s")
        else:
            book_detected_since = None

        # PHONE
        if any(o["class"] == "cell phone" for o in objects):
            if phone_detected_since is None:
                phone_detected_since = now
            elif now - phone_detected_since > 3:
                alerts.append("Phone Detected >3s")
        else:
            phone_detected_since = None

        # UPDATE METADATA
        with metadata_lock:
            latest_metadata = {
                "timestamp": datetime.now().isoformat(),
                "alerts": alerts,
                "gaze_status": gaze,
                "face_boxes": boxes,
                "yolo_boxes": objects
            }

        # CHEAT EVIDENCE
        if alerts and time.time() >= cheat_cooldown_until:
            threading.Thread(
                target=capture_cheat_evidence,
                args=(" & ".join(alerts), frame.copy(), gaze, faces, pr is not None, alerts, boxes, objects),
                daemon=True
            ).start()

        time.sleep(0.015)

# -----------------------
# API
# -----------------------
@app.route("/api/speech", methods=["POST"])
def api_speech():
    global last_frame, forbidden_cooldown_until, system_enabled

    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text or not system_enabled:
        return jsonify({"status": "ignored"})

    forbidden_hits = detect_forbidden(text)

    if forbidden_hits and time.time() >= forbidden_cooldown_until:
        if last_frame is not None:
            threading.Thread(
                target=capture_forbidden_evidence,
                args=(forbidden_hits, text, last_frame.copy()),
                daemon=True
            ).start()

    return jsonify({"status": "ok", "forbidden": forbidden_hits})

@app.route("/api/log")
def api_log():
    logs = json.load(open(LOG_FILE,"r",encoding="utf-8"))
    logs = sorted(logs, key=lambda x: x["timestamp"], reverse=True)
    return jsonify(logs)

@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    global last_frame
    if last_frame is None:
        return jsonify({"error":"no frame"}), 500
    threading.Thread(
        target=capture_cheat_evidence,
        args=("Manual Snapshot", last_frame.copy(), "Unknown", 0, False, ["Manual Snapshot"], [], []),
        daemon=True
    ).start()
    return jsonify({"status":"ok"})

@app.route("/api/live-metadata")
def api_meta():
    return jsonify(latest_metadata)

# ✅ LIST FILES IN EVIDENCE FOLDER
@app.route("/api/list-evidence")
def api_list_evidence():
    cheat_imgs = [f"evidence/cheat/images/{f}" for f in os.listdir(CHEAT_IMG_DIR) if f.endswith(".jpg")]
    forbidden_imgs = [f"evidence/forbidden/images/{f}" for f in os.listdir(FORBIDDEN_IMG_DIR) if f.endswith(".jpg")]
    forbidden_json = [f"evidence/forbidden/json/{f}" for f in os.listdir(FORBIDDEN_JSON_DIR) if f.endswith(".json")]

    return jsonify({
        "cheat_images": cheat_imgs,
        "forbidden_images": forbidden_imgs,
        "forbidden_json": forbidden_json
    })

# ✅ ONLY ONE TOGGLE
@app.route("/api/toggle", methods=["POST"])
def api_toggle():
    global system_enabled
    data = request.get_json() or {}
    system_enabled = bool(data.get("value", True))
    return jsonify({"status": "ok", "system": system_enabled})

# ✅ FIXED PATH SERVING
@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    return send_from_directory(EVIDENCE_DIR, filename)

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "viewer.html")

@app.route("/video")
def video():
    def gen():
        global last_frame
        while True:
            frame = last_frame
            if frame is None:
                frame = np.zeros((480,640,3),dtype=np.uint8)
            ok,jpg = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# -----------------------
# START
# -----------------------
if __name__ == "__main__":
    threading.Thread(target=anti_cheat_loop, daemon=True).start()
    print("Server running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
