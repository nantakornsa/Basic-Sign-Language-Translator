from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import uuid
from collections import Counter, deque
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# The predictor is loaded once; requests are serialized because a MediaPipe Hands
# tracker and the per-session rolling windows are stateful.
model = None
labels: list[str] = []
label_text: dict[str, str] = {}
metadata: dict = {}
hands = None
cv2 = np = tf = None
FEATURES = 126
SEQUENCE_LENGTH = 30
extract_landmarks = open_hands = None
sessions: dict[str, dict] = {}
predict_lock = threading.Lock()
confidence_threshold = 0.70
demo_mode = False
demo_phrases = ["สวัสดี", "ขอความช่วยเหลือ", "ขอบคุณ"]


def initialize(model_path: Path, metadata_path: Path) -> None:
    global model, labels, label_text, metadata, hands
    global cv2, np, tf, FEATURES, SEQUENCE_LENGTH, extract_landmarks, open_hands
    import cv2 as cv2_module
    import numpy as numpy_module
    import tensorflow as tensorflow_module
    from sign_utils import (FEATURES as feature_count, SEQUENCE_LENGTH as sequence_length,
                            extract_landmarks as landmark_extractor, open_hands as hands_factory)
    cv2, np, tf = cv2_module, numpy_module, tensorflow_module
    FEATURES, SEQUENCE_LENGTH = feature_count, sequence_length
    extract_landmarks, open_hands = landmark_extractor, hands_factory
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("ไม่พบโมเดล กรุณาเก็บข้อมูลและรัน train.py ก่อน")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("sequence_length") != SEQUENCE_LENGTH or metadata.get("features") != FEATURES:
        raise ValueError("รูปร่างข้อมูลใน metadata ไม่ตรงกับตัวสกัดจุดสำคัญ")
    model = tf.keras.models.load_model(model_path)
    labels = metadata["classes"]
    label_text = metadata.get("label_text", {name: name for name in labels})
    hands = open_hands()


def predict_frame(image, client_id: str) -> dict:
    rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    state = sessions.setdefault(client_id, {
        "frames": deque(maxlen=SEQUENCE_LENGTH),
        "recent": deque(maxlen=5),
        "previous": None,
        "updated": time.monotonic(),
    })
    state["updated"] = time.monotonic()
    feature = extract_landmarks(results, state["previous"])
    state["previous"] = feature
    state["frames"].append(feature)
    if not results.multi_hand_landmarks:
        state["recent"].clear()
        return {
            "ready": False,
            "text": "ไม่พบมือในภาพ",
            "label": None,
            "confidence": 0.0,
            "frames": len(state["frames"]),
            "sequence_length": SEQUENCE_LENGTH,
        }
    response = {
        "ready": len(state["frames"]) == SEQUENCE_LENGTH,
        "text": "กำลังตรวจจับท่ามือ",
        "label": None,
        "confidence": 0.0,
        "frames": len(state["frames"]),
        "sequence_length": SEQUENCE_LENGTH,
    }
    if not response["ready"]:
        return response

    scores = model(np.expand_dims(np.asarray(state["frames"], dtype=np.float32), 0), training=False).numpy()[0]
    index = int(np.argmax(scores))
    confidence = float(scores[index])
    state["recent"].append(index if confidence >= confidence_threshold else -1)
    response["confidence"] = confidence
    if confidence < confidence_threshold:
        response["text"] = "ไม่พบคำที่มั่นใจ"
        return response
    if len(state["recent"]) == state["recent"].maxlen and Counter(state["recent"])[index] == state["recent"].maxlen:
        label = labels[index]
        response.update({"text": label_text.get(label, label), "label": label})
    else:
        response["text"] = "กำลังยืนยันท่ามือ"
    return response


@app.get("/")
def index():
    return render_template("index.html", sequence_length=SEQUENCE_LENGTH,
                           demo=demo_mode, demo_phrases=demo_phrases)


@app.post("/api/predict")
def api_predict():
    if demo_mode:
        return jsonify({"error": "โหมดสาธิตไม่ประมวลผลหรือจำแนกภาพจริง"}), 409
    payload = request.get_json(silent=True) or {}
    client_id = payload.get("client_id")
    image_data = payload.get("image")
    if not isinstance(client_id, str) or len(client_id) > 80 or not isinstance(image_data, str):
        return jsonify({"error": "ข้อมูลภาพหรือ session ไม่ถูกต้อง"}), 400
    try:
        encoded = image_data.split(",", 1)[-1]
        image_bytes = base64.b64decode(encoded, validate=True)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("อ่านภาพไม่ได้")
        started = time.perf_counter()
        with predict_lock:
            expired = [key for key, state in sessions.items()
                       if time.monotonic() - state["updated"] > 900]
            for key in expired:
                sessions.pop(key, None)
            if len(sessions) >= 64 and client_id not in sessions:
                oldest = min(sessions, key=lambda key: sessions[key]["updated"])
                sessions.pop(oldest, None)
            result = predict_frame(image, client_id)
        result["server_ms"] = (time.perf_counter() - started) * 1000
        return jsonify(result)
    except (ValueError, cv2.error) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/demo")
def api_demo():
    if not demo_mode:
        return jsonify({"error": "เปิดใช้ได้เฉพาะโหมดสาธิต"}), 409
    payload = request.get_json(silent=True) or {}
    phrase = payload.get("phrase")
    if phrase not in demo_phrases:
        return jsonify({"error": "ไม่พบตัวอย่างวลีนี้"}), 400
    return jsonify({"text": phrase, "mode": "demo"})


@app.post("/api/reset")
def api_reset():
    payload = request.get_json(silent=True) or {}
    client_id = payload.get("client_id")
    with predict_lock:
        sessions.pop(client_id, None)
    return jsonify({"ok": True})


def main():
    global confidence_threshold, demo_mode
    parser = argparse.ArgumentParser(description="เว็บแอปรู้จำภาษามือจากกล้อง")
    parser.add_argument("--model", default="models/lstm.keras")
    parser.add_argument("--metadata", default="models/lstm_metadata.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--demo", action="store_true",
                        help="เปิดหน้าเว็บตัวอย่างโดยไม่ต้องมีชุดข้อมูลหรือโมเดลที่ฝึกแล้ว")
    args = parser.parse_args()
    # Keep the selected confidence threshold in one server-side setting.
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold ต้องอยู่ระหว่าง 0 ถึง 1")
    confidence_threshold = args.threshold
    demo_mode = args.demo
    if not demo_mode:
        initialize(Path(args.model), Path(args.metadata))
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        if hands is not None:
            hands.close()


if __name__ == "__main__":
    main()
