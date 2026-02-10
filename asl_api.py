# asl_api.py
from flask import Flask, jsonify, request
import time
from collections import deque

import cv2
import torch
import torch.nn.functional as F
import numpy as np

from live_i3d_infer import (
    preprocess_frame_bgr, hwc_to_tchw,
    build_idx2gloss_from_json, load_model,
    NSLT_JSON, WLASL_JSON, CKPT_PATH,
    NUM_CLASSES, CLIP_LEN, CONF_THRESH,
    STABLE_WINS, SILENCE_WINS, temporal_vote,
    MAX_WORDS_ON_SCREEN
)

app = Flask(__name__)

# --- Khởi tạo global ---
print("Loading label map & model ...")
idx2gloss, idx2vid = build_idx2gloss_from_json(NSLT_JSON, WLASL_JSON, NUM_CLASSES)
model = load_model(CKPT_PATH, NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def run_recognition_for_n_seconds(seconds=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return ""

    frames = deque(maxlen=CLIP_LEN)
    win_preds = deque(maxlen=32)
    output_words = []
    last_committed = None
    silent_counter = 0

    end_time = time.time() + seconds

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            break

        arr = preprocess_frame_bgr(frame)
        frames.append(arr)

        if len(frames) == CLIP_LEN:
            with torch.no_grad():
                clip = hwc_to_tchw(list(frames)).to(device)
                logits = model(clip)
                if logits.ndim == 3:
                    logits = logits.mean(dim=2)

                probs = F.softmax(logits, dim=1).squeeze(0)
                conf_tensor, idx_tensor = torch.max(probs, dim=0)
                conf = float(conf_tensor.item())
                idx = int(idx_tensor.item())
                pred_label = idx2gloss[idx]

            win_preds.append((pred_label, conf))

            if conf < CONF_THRESH:
                silent_counter += 1
            else:
                silent_counter = 0

            stable_label, stable_conf = temporal_vote(win_preds, STABLE_WINS)
            if stable_label and stable_conf >= CONF_THRESH:
                if stable_label != last_committed:
                    output_words.append(stable_label)
                    last_committed = stable_label
                    if len(output_words) >= MAX_WORDS_ON_SCREEN:
                        output_words[:] = [output_words[-1]]
                silent_counter = 0

            if silent_counter >= SILENCE_WINS:
                if len(output_words) and output_words[-1] not in [",", ".", "…"]:
                    output_words.append(".")
                last_committed = None
                silent_counter = 0

        # không cần imshow trong phiên bản API

    cap.release()
    # Trả về câu hoàn chỉnh (join tất cả từ)
    sentence = " ".join(output_words)
    return sentence.strip()


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    # Option: nhận 'seconds' từ body nếu muốn
    data = request.get_json(silent=True) or {}
    seconds = int(data.get("seconds", 5))

    text = run_recognition_for_n_seconds(seconds)
    return jsonify({"text": text})


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Chạy API Python trên cổng 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
