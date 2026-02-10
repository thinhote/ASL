from flask import Flask, Response, jsonify
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
    STABLE_WINS, SILENCE_WINS,
    temporal_vote, MAX_WORDS_ON_SCREEN
)

app = Flask(__name__)

#Khởi tạo model, label map
print("Loading label map & model ...")
idx2gloss, idx2vid = build_idx2gloss_from_json(NSLT_JSON, WLASL_JSON, NUM_CLASSES)
model = load_model(CKPT_PATH, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Biến global để lưu text hiện tại
current_text = ""

def generate_frames():
    global current_text

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        while True:
            # nếu ko mở được webcam, vẫn yield frame đen cho khỏi treo
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    frames = deque(maxlen=CLIP_LEN)
    win_preds = deque(maxlen=32)
    output_words = []
    last_committed = None
    silent_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuẩn hoá frame
        arr = preprocess_frame_bgr(frame)
        frames.append(arr)

        # Khi đủ CLIP_LEN khung thì suy luận
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

        # Xây câu hiện tại để hiển thị
        if len(output_words) >= MAX_WORDS_ON_SCREEN:
            cur_text = " ".join(output_words)
            clear_after = True
        else:
            cur_text = " ".join(output_words)
            clear_after = False

        # cập nhật biến global cho endpoint /current_text
        current_text = cur_text

        # Vẽ overlay lên frame
        # cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        # cv2.putText(frame, f"ASL words: {cur_text}", (10, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Encode thành JPEG và stream ra
        ret2, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if clear_after:
            output_words.clear()
            last_committed = None

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/current_text')
def get_current_text():
    return jsonify({"text": current_text})


# Cho phép CORS đơn giản để JS trên port 8080 gọi được
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "http://localhost:8080"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


if __name__ == '__main__':
    # chạy Flask trên port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
