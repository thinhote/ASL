import os
import json
import time
from collections import deque, Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# === 1) Import I3D từ repo ===
from pytorch_i3d import InceptionI3d as I3D  # tên lớp trong repo này

# ====== Cấu hình bạn cần chỉnh ======
CKPT_PATH   = "archived/asl1000/FINAL_nslt_1000_iters=5104_top1=47.33_top5=76.44_top10=84.33.pt"  # checkpoint 2000 từ
# LABELS_JSON = "preprocess/labels_2000.json"   # file labels_2000.json đã build từ WLASL_v0.3
NSLT_JSON   = "preprocess/nslt_1000.json"       # mapping video_id -> action[class_id,...]
WLASL_JSON  = "../../start_kit/WLASL_v0.3.json"           # file WLASL gốc (gloss + instances)
NUM_CLASSES = 1000                            # đúng với checkpoint
MODE        = "rgb"                           # dùng RGB
IMG_SIZE    = 224
CLIP_LEN    = 64                              # số khung/clip
STRIDE      = 8                               # (chưa dùng nhiều, có thể để vậy)
CONF_THRESH = 0.55                            # ngưỡng tin cậy để “chốt” từ
STABLE_WINS = 3                               # cần ổn định qua 3 cửa sổ
SILENCE_WINS= 12                              # nếu im đủ lâu -> chèn dấu ngắt
MAX_WORDS_ON_SCREEN = 1

# ====== Chuẩn hoá ảnh theo repo (img/255 * 2 - 1) ======
def preprocess_frame_bgr(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    arr = (frame_rgb.astype(np.float32) / 255.0) * 2.0 - 1.0  # [H,W,C]
    return arr  # HWC

def hwc_to_tchw(frames_hwc):
    # frames_hwc: [T, H, W, C] -> torch [1, C, T, H, W]
    arr = np.asarray(frames_hwc, dtype=np.float32)  # T,H,W,C
    arr = np.transpose(arr, (3, 0, 1, 2))           # C,T,H,W
    tensor = torch.from_numpy(arr).unsqueeze(0)     # 1,C,T,H,W
    return tensor

#ĐỌC LABELS từ labels_2000.json
def build_label_map(json_path):
    """
    Đọc file labels_2000.json: list [gloss_0, gloss_1, ..., gloss_1999]
    Trả về list idx2gloss sao cho idx2gloss[class_id] = GLOSS (chữ in hoa).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        labels = json.load(f)  # list
    labels = [str(x).upper() for x in labels]
    return labels

def build_idx2gloss_from_json(nslt_path, wlasl_path, num_classes):
    #B1: class_id -> list video_id từ nslt_2000
    with open(nslt_path, "r", encoding="utf-8") as f:
        nslt_data = json.load(f)   # dict: vid -> {subset, action}

    class_to_vids = {}
    for vid, info in nslt_data.items():
        class_id = info["action"][0]  # action[0] = class_id :contentReference[oaicite:2]{index=2}
        class_to_vids.setdefault(class_id, []).append(vid)

    #B2: video_id -> gloss từ WLASL_v0.3
    with open(wlasl_path, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)  # list entries :contentReference[oaicite:3]{index=3}

    vid_to_gloss = {}
    for entry in wlasl_data:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            vid = inst["video_id"]
            vid_to_gloss[vid] = gloss

    #B3: class_id -> gloss (qua video_id)
    idx2gloss = [f"CLASS_{i}" for i in range(num_classes)]
    idx2vid   = [None] * num_classes

    for class_id, vid_list in class_to_vids.items():
        gloss = None
        chosen_vid = None

        # chọn video_id đầu tiên có xuất hiện trong WLASL_v0.3.json
        for vid in vid_list:
            if vid in vid_to_gloss:
                gloss = vid_to_gloss[vid]
                chosen_vid = vid
                break

        if gloss is None:
            # fallback: không tìm thấy, dùng tên CLASS_i
            gloss = f"CLASS_{class_id}"

        idx2gloss[class_id] = str(gloss).upper()
        idx2vid[class_id]   = chosen_vid

    # nếu có class_id nào chưa set (class không có trong nslt_2000) thì để default CLASS_i
    return idx2gloss, idx2vid


def load_model(ckpt_path, num_classes):
    model = I3D(num_classes=num_classes, in_channels=3)  # RGB
    state = torch.load(ckpt_path, map_location="cpu")

    # nếu checkpoint được lưu dạng {'state_dict': ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # nếu checkpoint được train bằng DataParallel, bỏ tiền tố 'module.'
    new_state = {}
    for k, v in state.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

def temporal_vote(queue_preds, last_k=STABLE_WINS):
    """
    Lấy nhãn ổn định nhất qua K cửa sổ gần nhất.
    queue_preds chứa các tuple (label, conf).
    """
    if len(queue_preds) < last_k:
        return None, 0.0
    last = list(queue_preds)[-last_k:]
    counter = Counter([p[0] for p in last])
    label, count = counter.most_common(1)[0]
    same = [p for p in last if p[0] == label]
    avg_conf = sum([p[1] for p in same]) / max(1, len(same))
    return label, avg_conf

def main():
    # print("Loading labels...")
    # idx2gloss = build_label_map(LABELS_JSON)
    # # nếu label map ngắn hơn NUM_CLASSES, pad cho đủ
    # if NUM_CLASSES > len(idx2gloss):
    #     idx2gloss += [f"CLASS_{i}" for i in range(len(idx2gloss), NUM_CLASSES)]
    # print("Num labels:", len(idx2gloss))

    print("Loading label map from nslt_2000.json + WLASL_v0.3.json ...")
    idx2gloss, idx2vid = build_idx2gloss_from_json(NSLT_JSON, WLASL_JSON, NUM_CLASSES)
    print("Num labels:", len(idx2gloss))


    print("Loading model...")
    model = load_model(CKPT_PATH, NUM_CLASSES)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frames = deque(maxlen=CLIP_LEN)   # buffer T khung
    win_preds = deque(maxlen=32)      # lưu (label, conf) theo cửa sổ
    output_words = []                 # chuỗi từ xuất ra
    last_committed = None             # từ vừa “chốt” gần nhất
    silent_counter = 0                # đếm cửa sổ “im lặng”

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuẩn hoá & đẩy vào buffer
        arr = preprocess_frame_bgr(frame)  # HWC float32 [-1,1]
        frames.append(arr)

        # Khi đủ T khung -> suy luận
        if len(frames) == CLIP_LEN and (len(win_preds) == 0 or (len(win_preds) % 1 == 0)):
            with torch.no_grad():
                clip = hwc_to_tchw(list(frames))  # 1,C,T,H,W
                if torch.cuda.is_available():
                    clip = clip.cuda(non_blocking=True)

                logits = model(clip)  # có thể là [1, C] hoặc [1, C, T]

                # Nếu còn trục thời gian T thì lấy trung bình theo T
                if logits.ndim == 3:          # [B, C, T]
                    logits = logits.mean(dim=2)  # -> [B, C]

                # Softmax theo lớp
                probs = F.softmax(logits, dim=1)  # [B, C]
                probs = probs.squeeze(0)          # -> [C]

                # Lấy lớp có xác suất cao nhất
                conf_tensor, idx_tensor = torch.max(probs, dim=0)  # scalar tensor
                conf = float(conf_tensor.item())
                idx = int(idx_tensor.item())
                pred_label = idx2gloss[idx]

            # lưu lại dự đoán theo cửa sổ
            win_preds.append((pred_label, conf))

            # nếu tự tin thấp → tăng “im lặng”, ngược lại reset
            if conf < CONF_THRESH:
                silent_counter += 1
            else:
                silent_counter = 0

            # kiểm tra ổn định qua nhiều cửa sổ
            # stable_label, stable_conf = temporal_vote(win_preds, STABLE_WINS)
            # if stable_label and stable_conf >= CONF_THRESH:
            #     # tránh spam từ lặp liên tục
            #     if stable_label != last_committed:
            #         output_words.append(stable_label)
            #         last_committed = stable_label
            #     silent_counter = 0

            stable_label, stable_conf = temporal_vote(win_preds, STABLE_WINS)
            if stable_label and stable_conf >= CONF_THRESH:
                # tránh spam từ lặp liên tục
                if stable_label != last_committed:
                    output_words.append(stable_label)
                    last_committed = stable_label

                    # 🔹 Nếu đã có đủ 3 từ, chỉ giữ lại từ mới nhất
                    if len(output_words) >= 3:
                        # giữ lại phần tử cuối cùng và đẩy nó về đầu
                        output_words[:] = [output_words[-1]]

                silent_counter = 0

            # nếu “im lặng” đủ lâu → chèn dấu chấm
            if silent_counter >= SILENCE_WINS:
                if len(output_words) and output_words[-1] not in [",", ".", "…"]:
                    output_words.append(".")
                last_committed = None
                silent_counter = 0

        # Vẽ overlay lên video
        # cur_text = " ".join(output_words[-12:])  # lấy 12 “từ” cuối hiển thị
        # cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0,0,0), -1)
        # cv2.putText(frame, f"ASL words: {cur_text}", (10, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # cv2.imshow("I3D ASL -> text (demo)", frame)
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('q'):
        #     break

        # Vẽ overlay lên video
        if len(output_words) >= MAX_WORDS_ON_SCREEN:
            # hiển thị đúng 3 từ đầu tiên
            # cur_text = " ".join(output_words[:MAX_WORDS_ON_SCREEN])
            cur_text = " ".join(output_words)
            clear_after = True
        else:
            # ít hơn 3 từ thì hiện hết
            cur_text = " ".join(output_words)
            clear_after = False

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(frame, f"ASL words: {cur_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("I3D ASL -> text (demo)", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        # Sau khi 3 từ đã được hiển thị ít nhất 1 frame, xóa đi để trống
        if clear_after:
            output_words.clear()
            last_committed = None  # cho phép lại cùng từ đó nếu bạn ký lần nữa

    cap.release()
    cv2.destroyAllWindows()
    print("Result:", " ".join(output_words))

if __name__ == "__main__":
    main()
