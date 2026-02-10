import json
from pathlib import Path
import argparse

def build_label_map(nslt_path, wlasl_path, out_path):
    nslt_path = Path(nslt_path)
    wlasl_path = Path(wlasl_path)
    out_path = Path(out_path)

    print("[INFO] Loading nslt:", nslt_path)
    with nslt_path.open("r", encoding="utf-8") as f:
        nslt = json.load(f)   # dict: video_id -> {subset, action=[class_id, start, end], ...}

    print("[INFO] Loading WLASL_v0.3:", wlasl_path)
    with wlasl_path.open("r", encoding="utf-8") as f:
        wlasl = json.load(f)  # list các gloss, mỗi gloss có instances

    # B1: build map video_id -> gloss từ WLASL_v0.3.json
    vid2gloss = {}
    for entry in wlasl:
        gloss = entry.get("gloss", None)
        if not gloss:
            continue
        for inst in entry.get("instances", []):
            vid = inst.get("video_id")
            if not vid:
                continue
            # nhiều video cùng 1 gloss -> OK
            vid2gloss[str(vid)] = gloss.upper()

    # B2: build map class_id -> gloss dựa vào nslt_2000.json
    class2gloss = {}
    for vid, info in nslt.items():
        action = info.get("action", None)
        if not action or len(action) == 0:
            continue
        class_id = int(action[0])

        gloss = vid2gloss.get(str(vid))
        if gloss is None:
            # nếu không match được, để tạm
            gloss = class2gloss.get(class_id, f"CLASS_{class_id}")
        # chỉ set 1 lần cho mỗi class_id (giữ gloss đầu tiên tìm được)
        if class_id not in class2gloss:
            class2gloss[class_id] = gloss

    # B3: tạo list labels theo index từ 0..max_class_id
    max_id = max(class2gloss.keys())
    labels = [class2gloss.get(i, f"CLASS_{i}") for i in range(max_id + 1)]

    print("[INFO] Num classes in labels:", len(labels))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print("[DONE] Saved labels to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nslt_json", default="preprocess/nslt_2000.json",
                        help="Đường dẫn nslt_2000.json")
    parser.add_argument("--wlasl_json", default="../../start_kit/WLASL_v0.3.json",
                        help="Đường dẫn WLASL_v0.3.json")
    parser.add_argument("--out", default="preprocess/labels_2000.json",
                        help="File output chứa list labels")
    args = parser.parse_args()

    build_label_map(args.nslt_json, args.wlasl_json, args.out)
