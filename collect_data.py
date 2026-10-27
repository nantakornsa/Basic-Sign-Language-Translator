from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from sign_utils import capture_sequence, open_hands


def main():
    parser = argparse.ArgumentParser(description="เก็บลำดับจุดสำคัญสำหรับคลาสภาษามือหนึ่งคลาส")
    parser.add_argument("--label", required=True, help="ชื่อคลาส เช่น HELLO")
    parser.add_argument("--text", help="ข้อความภาษาไทยที่จะแสดง เช่น สวัสดี (ถ้าไม่ระบุจะแสดง --label)")
    parser.add_argument("--subject", required=True, help="รหัสผู้ทำท่า เช่น S01")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples ต้องมากกว่า 0")

    path = Path("data/sequences.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            xs, ys, subjects = list(old["X"]), list(old["y"]), list(old["subjects"])
            texts = list(old["texts"]) if "texts" in old.files else list(ys)
    else:
        xs, ys, subjects, texts = [], [], [], []

    label = args.label.strip()
    display_text = (args.text or label).strip()
    if not label or not display_text or not args.subject.strip():
        parser.error("--label, --text และ --subject ห้ามเป็นค่าว่าง")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"เปิดกล้องหมายเลข {args.camera} ไม่ได้")
    count = 0
    try:
        with open_hands() as hands:
            while count < args.samples:
                seq = capture_sequence(hands, camera)
                if seq is None:
                    break
                xs.append(seq)
                ys.append(label)
                subjects.append(args.subject.strip())
                texts.append(display_text)
                count += 1
                np.savez_compressed(path, X=np.asarray(xs, dtype=np.float32),
                                    y=np.asarray(ys), subjects=np.asarray(subjects),
                                    texts=np.asarray(texts))
                print(f"บันทึก {count}/{args.samples} -> {path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"เก็บเพิ่ม {count} ตัวอย่างสำหรับ {label} / {args.subject}")


if __name__ == "__main__":
    main()
