from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit

from sign_utils import FEATURES, SEQUENCE_LENGTH


def build_model(kind: str, classes: int):
    layers = [tf.keras.layers.Input((SEQUENCE_LENGTH, FEATURES))]
    if kind == "lstm":
        layers += [tf.keras.layers.LSTM(64, return_sequences=True),
                   tf.keras.layers.Dropout(0.25), tf.keras.layers.LSTM(48)]
    else:
        layers += [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48, return_sequences=True)),
                   tf.keras.layers.Dropout(0.25),
                   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))]
    layers += [tf.keras.layers.Dense(64, activation="relu"),
               tf.keras.layers.Dropout(0.25),
               tf.keras.layers.Dense(classes, activation="softmax")]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="ฝึกและประเมิน LSTM/BiLSTM")
    parser.add_argument("--model", choices=["lstm", "bilstm"], default="lstm")
    parser.add_argument("--data", default="data/sequences.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as data:
        X, labels, groups = data["X"], data["y"].astype(str), data["subjects"].astype(str)
        display_texts = data["texts"].astype(str) if "texts" in data.files else labels.copy()
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise SystemExit("ต้องมีอย่างน้อย 2 คลาส")
    unique_subjects = np.unique(groups)
    if len(unique_subjects) < 3:
        raise SystemExit("ต้องมีผู้ทำท่าอย่างน้อย 3 คนเพื่อแยก train/validation/test ตามบุคคล; แนะนำอย่างน้อย 5 คน")
    if not (len(X) == len(labels) == len(groups) == len(display_texts)):
        raise SystemExit("จำนวน X, y, subjects และ texts ในชุดข้อมูลไม่เท่ากัน")
    if any(len(set(display_texts[labels == label].tolist())) != 1 for label in classes):
        raise SystemExit("ข้อความภาษาไทยของ label เดียวกันไม่ตรงกัน; เก็บข้อมูลแต่ละคลาสด้วย --text ค่าเดียวกัน")
    if X.ndim != 3 or X.shape[1:] != (SEQUENCE_LENGTH, FEATURES):
        raise SystemExit(f"รูปทรงข้อมูลไม่ถูกต้อง: ต้องเป็น (N, {SEQUENCE_LENGTH}, {FEATURES})")

    label_to_id = {label: i for i, label in enumerate(classes)}
    y = np.asarray([label_to_id[label] for label in labels], dtype=np.int64)
    split_test = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    trainval_idx, test_idx = next(split_test.split(X, y, groups))
    split_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed + 1)
    train_rel, val_rel = next(split_val.split(X[trainval_idx], y[trainval_idx], groups[trainval_idx]))
    train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]
    expected_classes = set(range(len(classes)))
    if any(set(y[idx]) != expected_classes for idx in (train_idx, val_idx, test_idx)):
        raise SystemExit("การแบ่งตามผู้ทำท่าทำให้บางคลาสหายจาก train/validation/test; เพิ่มผู้ทำท่าที่มีข้อมูลครบทุกคลาส")

    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(args.model, len(classes))
    model.fit(X[train_idx], y[train_idx], validation_data=(X[val_idx], y[val_idx]), epochs=args.epochs,
              batch_size=32, callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=8, restore_best_weights=True)], verbose=1)
    started = time.perf_counter()
    probs = model.predict(X[test_idx], verbose=0)
    inference_ms = (time.perf_counter() - started) * 1000 / len(test_idx)
    pred = probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y[test_idx], pred, labels=np.arange(len(classes)), average="macro", zero_division=0)
    report = {
        "model": args.model,
        "classes": classes,
        "seed": args.seed,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "validation_samples": int(len(val_idx)),
        "train_subjects": sorted(set(groups[train_idx].tolist())),
        "validation_subjects": sorted(set(groups[val_idx].tolist())),
        "test_subjects": sorted(set(groups[test_idx].tolist())),
        "label_text": {label: sorted(set(display_texts[labels == label].tolist()))[0] for label in classes},
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "inference_ms_per_sequence": float(inference_ms),
        "classification_report": classification_report(
            y[test_idx], pred, labels=np.arange(len(classes)), target_names=classes,
            zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y[test_idx], pred,
                                                labels=np.arange(len(classes))).tolist(),
    }
    out = Path("models")
    out.mkdir(exist_ok=True)
    model.save(out / f"{args.model}.keras")
    (out / f"{args.model}_metadata.json").write_text(
        json.dumps({"classes": classes,
                    "label_text": {label: sorted(set(display_texts[labels == label].tolist()))[0]
                                   for label in classes},
                    "sequence_length": SEQUENCE_LENGTH,
                    "features": FEATURES}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{args.model}_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "model", "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "inference_ms_per_sequence", "train_subjects", "test_subjects")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
