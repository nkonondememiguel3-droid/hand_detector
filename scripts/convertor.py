#!/usr/bin/env python3
import scipy.io
from PIL import Image
import os
import sys

SPLIT_PREFIXES = {
    "training_dataset": "train",
    "validation_dataset": "val",
    "test_dataset": "test",
}

SPLITS = {
    "training_dataset": "training_data",
    "validation_dataset": "validation_data",
    "test_dataset": "test_data",
}


def link_image(out_img, img_path):
    target = os.path.abspath(img_path)
    if os.path.islink(out_img):
        if os.readlink(out_img) == target:
            return
        os.remove(out_img)
    elif os.path.exists(out_img):
        os.remove(out_img)
    os.symlink(target, out_img)


def convert_split(split_dir, split_prefix, out_labels_dir, out_images_dir):
    ann_dir = os.path.join(split_dir, "annotations")
    img_dir = os.path.join(split_dir, "images")
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_images_dir, exist_ok=True)

    converted, skipped = 0, 0

    for fname in os.listdir(ann_dir):
        if not fname.endswith(".mat"):
            continue
        stem = f"{split_prefix}_{fname[:-4]}"
        mat_path = os.path.join(ann_dir, fname)
        img_path = os.path.join(img_dir, fname[:-4] + ".jpg")

        if not os.path.exists(img_path):
            skipped += 1
            continue

        try:
            data = scipy.io.loadmat(mat_path)
        except Exception as e:
            print(f"skip {fname}: {e}")
            skipped += 1
            continue

        boxes = data.get("boxes")
        out_path = os.path.join(out_labels_dir, stem + ".txt")

        if boxes is None or boxes.size == 0:
            with open(out_path, "w") as f:
                f.write("0 0.0 0.0 0.0 0.0\n")
            link_image(os.path.join(out_images_dir, stem + ".jpg"), img_path)
            converted += 1
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        best_area = -1
        best_box = None

        n = boxes.shape[1]
        for j in range(n):
            entry = boxes[0, j][0, 0]
            try:
                a = entry["a"][0]
                b = entry["b"][0]
                c = entry["c"][0]
                d = entry["d"][0]
            except (KeyError, IndexError):
                continue

            pts = [a, b, c, d]
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            xmin = max(0, min(xmin, W - 1))
            xmax = max(0, min(xmax, W - 1))
            ymin = max(0, min(ymin, H - 1))
            ymax = max(0, min(ymax, H - 1))

            area = (xmax - xmin) * (ymax - ymin)
            if area > best_area:
                best_area = area
                best_box = (xmin, ymin, xmax, ymax)

        if best_box is None or best_area <= 0:
            with open(out_path, "w") as f:
                f.write("0 0.0 0.0 0.0 0.0\n")
        else:
            xmin, ymin, xmax, ymax = best_box
            cx = (xmin + xmax) / 2.0 / W
            cy = (ymin + ymax) / 2.0 / H
            w = (xmax - xmin) / W
            h = (ymax - ymin) / H
            with open(out_path, "w") as f:
                f.write(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        link_image(os.path.join(out_images_dir, stem + ".jpg"), img_path)
        converted += 1

    print(f"{split_dir}: converted={converted} skipped={skipped}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-oxford>")
        sys.exit(1)

    base = os.path.abspath(sys.argv[1])
    out_labels = os.path.join(base, "unified", "labels")
    out_images = os.path.join(base, "unified", "images")
    os.makedirs(out_labels, exist_ok=True)
    os.makedirs(out_images, exist_ok=True)

    for split, subdir in SPLITS.items():
        split_dir = os.path.join(base, split, subdir)
        if not os.path.isdir(split_dir):
            print(f"skip missing {split_dir}")
            continue
        convert_split(split_dir, SPLIT_PREFIXES[split], out_labels, out_images)
