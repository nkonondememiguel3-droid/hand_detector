#!/usr/bin/env python3
# convert_egohands.py
import scipy.io
import numpy as np
from PIL import Image
import os
import sys

HAND_FIELDS = ["myleft", "myright", "yourleft", "yourright"]


def link_image(out_img, img_path):
    target = os.path.abspath(img_path)
    if os.path.islink(out_img):
        if os.readlink(out_img) == target:
            return
        os.remove(out_img)
    elif os.path.exists(out_img):
        os.remove(out_img)
    os.symlink(target, out_img)


def best_hand_box(frame, width, height):
    best_area = -1
    best_box = None

    for field in HAND_FIELDS:
        poly = frame[field]
        if poly is None or poly.size == 0:
            continue
        pts = np.asarray(poly, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue

        xs = pts[:, 0]
        ys = pts[:, 1]
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        xmin = max(0, min(xmin, width - 1))
        xmax = max(0, min(xmax, width - 1))
        ymin = max(0, min(ymin, height - 1))
        ymax = max(0, min(ymax, height - 1))

        area = (xmax - xmin) * (ymax - ymin)
        if area > best_area:
            best_area = area
            best_box = (xmin, ymin, xmax, ymax)

    return best_box, best_area


def write_label(out_path, best_box, best_area, width, height):
    if best_box is None or best_area <= 0:
        line = "0 0.0 0.0 0.0 0.0\n"
    else:
        xmin, ymin, xmax, ymax = best_box
        cx = (xmin + xmax) / 2.0 / width
        cy = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        line = f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"

    with open(out_path, "w") as f:
        f.write(line)


def convert_video(vid_meta, labelled_samples_root, out_labels_dir, out_images_dir):
    video_id = vid_meta["video_id"][0]
    video_dir = os.path.join(labelled_samples_root, video_id)
    labelled_frames = vid_meta["labelled_frames"][0]

    converted, skipped = 0, 0

    for frame in labelled_frames:
        frame_num = int(frame["frame_num"][0, 0])
        img_path = os.path.join(video_dir, f"frame_{frame_num:04d}.jpg")
        if not os.path.exists(img_path):
            skipped += 1
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        best_box, best_area = best_hand_box(frame, width, height)
        out_stem = f"{video_id}_{frame_num:04d}"
        out_path = os.path.join(out_labels_dir, out_stem + ".txt")
        write_label(out_path, best_box, best_area, width, height)
        link_image(os.path.join(out_images_dir, out_stem + ".jpg"), img_path)
        converted += 1

    return converted, skipped


def resolve_paths(arg_path):
    root = os.path.abspath(arg_path)
    if os.path.basename(root) == "_LABELLED_SAMPLES":
        labelled_samples = root
        egohands_root = os.path.dirname(root)
    else:
        egohands_root = root
        labelled_samples = os.path.join(egohands_root, "_LABELLED_SAMPLES")
    return egohands_root, labelled_samples


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <egohands-root-or-_LABELLED_SAMPLES>")
        sys.exit(1)

    egohands_root, labelled_samples = resolve_paths(sys.argv[1])
    metadata_path = os.path.join(egohands_root, "metadata.mat")
    if not os.path.exists(metadata_path):
        print(f"metadata not found: {metadata_path}")
        sys.exit(1)

    out_labels = os.path.join(egohands_root, "unified", "labels")
    out_images = os.path.join(egohands_root, "unified", "images")
    os.makedirs(out_labels, exist_ok=True)
    os.makedirs(out_images, exist_ok=True)

    meta = scipy.io.loadmat(metadata_path)
    videos = meta["video"][0]

    total_c, total_s = 0, 0
    for vid in videos:
        c, s = convert_video(vid, labelled_samples, out_labels, out_images)
        total_c += c
        total_s += s

    print(f"Total converted={total_c} skipped={total_s}")
