#!/usr/bin/env python3
# convert_coco_negatives.py
#
# Downloads a subset of COCO val2017 images that contain NO 'person'
# annotations (and therefore essentially no hands), and labels them
# all as negatives (has_hand=0) in the unified format.
#
# Usage:
#   python3 convert_coco_negatives.py <output_root> [--count N]
#
# Output:
#   <output_root>/unified/labels/coco_<id>.txt   -> "0 0.0 0.0 0.0 0.0"
#   <output_root>/unified/images/coco_<id>.jpg   -> downloaded image

import json
import os
import sys
import urllib.request
import random

from concurrent.futures import ThreadPoolExecutor, as_completed

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_BASE = "http://images.cocodataset.org/val2017/"
ANN_FILE = "instances_val2017.json"

# Categories that bias toward desk/room/indoor scenes without people
PREFERRED_CATEGORIES = {
    "chair", "laptop", "keyboard", "cup", "book", "tv",
    "cell phone", "bottle", "couch", "potted plant", "clock",
    "dining table", "remote", "bowl", "vase",
}


def download_annotations(cache_dir):
    """Downloads and extracts instances_val2017.json if not already cached."""
    ann_path = os.path.join(cache_dir, ANN_FILE)
    if os.path.exists(ann_path):
        return ann_path

    import zipfile
    import tempfile

    print("Downloading COCO annotations (this is ~240MB, one-time)...")
    zip_path = os.path.join(cache_dir, "annotations_trainval2017.zip")
    os.makedirs(cache_dir, exist_ok=True)

    urllib.request.urlretrieve(COCO_ANN_URL, zip_path)

    print("Extracting annotations json...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip contains annotations/instances_val2017.json among others
        member = f"annotations/{ANN_FILE}"
        with zf.open(member) as src, open(ann_path, "wb") as dst:
            dst.write(src.read())

    os.remove(zip_path)
    return ann_path


def find_negative_image_ids(ann_path, count, seed=42):
    """
    Returns a list of (image_id, file_name, width, height) tuples for
    images that contain NO 'person' annotation.
    """
    print("Loading annotations json (this may take a moment)...")
    with open(ann_path, "r") as f:
        data = json.load(f)

    cat_name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    person_cat_id = cat_name_to_id.get("person")

    # Build set of image_ids that contain at least one 'person' annotation
    images_with_person = set()
    for ann in data["annotations"]:
        if ann["category_id"] == person_cat_id:
            images_with_person.add(ann["image_id"])

    print(f"Images with person: {len(images_with_person)}")

    all_images = {img["id"]: img for img in data["images"]}
    candidate_ids = [
        img_id for img_id in all_images
        if img_id not in images_with_person
    ]

    print(f"Candidate person-free images: {len(candidate_ids)}")

    # Optional: bias toward images containing at least one preferred
    # category (desk/room context). Fall back to any candidate if not
    # enough preferred-category images exist.
    preferred_cat_ids = {
        cat_name_to_id[name] for name in PREFERRED_CATEGORIES
        if name in cat_name_to_id
    }
    images_with_preferred = set()
    for ann in data["annotations"]:
        if ann["category_id"] in preferred_cat_ids:
            images_with_preferred.add(ann["image_id"])

    preferred_candidates = [
        img_id for img_id in candidate_ids
        if img_id in images_with_preferred
    ]

    print(f"Preferred-category candidates: {len(preferred_candidates)}")

    random.seed(seed)

    chosen_ids = []
    if len(preferred_candidates) >= count:
        chosen_ids = random.sample(preferred_candidates, count)
    else:
        # Take all preferred candidates, fill the rest from general pool
        chosen_ids = list(preferred_candidates)
        remaining_pool = [i for i in candidate_ids if i not in images_with_preferred]
        needed = count - len(chosen_ids)
        chosen_ids += random.sample(remaining_pool, min(needed, len(remaining_pool)))

    result = []
    for img_id in chosen_ids:
        img = all_images[img_id]
        result.append((img_id, img["file_name"], img["width"], img["height"]))

    return result

def download_one(img_id, file_name, out_images_dir, out_labels_dir):
    stem = f"coco_{img_id}"
    out_img_path = os.path.join(out_images_dir, stem + ".jpg")
    out_label_path = os.path.join(out_labels_dir, stem + ".txt")

    if not os.path.exists(out_img_path):
        url = COCO_IMG_BASE + file_name
        try:
            urllib.request.urlretrieve(url, out_img_path)
        except Exception as e:
            return False, file_name, str(e)

    with open(out_label_path, "w") as f:
        f.write("0 0.0 0.0 0.0 0.0\n")

    return True, file_name, None


def download_and_label(images, out_labels_dir, out_images_dir, workers=16):
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_images_dir, exist_ok=True)

    converted, skipped = 0, 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, img_id, file_name, out_images_dir, out_labels_dir): file_name
            for img_id, file_name, _, _ in images
        }

        for i, future in enumerate(as_completed(futures), 1):
            ok, file_name, err = future.result()
            if ok:
                converted += 1
            else:
                print(f"skip {file_name}: {err}")
                skipped += 1

            if i % 50 == 0:
                print(f"  {i}/{len(images)} processed")

    print(f"Converted={converted} Skipped={skipped}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_root> [--count N]")
        sys.exit(1)

    out_root = os.path.abspath(sys.argv[1])

    count = 800
    if "--count" in sys.argv:
        idx = sys.argv.index("--count")
        count = int(sys.argv[idx + 1])

    cache_dir = os.path.join(out_root, "_cache")
    out_labels = os.path.join(out_root, "unified", "labels")
    out_images = os.path.join(out_root, "unified", "images")

    ann_path = download_annotations(cache_dir)
    images = find_negative_image_ids(ann_path, count)

    print(f"Downloading {len(images)} negative images...")
    download_and_label(images, out_labels, out_images)
