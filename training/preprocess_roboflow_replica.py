"""Code replica of the preprocessing and augmentations Roboflow applies.

WHAT THIS IS AND IS NOT:
This script REPLICATES in code the transformations the Roboflow platform
applies at export time. It is an "equivalent of", NOT a bit-for-bit
reproduction: the libraries, the internal order of operations and the seeds
Roboflow uses are not public. On the same input, this script and Roboflow
produce statistically comparable datasets, not identical files.

Transformations replicated (from the original export's README.roboflow.txt):

  Preprocessing, on ALL images
    * auto-orient with EXIF orientation stripping
    * resize to 576x576 (stretch, not letterbox)

  Augmentation, to create N versions of each source image
    * 50% chance of horizontal flip
    * random rotation between -5 and +5 degrees
    * random brightness between -20% and +20%
    * random exposure between -15% and +15%
    * random Gaussian blur between 0 and 0.5 pixels

Fidelity of the mapping onto Albumentations:

  | Roboflow            | Albumentations                                  | fidelity    |
  |---------------------|-------------------------------------------------|-------------|
  | horizontal flip 50% | HorizontalFlip(p=0.5)                           | equivalent  |
  | rotation +/-5 deg   | Rotate(limit=5)                                 | equivalent* |
  | brightness +/-20%   | RandomBrightnessContrast(brightness_limit=0.2)  | approximate |
  | exposure +/-15%     | RandomGamma(gamma_limit=(85, 115))              | approximate |
  | blur 0-0.5 px       | GaussianBlur(sigma_limit=(0.0, 0.5))            | equivalent  |

  * border handling after rotation may differ.
  The two "approximate" rows are the least certain: Roboflow's "brightness" and
  "exposure" have no documented 1:1 equivalent, and are rendered here as an
  additive shift and a gamma correction respectively.

WHY 576x576: it is RF-DETR-Medium's native resolution
(rfdetr/config.py, RFDETRMediumConfig.resolution = 576).

CLASS CONTRACT: the input COCO categories are copied out UNCHANGED. The script
neither remaps nor renumbers ids: a dataset that goes in as ball=1, dot=2,
hand=3 comes out identical. See training/README.md.

Usage (from the repo root):
    python -m training.preprocess_roboflow_replica \\
        --input ./data/raw_labeled --output ./data/basketball_coco --versions 5

TESTING STATUS: the COCO logic (split discovery, the double-augmentation guard,
id handling, annotation writing) has been verified on a real dataset. The
Albumentations chain has NEVER been run in this environment: the library was
not installable (PyPI unreachable). Test it on a handful of images before
launching it on a whole dataset.
"""

import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from spindoctor.utils import ensure_dir

# Albumentations is a data-prep-only dependency (see
# training/requirements-training.txt). The import is guarded so that --help and
# argument validation still work without it, and so the error when it is
# missing is readable rather than a raw ModuleNotFoundError.
try:
    import albumentations as A
except ImportError:
    A = None

TARGET_SIZE = 576          # RF-DETR-Medium's native resolution
DEFAULT_VERSIONS = 5       # as in the original Roboflow export
DEFAULT_INPUT = "./data/raw_labeled"
DEFAULT_OUTPUT = "./data/basketball_coco"
DEFAULT_SEED = 42
ANNOTATION_FILE = "_annotations.coco.json"
SPLIT_NAMES = ("train", "valid", "test")

# Roboflow augments train only, leaving valid/test with preprocessing alone so
# evaluation is not skewed. Same here.
DEFAULT_AUGMENT_SPLITS = ("train",)

# Minimum fraction of a box that must remain visible for it to be kept. Matters
# for rotation, which can push annotations near the edge out of frame.
MIN_VISIBILITY = 0.3


# --- Double-augmentation guard ---
def detect_already_processed(input_dir):
    """
    Detect an already-processed Roboflow export.

    Re-running the augmentations on a dataset that has already had them
    produces images rotated twice, blurred twice and brightened twice: training
    comes out worse, not better. Better to stop.

    Two independent signals:
      1. a README.roboflow.txt declaring augmentations were applied
      2. filenames carrying the `.rf.` hash Roboflow assigns on export

    Returns:
        A string describing the reason for suspicion, or None.
    """
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.endswith(".roboflow.txt"):
                path = os.path.join(root, name)
                try:
                    text = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if "augmentation was applied" in text.lower():
                    return (f"{os.path.relpath(path, input_dir)} declares that "
                            f"augmentations were already applied by Roboflow")

    rf_hashed = 0
    checked = 0
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                checked += 1
                if ".rf." in name:
                    rf_hashed += 1
                if checked >= 200:
                    break
        if checked >= 200:
            break

    if checked and rf_hashed / checked > 0.5:
        return (f"{rf_hashed} of {checked} sampled images carry the `.rf.` "
                f"hash typical of an already-processed Roboflow export")

    return None


# --- COCO I/O ---
def discover_splits(input_dir):
    """
    Work out the structure of the input dataset.

    Returns:
        List of (split_name, folder). For a flat dataset (annotations in the
        root folder) returns [(None, input_dir)].
    """
    splits = []
    for name in SPLIT_NAMES:
        folder = os.path.join(input_dir, name)
        if os.path.isfile(os.path.join(folder, ANNOTATION_FILE)):
            splits.append((name, folder))

    if splits:
        return splits

    if os.path.isfile(os.path.join(input_dir, ANNOTATION_FILE)):
        return [(None, input_dir)]

    return []


def load_coco(folder):
    with open(os.path.join(folder, ANNOTATION_FILE), encoding="utf-8") as f:
        return json.load(f)


def clamp_bbox(bbox, width, height):
    """
    Clamp a COCO box [x, y, w, h] into the image bounds.

    Albumentations rejects boxes that fall outside the frame, and hand-made
    annotations often contain some a few pixels over the edge.

    Returns:
        The clamped box, or None if degenerate (width or height <= 0).
    """
    x, y, w, h = bbox
    x1 = max(0.0, min(float(x), width))
    y1 = max(0.0, min(float(y), height))
    x2 = max(0.0, min(float(x) + float(w), width))
    y2 = max(0.0, min(float(y) + float(h), height))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def load_image_rgb(path):
    """
    Load an image, applying EXIF orientation and dropping the metadata.

    This is Roboflow's "auto-orient with EXIF-orientation stripping": the
    rotation declared in EXIF is baked into the pixels, and the tag is not
    written back out (PIL does not copy EXIF unless asked).
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.array(img.convert("RGB"))


# --- Transformation chain ---
def _bbox_params():
    return A.BboxParams(format="coco", label_fields=["category_ids"],
                        min_visibility=MIN_VISIBILITY)


def build_preprocess():
    """Preprocessing only: stretch resize to 576x576, no augmentation."""
    if A is None:
        raise SystemExit(
            "albumentations is not installed.\n"
            "  pip install -r training/requirements-training.txt")
    return A.Compose(
        [A.Resize(height=TARGET_SIZE, width=TARGET_SIZE)],
        bbox_params=_bbox_params(),
    )


def build_augment():
    """Preprocessing plus the five augmentations of the Roboflow export."""
    if A is None:
        raise SystemExit(
            "albumentations is not installed.\n"
            "  pip install -r training/requirements-training.txt")
    return A.Compose(
        [
            A.Resize(height=TARGET_SIZE, width=TARGET_SIZE),
            # 50% probability of horizontal flip
            A.HorizontalFlip(p=0.5),
            # Random rotation of between -5 and +5 degrees
            A.Rotate(limit=5, p=1.0),
            # Random brightness adjustment of between -20 and +20 percent
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
            # Random exposure adjustment of between -15 and +15 percent,
            # rendered as a gamma correction (approximate mapping)
            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
            # Random Gaussian blur of between 0 and 0.5 pixels
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.0, 0.5), p=1.0),
        ],
        bbox_params=_bbox_params(),
    )


# --- Split processing ---
def process_split(src_folder, dst_folder, versions, augment, quality=95):
    """
    Process one split, writing images and COCO annotations to the destination.

    Categories are copied unchanged. Images and annotations are renumbered from
    scratch, because each source produces several outputs.

    Args:
        src_folder: Folder holding ANNOTATION_FILE and the images.
        dst_folder: Destination folder.
        versions: Versions per source image. Version 0 is preprocessing only;
            the rest are augmented.
        augment: If False, produce a single preprocessing-only version and
            ignore `versions`.
        quality: Output JPEG quality.

    Returns:
        Dict of processing statistics.
    """
    coco = load_coco(src_folder)
    ensure_dir(dst_folder)

    # annotations grouped by image
    by_image = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(ann["image_id"], []).append(ann)

    preprocess = build_preprocess()
    augmenter = build_augment() if augment else None

    out_images = []
    out_annotations = []
    next_image_id = 1
    next_ann_id = 1

    stats = {"sources": 0, "produced": 0, "degenerate_boxes_dropped": 0,
             "boxes_lost_in_transform": 0, "unreadable_images": 0,
             "exif_mismatch": 0}

    n_versions = versions if augment else 1

    for img_rec in tqdm(coco.get("images", []), desc=f"  {os.path.basename(dst_folder)}",
                        leave=False):
        src_path = os.path.join(src_folder, img_rec["file_name"])
        try:
            image = load_image_rgb(src_path)
        except (OSError, ValueError):
            stats["unreadable_images"] += 1
            continue

        stats["sources"] += 1
        h_src, w_src = image.shape[:2]

        # If auto-orient changed the dimensions from those declared in the
        # COCO, the annotations referred to the unrotated image and no longer
        # line up.
        if (img_rec.get("width") and img_rec.get("height")
                and (w_src != img_rec["width"] or h_src != img_rec["height"])):
            stats["exif_mismatch"] += 1

        anns = by_image.get(img_rec["id"], [])
        bboxes = []
        category_ids = []
        for ann in anns:
            clamped = clamp_bbox(ann["bbox"], w_src, h_src)
            if clamped is None:
                stats["degenerate_boxes_dropped"] += 1
                continue
            bboxes.append(clamped)
            category_ids.append(ann["category_id"])

        stem = os.path.splitext(os.path.basename(img_rec["file_name"]))[0]

        for version in range(n_versions):
            transform = preprocess if version == 0 else augmenter
            result = transform(image=image, bboxes=bboxes, category_ids=category_ids)

            out_name = f"{stem}_v{version}.jpg"
            Image.fromarray(result["image"]).save(
                os.path.join(dst_folder, out_name), quality=quality)

            out_images.append({
                "id": next_image_id,
                "file_name": out_name,
                "width": TARGET_SIZE,
                "height": TARGET_SIZE,
            })

            kept = len(result["bboxes"])
            stats["boxes_lost_in_transform"] += len(bboxes) - kept

            for bbox, cat_id in zip(result["bboxes"], result["category_ids"]):
                x, y, w, h = [float(v) for v in bbox[:4]]
                out_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": cat_id,   # PRESERVED: the class contract
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    # iscrowd is zeroed: transformations can drop and reorder
                    # boxes, so they cannot be reliably tied back to the source
                    # annotations. Irrelevant for single-box detection data.
                    "iscrowd": 0,
                })
                next_ann_id += 1

            next_image_id += 1
            stats["produced"] += 1

    out_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        # copied unchanged: this is where ball=1, dot=2, hand=3 is preserved
        "categories": coco.get("categories", []),
        "images": out_images,
        "annotations": out_annotations,
    }

    with open(os.path.join(dst_folder, ANNOTATION_FILE), "w", encoding="utf-8") as f:
        json.dump(out_coco, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Code replica of Roboflow's preprocessing and augmentations.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"COCO dataset of RAW labelled frames (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Destination folder (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--versions", type=int, default=DEFAULT_VERSIONS,
                        help=f"Versions per source image, the first without "
                             f"augmentation (default: {DEFAULT_VERSIONS})")
    parser.add_argument("--augment-splits", type=str, nargs="*",
                        default=list(DEFAULT_AUGMENT_SPLITS),
                        help="Splits to augment; the rest get preprocessing "
                             "only (default: train)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Seed for reproducibility (default: {DEFAULT_SEED})")
    parser.add_argument("--quality", type=int, default=95,
                        help="Output JPEG quality (default: 95)")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even if the input looks like an already-"
                             "processed Roboflow export (double-augmentation risk)")
    args = parser.parse_args()

    if args.versions < 1:
        parser.error("--versions must be >= 1")
    if not os.path.isdir(args.input):
        parser.error(f"input folder not found: {args.input}")

    # --- Double-augmentation guard ---
    reason = detect_already_processed(args.input)
    if reason and not args.force:
        print("BLOCKED: the input looks like an ALREADY-processed Roboflow export.")
        print(f"  Reason: {reason}")
        print()
        print("  This script is meant for RAW labelled frames, not for an export")
        print("  that has already been preprocessed and augmented. Running it here")
        print("  would augment the same images twice.")
        print()
        print("  If you know what you are doing, re-run with --force.")
        raise SystemExit(2)
    if reason and args.force:
        print(f"WARNING: {reason}")
        print("  Proceeding anyway because of --force (double augmentation).")

    splits = discover_splits(args.input)
    if not splits:
        parser.error(
            f"no {ANNOTATION_FILE} found in {args.input}: expected a flat COCO "
            f"dataset, or one split into {'/'.join(SPLIT_NAMES)}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # First split's categories, printed to make the contract visible.
    first_coco = load_coco(splits[0][1])
    print("Input categories (copied out unchanged):")
    for cat in first_coco.get("categories", []):
        print(f"  id={cat.get('id')}  name={cat.get('name')}")
    print()

    ensure_dir(args.output)
    totals = {}

    for split_name, src_folder in splits:
        augment = split_name in args.augment_splits if split_name else True
        dst_folder = os.path.join(args.output, split_name) if split_name else args.output
        label = split_name or "(flat)"
        n = args.versions if augment else 1
        print(f"Split {label}: {n} version(s) per image"
              f"{'' if augment else ' (preprocessing only)'}")
        totals[label] = process_split(src_folder, dst_folder, args.versions,
                                      augment, quality=args.quality)

    # The source export's readme is deliberately not copied: it describes
    # different transformations from the ones just applied, and would mislead
    # anyone reading the output.
    print("\nSummary:")
    for label, stats in totals.items():
        print(f"  {label}: {stats['sources']} sources -> {stats['produced']} images")
        if stats["degenerate_boxes_dropped"]:
            print(f"    degenerate boxes dropped on input: {stats['degenerate_boxes_dropped']}")
        if stats["boxes_lost_in_transform"]:
            print(f"    boxes pushed out of frame by the transform: "
                  f"{stats['boxes_lost_in_transform']}")
        if stats["unreadable_images"]:
            print(f"    unreadable images skipped: {stats['unreadable_images']}")
        if stats["exif_mismatch"]:
            print(f"    WARNING: {stats['exif_mismatch']} images change size "
                  f"after auto-orient; their annotations may no longer line up")

    print(f"\nDataset written to {args.output}")
    print("Next step: python -m training.train_rfdetr --dataset-dir "
          f"{args.output}")


if __name__ == "__main__":
    main()
