"""Fine-tune RF-DETR-Medium on the ball/dots/hands dataset.

PROVENANCE - READ BEFORE USE:
This is NOT the script that produced the weights distributed with the project.
That model was trained on the Roboflow PLATFORM, from its web interface, with
no code. This file is a code reproduction of an equivalent training run,
written to make the process reproducible and inspectable.

Consequently the metrics reported in the README for the distributed model
(mAP@50, precision, recall, F1) come from that Roboflow run and are NOT a
promise of what this script will produce. The hyperparameters below are a
stable starting point for a dataset of roughly 1100 images, NOT the ones used
on Roboflow, which are not public.

TESTING STATUS: the validations (class contract, dataset structure,
configuration assembly) have been run and verified. Training itself has NEVER
been run end to end in this environment, which has no GPU. Before launching a
long run, try a couple of epochs first:

    python -m training.train_rfdetr --epochs 2 --dataset-dir ./data/basketball_coco

FINE-TUNING, NOT TRAINING FROM SCRATCH: RF-DETR starts from `pretrain_weights`
(default `rf-detr-medium.pth`, the COCO-pretrained checkpoint the library
downloads on first use) and adapts them to the custom dataset.

Usage (from the repo root):
    python -m training.train_rfdetr --dry-run           # validate, no training
    python -m training.train_rfdetr                     # train with defaults
    python -m training.train_rfdetr --epochs 100 --batch-size 8
"""

import argparse
import json
import os

DEFAULT_DATASET_DIR = "./data/basketball_coco"
DEFAULT_OUTPUT_DIR = "./runs/train"     # gitignored
ANNOTATION_FILE = "_annotations.coco.json"
REQUIRED_SPLITS = ("train", "valid")    # `test` is recommended, not required

# --- CLASS CONTRACT ---
# These ids must match what inference expects (spindoctor/config.py:
# CLASS_BALL=1, CLASS_DOT=2, CLASS_HAND=3). A model trained with different ids
# produces detections that inference reads as the wrong classes, silently.
EXPECTED_CLASSES = {1: "ball", 2: "dot", 3: "hand"}

# --- Default hyperparameters ---
# A reasonable starting point for ~1100 images, TO BE TUNED for your dataset.
DEFAULTS = {
    "epochs": 50,               # ours (library: 100); 50 is usually enough
                                # when fine-tuning on a small dataset
    "batch_size": 4,            # library default; raise it if you have VRAM
    "grad_accum_steps": 4,      # library default; effective batch = 4 x 4 = 16
    "lr": 1e-4,                 # library default
    "lr_encoder": 1.5e-4,       # library default
    "resolution": 576,          # RF-DETR-Medium's native resolution, and the
                                # size of the dataset images
    "num_workers": 2,           # library default
    "weight_decay": 1e-4,       # library default
    "early_stopping_patience": 10,
}


def load_categories(annotation_path):
    with open(annotation_path, encoding="utf-8") as f:
        return json.load(f).get("categories", [])


def validate_dataset_structure(dataset_dir):
    """
    Check the dataset has the expected COCO structure.

    Returns:
        List of (split_name, annotation_path) for the splits found.

    Raises:
        SystemExit: if a required split or its annotation file is missing.
    """
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"ERROR: dataset folder not found: {dataset_dir}")

    found = []
    missing = []
    for split in ("train", "valid", "test"):
        ann = os.path.join(dataset_dir, split, ANNOTATION_FILE)
        if os.path.isfile(ann):
            found.append((split, ann))
        elif split in REQUIRED_SPLITS:
            missing.append(os.path.join(split, ANNOTATION_FILE))

    if missing:
        raise SystemExit(
            f"ERROR: incomplete dataset structure in {dataset_dir}\n"
            f"  missing: {', '.join(missing)}\n"
            f"  expected a COCO structure:\n"
            f"    {dataset_dir}/train/{ANNOTATION_FILE}\n"
            f"    {dataset_dir}/valid/{ANNOTATION_FILE}\n"
            f"    {dataset_dir}/test/{ANNOTATION_FILE}   (optional)")

    return found


def validate_class_contract(splits):
    """
    Check the dataset classes honour the contract ball=1, dot=2, hand=3.

    Ids are enforced strictly: if any is missing, training stops. Names are only
    a consistency check and a mismatch is a warning, because only whoever did
    the labelling knows what an id really means.

    Raises:
        SystemExit: if the expected ids are not all present, or if the splits
            declare inconsistent categories.
    """
    reference = None
    reference_split = None

    for split, ann_path in splits:
        categories = load_categories(ann_path)
        by_id = {c["id"]: c.get("name", "") for c in categories}

        missing = [cid for cid in EXPECTED_CLASSES if cid not in by_id]
        if missing:
            listed = "\n".join(f"    id={c.get('id')}  name={c.get('name')}"
                               for c in categories) or "    (no categories)"
            raise SystemExit(
                f"ERROR: the class contract is violated in '{split}'.\n"
                f"  Missing ids: {missing}\n"
                f"  Categories found:\n{listed}\n\n"
                f"  The inference pipeline (spindoctor/config.py) expects:\n"
                f"    id=1 -> ball, id=2 -> dot, id=3 -> hand\n"
                f"  A model trained with different ids is NOT compatible with\n"
                f"  inference: detections would be read as the wrong classes\n"
                f"  with no visible error.\n"
                f"  Fix the labels and re-export the dataset.")

        for cid, expected_name in EXPECTED_CLASSES.items():
            actual = str(by_id[cid]).strip().lower()
            if actual != expected_name:
                print(f"  WARNING [{split}]: id={cid} is named '{by_id[cid]}' "
                      f"instead of '{expected_name}'. The ids are right, but "
                      f"check the meaning is what you expect.")

        current = {cid: by_id[cid] for cid in sorted(EXPECTED_CLASSES)}
        if reference is None:
            reference, reference_split = current, split
        elif current != reference:
            raise SystemExit(
                f"ERROR: the splits declare different categories.\n"
                f"  '{reference_split}': {reference}\n"
                f"  '{split}': {current}\n"
                f"  Re-export the dataset with a consistent set of classes.")

    return reference


def describe_dataset(splits):
    """Print a per-split summary of images and annotations."""
    print("Dataset:")
    for split, ann_path in splits:
        with open(ann_path, encoding="utf-8") as f:
            data = json.load(f)
        counts = {}
        for ann in data.get("annotations", []):
            counts[ann["category_id"]] = counts.get(ann["category_id"], 0) + 1
        per_class = ", ".join(
            f"{EXPECTED_CLASSES.get(cid, f'id={cid}')}={n}"
            for cid, n in sorted(counts.items()) if cid in EXPECTED_CLASSES)
        print(f"  {split:6s} {len(data.get('images', [])):5d} images, "
              f"{len(data.get('annotations', [])):5d} boxes  ({per_class})")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RF-DETR-Medium on a COCO ball/dots/hands dataset.")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR,
                        help=f"COCO dataset with train/valid/test (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Folder for checkpoints and logs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"],
                        help=f"Epochs (default: {DEFAULTS['epochs']}, tune it)")
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"],
                        help=f"Batch size (default: {DEFAULTS['batch_size']})")
    parser.add_argument("--grad-accum-steps", type=int, default=DEFAULTS["grad_accum_steps"],
                        help=f"Gradient accumulation steps; effective batch = "
                             f"batch_size x this (default: {DEFAULTS['grad_accum_steps']})")
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"],
                        help=f"Learning rate (default: {DEFAULTS['lr']})")
    parser.add_argument("--lr-encoder", type=float, default=DEFAULTS["lr_encoder"],
                        help=f"Encoder learning rate (default: {DEFAULTS['lr_encoder']})")
    parser.add_argument("--resolution", type=int, default=DEFAULTS["resolution"],
                        help=f"Training resolution (default: {DEFAULTS['resolution']}, "
                             f"RF-DETR-Medium's native)")
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"],
                        help=f"Dataloader workers (default: {DEFAULTS['num_workers']})")
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"],
                        help=f"Weight decay (default: {DEFAULTS['weight_decay']})")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Stop training when validation stops improving")
    parser.add_argument("--early-stopping-patience", type=int,
                        default=DEFAULTS["early_stopping_patience"],
                        help=f"Early-stopping patience in epochs "
                             f"(default: {DEFAULTS['early_stopping_patience']})")
    parser.add_argument("--device", type=str, default=None,
                        help="Training device (default: the library's, cuda)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only run the validations and print the config, "
                             "without training")
    args = parser.parse_args()

    # --- Validate before loading anything heavy ---
    print(f"Validating the dataset in {args.dataset_dir}")
    splits = validate_dataset_structure(args.dataset_dir)
    print(f"  splits found: {', '.join(name for name, _ in splits)}")

    contract = validate_class_contract(splits)
    print(f"  class contract honoured: "
          f"{', '.join(f'{k}={v}' for k, v in contract.items())}")

    describe_dataset(splits)

    model_config = {"resolution": args.resolution}
    if args.device:
        model_config["device"] = args.device

    train_config = {
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "num_workers": args.num_workers,
        "weight_decay": args.weight_decay,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
    }

    print("\nModel configuration:")
    for key, value in model_config.items():
        print(f"  {key:24s} {value}")
    print("Training configuration:")
    for key, value in train_config.items():
        print(f"  {key:24s} {value}")
    print(f"  effective batch          {args.batch_size * args.grad_accum_steps}")

    if args.dry_run:
        print("\n--dry-run: validations passed, no training started.")
        return

    # Imported here rather than at the top on purpose, so --help, --dry-run and
    # the validations work without downloaded weights or an available GPU.
    from rfdetr import RFDETRMedium

    print(f"\nLoading RF-DETR-Medium (fine-tuning from the library's "
          f"COCO-pretrained checkpoint)...")
    model = RFDETRMedium(**model_config)

    print(f"Starting training. Checkpoints and logs in {args.output_dir}\n")
    model.train(**train_config)

    print(f"\nTraining complete. The weights are in {args.output_dir}")
    print("To use them for inference, point spindoctor at them:")
    print(f"  copy the checkpoint to rf-detr/weights.pt, or change")
    print(f"  DEFAULT_WEIGHTS in spindoctor/config.py")


if __name__ == "__main__":
    main()
