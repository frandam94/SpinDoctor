# Data preparation and detector training

This directory holds the **upstream** half of the pipeline: how the dataset is
built and how the detector is trained, the one that inference later consumes. It
is kept separate from the `spindoctor/` package, which only measures spin from
already-trained weights.

The scripts are invoked **as modules, from the repo root**, so that
`import spindoctor` works without touching `sys.path`:

```bash
python -m training.extract_frames --help
python -m training.preprocess_roboflow_replica --help
python -m training.train_rfdetr --help
```

---

## A note on naming, to avoid confusion

The Roboflow project the dataset comes from is called
**`Basketball_BallDots_YOLO`**. The "YOLO" in that name is a historical leftover:
the project started with the idea of using YOLO, and the name was never changed.

In this repo:

- the dataset is exported in **COCO** format, not YOLO;
- the model is **RF-DETR-Medium**, not YOLO.

No part of this project uses YOLO. If you download the dataset from the original
Roboflow project, expect a name that does not describe what is inside.

---

## Class contract

**This is the single most important constraint in this section.** The class IDs
in the training dataset must match exactly the ones the inference pipeline
expects:

| id | name   | constant in `spindoctor/config.py` |
|----|--------|------------------------------------|
| 1  | `ball` | `CLASS_BALL = 1` |
| 2  | `dot`  | `CLASS_DOT = 2` |
| 3  | `hand` | `CLASS_HAND = 3` |

Roboflow's COCO exports also include an `id=0` category, which is only a
supercategory placeholder: that is normal and is ignored.

**If you annotate with different IDs, the trained model will not be compatible
with inference.** The failure is particularly nasty because it is silent: no
error is raised, `spindoctor` will simply read dots as balls and hands as dots,
and the spin results will be nonsense with nothing to flag it.

That is why `train_rfdetr.py` **checks the contract and refuses to start** if
IDs 1, 2 and 3 are not all present, or if the splits declare mutually
inconsistent classes. If the IDs are right but the names differ (`Palla` instead
of `ball`) it emits a warning and proceeds: IDs can be checked mechanically,
meaning cannot.

---

## The full workflow

```
video  ──▶  extract frames  ──▶  annotate by hand  ──▶  export COCO
                                  (ball/dot/hand)           │
                                                            ▼
                                              [optional] preprocessing
                                                            │
                                                            ▼
spindoctor inference  ◀──  weights  ◀──  RF-DETR training  ◀┘
```

### 1. Extract frames from the videos

```bash
python -m training.extract_frames --input ./videos --output ./data/raw_frames --every 5
```

Saves one frame every `--every` (default 5: at 60fps that is 12 images per
second). The frames are **raw**, with no boxes drawn on them.

> Note: `spindoctor.detection.debug_all_frames` also produces frames, but with
> detections already drawn on top: those are for debugging, **not** for
> annotation.

Useful options: `--max-per-video` to cap how many frames to take from each clip,
`--quality` for JPEG quality.

### 2. Annotate by hand

Upload the frames to Roboflow (or any other annotation tool) and draw the boxes
for the three classes, **respecting the IDs in the table above**.

This step is manual and cannot be automated: it is the real work.

### 3. Export to COCO

Export in **COCO Detection** format, with this structure:

```
data/basketball_coco/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

`train` and `valid` are mandatory, `test` is recommended.

### 4. Preprocessing — only if you export WITHOUT transformations

```bash
python -m training.preprocess_roboflow_replica \
    --input ./data/raw_labeled --output ./data/basketball_coco --versions 5
```

> **⚠ Do not run this script on an already-processed Roboflow export.**
> If you configured preprocessing and augmentation in Roboflow, the export you
> download already contains them. Running them again on top means rotating
> twice, blurring twice and brightening twice the same images: the dataset gets
> worse, not better.
>
> The script detects this on its own and **stops**, recognising the export's
> `README.roboflow.txt` or the `.rf.` hash in filenames. The block can be
> overridden with `--force`, but think twice.

So it is only needed in one of two cases: you export from Roboflow **without**
configured transformations, or you annotate with another tool and want to apply
the same transformations.

What it does, and how faithfully:

**Preprocessing** (on all images, all splits)
- auto-orient with EXIF orientation stripping
- resize to 576×576 (stretch, not letterbox)

**Augmentation** (by default on `train` only, as Roboflow does; configurable
with `--augment-splits`)

| Roboflow | Albumentations | fidelity |
|---|---|---|
| horizontal flip 50% | `HorizontalFlip(p=0.5)` | equivalent |
| rotation ±5° | `Rotate(limit=5)` | equivalent, borders may differ |
| brightness ±20% | `RandomBrightnessContrast(brightness_limit=0.2)` | **approximate** |
| exposure ±15% | `RandomGamma(gamma_limit=(85,115))` | **approximate** |
| blur 0–0.5px | `GaussianBlur(sigma_limit=(0.0,0.5))` | equivalent |

`--versions N` (default 5) controls how many versions to produce per source
image: version `_v0` gets preprocessing only, the following ones also get the
augmentations.

**This is a replica, not a copy.** The libraries, the internal order of
operations and the seeds Roboflow uses are not public. On the same input, this
script and Roboflow produce statistically comparable datasets, not identical
files. The two rows marked "approximate" are the least certain: Roboflow's
"brightness" and "exposure" have no documented 1:1 equivalent.

Why 576×576: it is RF-DETR-Medium's native resolution
(`rfdetr/config.py`, `RFDETRMediumConfig.resolution = 576`).

### 5. Training

```bash
python -m training.train_rfdetr --dry-run     # validate without training
python -m training.train_rfdetr               # train with the defaults
```

`--dry-run` runs every validation (dataset structure, class contract, per-class
counts) and prints the configuration without touching the GPU. Always worth
starting there.

Checkpoints and logs land in `runs/train/`, which is gitignored. Training logs
to TensorBoard by default: `tensorboard --logdir ./runs`.

### 6. Use the weights for inference

Copy the checkpoint to `rf-detr/weights.pt`, or change `DEFAULT_WEIGHTS` in
`spindoctor/config.py`. Then:

```bash
python main.py --video shot1.MOV --output ./output
```

---

## How the distributed model was trained

The model the pipeline uses is **RF-DETR-Medium, fine-tuned** (not trained from
scratch) on a custom dataset of ball, dots and hands.

Training was run on the **Roboflow platform, from its web interface**, with no
code. The preprocessing and augmentations described above were likewise applied
by Roboflow at export time.

Dataset: 1109 images (1025 train / 42 valid / 42 test), the result of 5
augmented versions per source image in the train split.

Metrics from that training run:

| metric | value |
|---|---|
| mAP@50 | 75.6% |
| Precision | 85.4% |
| Recall | 75.4% |
| F1 | 79.6% |

## How to reproduce the training in code

`train_rfdetr.py` is a **code reproduction of an equivalent training run**,
written to make the process reproducible and inspectable. **It is not the script
that produced the distributed weights** — that script does not exist: that
training was done by clicking buttons.

Consequently, **the metrics in the previous section are not a promise of what
you will get by running this script.** They come from a different training run,
on a different platform, with hyperparameters that are not public.

The default hyperparameters are a stable configuration chosen as a reasonable
starting point for a dataset of roughly 1100 images, **to be tuned for your own
case**:

| parameter | default | origin |
|---|---|---|
| `--epochs` | 50 | our choice (the library uses 100) |
| `--batch-size` | 4 | library default |
| `--grad-accum-steps` | 4 | library default (effective batch 16) |
| `--lr` | 1e-4 | library default |
| `--lr-encoder` | 1.5e-4 | library default |
| `--resolution` | 576 | RF-DETR-Medium's native resolution |
| `--weight-decay` | 1e-4 | library default |

Fine-tuning starts from the COCO-pretrained checkpoint the library downloads
automatically (`rf-detr-medium.pth`).

### Testing status of the scripts

Honesty about the level of verification, which is not uniform:

| script | status |
|---|---|
| `extract_frames.py` | run on real videos, output verified |
| `preprocess_roboflow_replica.py` | COCO logic and guard verified on a real dataset; **the Albumentations chain has never been executed** (library not installable in the development environment) |
| `train_rfdetr.py` | validations executed and verified; **training has never been run end to end** (no GPU available) |

Before launching a long training run, it is worth trying a couple of epochs
(`--epochs 2`) first, and for preprocessing, trying it on a handful of images.

---

## Data and weights are not in the repo

Neither the dataset nor the weights are versioned: `data/`, `runs/` and `*.pt`
are in `.gitignore`. Anyone cloning the repo starts from **their own** videos and
**their own** labels, following the workflow above.

If you need the Roboflow API key (only required if you download the dataset via
the SDK rather than by hand from the UI), copy `.env.example` to `.env` and fill
it in. The `.env` file is excluded from git and no script contains credentials.

## Dependencies

```bash
pip install -r requirements.txt -r training/requirements-training.txt
```

`albumentations` is only needed for preprocessing: if you only want to run
inference, the root `requirements.txt` is enough.
