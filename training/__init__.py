"""Data preparation and RF-DETR detector training.

This part of the repo is SEPARATE from the inference pipeline (the
`spindoctor/` package): it builds the dataset and trains the weights that
inference later consumes.

The scripts are invoked as modules from the repo root, so that
`import spindoctor` works without touching sys.path:

    python -m training.extract_frames --help
    python -m training.preprocess_roboflow_replica --help
    python -m training.train_rfdetr --help

See training/README.md for the full workflow and the class contract.
"""
