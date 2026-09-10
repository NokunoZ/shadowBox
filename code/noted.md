# code/ — all runnable python

Split by *when* the code runs, not by what it imports.

| folder | when it runs |
|---|---|
| `trainingCode/` | offline, once per model version |
| `runTest/` | after training, to demo and to score |
| `unitTest/` | any time, on every module |

## Missing piece: shared code

`trainingCode/` and `runTest/` both need to turn video into keypoints, and they must
do it **identically** — same YOLO weights, same normalisation, same window length.
If the two drift apart the model sees different inputs at train and at run time and
accuracy quietly collapses. That bug is invisible and expensive.

Proposal — add `code/common/`:

- `pose.py` — load YOLO26-pose once, run a clip or a frame, return keypoints
- `features.py` — normalise keypoints (centre on hips, scale by torso length),
  build the fixed-length window the classifier expects
- `model.py` — the classifier architecture definition, imported by both training
  and inference so there is exactly one copy. Takes `num_classes` as an argument; no
  class count is baked into it, so a new folder in `dataSet/` never reaches this file
- `labels.py` — thin wrapper over `trainDataSet/classType.py`, and the **only** place
  that scans the dataset folders. Classes are declared by folders, so the scan has to
  happen somewhere; having it happen in exactly one place is what stops training and
  inference from disagreeing about what index 3 means. Inference does not call it —
  it reads the class list out of the checkpoint (see `modelholder/noted.md`).

Nothing here duplicates logic; every module has one home.

## Import note

These folders are not a package yet (no `__init__.py`, no `pyproject.toml`), so
`import common.pose` will fail depending on where you launch python from. Either add
`__init__.py` files and run with `python -m`, or make the project pip-installable.
Worth settling before the first real import is written.
