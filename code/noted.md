# code/ — all runnable python

Split by *when* the code runs, not by what it imports.

| folder | when it runs | which weights it touches |
|---|---|---|
| `trainingCode/` | offline, once per model version | writes `modelholder/modelVersion/`, reads `runningVersion/yolo26s-pose.pt` |
| `runTest/` | after training, to demo and to score | reads `modelholder/runningVersion/` only |
| `unitTest/` | any time, on every module | none — tests must not need weights |

Training writes to the archive, running reads from the promoted copy, and the two
never overlap: a training run cannot change what the demo is doing, and a demo cannot
be pointed at a half-written file. Promotion between them is a manual copy, described
in `modelholder/noted.md`.

## Shared code — `code/common/`, built

`trainingCode/` and `runTest/` both need to turn video into keypoints, and they must
do it **identically** — same YOLO weights, same normalisation, same window length.
If the two drift apart the model sees different inputs at train and at run time and
accuracy quietly collapses. That bug is invisible and expensive.

Every step of that path lives in `code/common/` exactly once:

- `pose.py` — load YOLO26-pose once from `modelholder/runningVersion/yolo26s-pose.pt`,
  run a clip or a frame, return keypoints. One place holds that path, so training and
  inference cannot end up on different pose weights
- `checkpoint.py` — the one place that knows the `modelVersion/` and `runningVersion/`
  paths: work out the next version number when saving, load
  `runningVersion/classifier.pt` when running, and refuse a checkpoint whose class
  list or preprocessing settings do not match what the caller expects. Without it
  those paths get retyped in three scripts and drift
- `features.py` — normalise keypoints (centre on hips, scale by torso length),
  build the fixed-length window the classifier expects
- `model.py` — the classifier architecture definition, imported by both training
  and inference so there is exactly one copy. Takes `num_classes` as an argument; no
  class count is baked into it, so a new folder in `dataSet/` never reaches this file
- `paths.py` — every path the project uses, including `modelVersion/` and
  `runningVersion/`. Retyping those in three scripts is how training ends up writing
  where inference reads
- `labels.py` — thin wrapper over `trainDataSet/classType.py`, and the **only** place
  that scans the dataset folders. Classes are declared by folders, so the scan has to
  happen somewhere; having it happen in exactly one place is what stops training and
  inference from disagreeing about what index 3 means. Inference does not call it —
  it reads the class list out of the checkpoint (see `modelholder/noted.md`).

Nothing here duplicates logic; every module has one home. In particular
`features.prepare()` is the single normalise-then-window path — training and the live
demo both call it, so they cannot drift apart.

## Import note — settled

`code/` cannot become a package: `import code` would shadow the standard library's
`code` module, and that breaks tools in ways that are hard to trace. The project is
therefore not pip-installed. Instead:

- `code/common/` is a package (`__init__.py`) and every entry script puts `code/` on
  `sys.path` before importing it, so the scripts run from anywhere
- `pytest.ini` sets `pythonpath = code` for the same reason
- `trainDataSet/classType.py` is not importable normally either (that folder is not a
  package), so `common/labels.py` loads it by path — in one place, once

Entry points are plain scripts: `python code/trainingCode/train.py`,
`python code/runTest/TestRun.py`, `python trainDataSet/dataSpliter.py`.
