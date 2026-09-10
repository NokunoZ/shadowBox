# shadowBox — project plan

Recognise boxing actions from video: **Jab, Cross, Hook, Uppercut, block, dodge** —
and whatever else gets added later.

**Standing requirement: the class set is open.** A new move is added by creating one
folder under `trainDataSet/dataSet/` and filling it with clips. No source file holds
a list of classes or a class count; every module derives them. Full rules, and the
index-shift trap that comes with it, in `trainDataSet/dataSet/noted.md`.

## The pipeline (two stages)

```
video clip / webcam
        |
        v
[ stage 1 ]  YOLO26-pose                                  <- pretrained, NOT trained by us
             modelholder/runningVersion/yolo26s-pose.pt
        |
        v
keypoint sequence:  T frames x K joints x (x, y, confidence)
        |
        v
[ stage 2 ]  PyTorch classifier                           <- this is what we train
             modelholder/runningVersion/classifier.pt
        |
        v
class label + confidence
```

Stage 1 is frozen. Stage 2 is the only thing that learns. This is deliberate: pose
estimation is a solved problem, and training on 34 numbers per frame instead of raw
pixels means we need hundreds of clips, not hundreds of thousands.

## Folder map

| folder | role |
|---|---|
| `trainDataSet/dataSet/` | raw video clips, one folder per class — **add a folder, get a class** |
| `trainDataSet/` | class list + train/test splitter |
| `code/trainingCode/` | trains the stage-2 classifier |
| `code/runTest/` | live webcam demo + accuracy report |
| `code/unitTest/` | per-module tests |
| `modelholder/modelVersion/` | every trained classifier, one file per run, never overwritten |
| `modelholder/runningVersion/` | the weights that actually run — pose model + the promoted classifier |

## Two model folders, one rule

Training writes a new `classifier_v<N>.pt` into `modelholder/modelVersion/` and stops
there. `code/runTest/` loads `modelholder/runningVersion/classifier.pt` and looks
nowhere else. Moving a version from the first to the second is a manual copy:

```
cp modelholder/modelVersion/classifier_v3.pt modelholder/runningVersion/classifier.pt
```

So a training run can never break the demo, "which model produced this score?" has
one answer, and a regression is one `cp` from fixed because the old file was never
overwritten. Details and the checkpoint contract: `modelholder/noted.md`.

## Setup

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     --extra-index-url https://download.pytorch.org/whl/cu128
```

The cu128 index is for the dev machine's RTX 5060 (Blackwell needs CUDA 12.8 wheels);
swap `cu128` for `cpu` elsewhere. Nothing in the project requires a GPU.

## Running it

```
python trainDataSet/classType.py          # what classes exist, and how many clips each
python trainDataSet/dataSpliter.py        # regenerate trainData/ and testData/
python code/trainingCode/train.py         # -> modelholder/modelVersion/classifier_v<N>.pt
cp modelholder/modelVersion/classifier_v1.pt modelholder/runningVersion/classifier.pt
python code/runTest/getTeststatistic.py   # -> code/runTest/confident.csv
python code/runTest/TestRun.py            # live webcam demo
python -m pytest                          # 75 tests, no GPU, no weights, no downloads
```

## Decisions — settled

1. **"CNN" over what?** A **1D temporal CNN**, sliding over the time axis of the
   `T x 51` sequence. Stage 2 never sees pixels, so an image CNN has nothing to look
   at; convolving over time is what makes "how fast the wrist travels, when the hip
   rotates" learnable. Still a CNN, as `claude.md` asks for — a different axis.
   Built in `code/common/model.py`.
2. **`Idle` class** — added as `trainDataSet/dataSet/idle/`. Without it the model
   must claim a punch is happening while the person stands still. Costing one folder
   and no code change is the open-class rule working as intended.
3. **Accuracy comes from labels.** `getTeststatistic.py` is an offline pass over
   `testData/`, not a consumer of `TestRun.py`. The webcam has no ground truth.
4. **Shared code** — `code/common/` exists: `pose`, `features`, `model`, `labels`,
   `checkpoint`, `paths`. Training and inference share one preprocessing path.
5. **Environment** — `requirements.txt`, pinned, installed and verified on CUDA.
6. **Naming convention** — lowercase with underscores, enforced by `classType.py`,
   which *rejects* anything else. The four capitalised folders were renamed while
   they were still empty.

## What is left

**Footage.** Every class folder is empty, so `classType.py` correctly reports zero
classes and nothing downstream can run. That is the one blocking item; the code above
was exercised end to end against synthetic keypoints and is waiting on clips.

Two numbers should be chosen from real footage rather than kept at their defaults:

- **window length T** (default 32). `train.py` prints the real clip-length
  distribution and how many clips get stretched or thinned, which is the number to
  pick from — a jab is fast, a dodge is slower.
- **test ratio** (default 0.2), once it is clear how many clips per class exist.
