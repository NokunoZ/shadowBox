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

## Decisions to confirm

These are places where I am not sure our plans match. See each folder's `noted.md`
for the detail.

1. **"CNN" over what?** `claude.md` says CNN. Since stage 2 eats keypoints, not
   pixels, the CNN would be a **1D temporal CNN** sliding over the time axis of the
   `T x K x 2` sequence — not an image CNN. That is a good fit and still a CNN, but
   it is a different shape of model than "CNN" usually implies. Confirm this is what
   you meant.
2. **No `Idle` class.** Six classes are all active moves. On a live webcam the model
   is forced to pick one of them every frame even when the person is just standing
   there. We probably need a seventh `Idle` / `Guard` class.
3. **Accuracy needs labels.** `getTeststatistic.py` measures "percentage of correct
   guesses", but `TestRun.py` reads a webcam, which has no ground truth. Accuracy
   has to be measured against `trainDataSet/testData/`, not the live feed.
4. **No shared pose module yet.** Both training and live inference need the exact
   same "video -> keypoints" code. Right now there is nowhere for it to live.
   Proposal: `code/common/`.
5. **Environment is empty.** `.venv` has no torch and no ultralytics. We need a
   `requirements.txt` with pinned versions before any of this runs.
6. **Folder naming convention.** Now that the folder name *is* the class declaration,
   freeform names let `body-shot` and `body_shot` become two separate classes.
   Suggest lowercase with underscores, enforced by `classType.py`. Worth deciding
   before the first clip lands, since it means renaming the six existing folders.
