# code/trainingCode/ — trains the stage-2 classifier

## train.py

Stated intent (from the file header): *train the classification deep learning model;
if a model already exists, edit and replace it.*

### Contract

- **in** — `trainDataSet/trainData/` (and a validation split, see below)
- **out** — a new versioned checkpoint in `modelholder/modelVersion/`, e.g. `classifier_v4.pt`
- **never** — this script does not train YOLO. Stage 1 is frozen pretrained weights.
- **never** — this script does not write to `modelholder/runningVersion/`. Training must
  not be able to change what the demo and the score are using; promoting is a separate,
  deliberate copy (see `modelholder/noted.md`).

### Steps

1. Read the class list from `trainDataSet/classType.py` so labels stay in one place.
   It is discovered from the `dataSet/` folders, so **nothing in this script may
   assume how many classes there are** — the final layer is `nn.Linear(h, len(classes))`,
   never `nn.Linear(h, 6)`, and the same goes for any class-weight vector, metric
   array or plot. Adding a boxing move should require editing zero lines here.
2. For every clip, run YOLO26-pose and cache the keypoints to `.npy`.
   Do this **once**, not every epoch — pose estimation is far slower than the
   classifier and re-running it each epoch would dominate the training time.
3. Normalise: centre on the hip midpoint, scale by torso length. Without this the
   model learns "person stands near the left of the frame" instead of "this is a jab".
4. Sample a fixed-length window of T frames per clip.
5. Train the 1D temporal CNN. Track train and validation loss separately.
6. Save weights **plus** the class-index-to-name map, the preprocessing settings
   (T, normalisation, keypoint format) and the version tag in the same checkpoint,
   written to `modelholder/modelVersion/classifier_v<N>.pt`. The class map is what
   makes the checkpoint survive someone adding a folder later — without it, a model
   trained on six classes is unreadable the moment the folder count changes.

### "replace the existing model" — careful, and the answer is: don't

Overwriting in place means a bad training run destroys a good model with no way back.
The folder layout settles this instead:

- **Write** a new numbered file into `modelholder/modelVersion/` — never overwrite an
  existing one. The next number is `max(existing) + 1`, discovered by scanning that
  folder, so nothing has to be bumped by hand. If the target name somehow already
  exists, stop rather than replace.
- **Do not touch** `modelholder/runningVersion/`. Finishing a training run changes
  nothing about what runs; the previous model keeps serving the demo and the score
  until someone promotes the new one on purpose.
- **Promote** by copying the archive file over `runningVersion/classifier.pt`, once
  the new version has actually been scored against the old one. Rollback is the same
  copy with the previous version.

So the original intent — *edit and replace the existing model* — becomes: add a
version, then decide. The replacing step is one `cp`, and it is reversible because
the archive still holds both files.

Stamp the version tag and the archive filename into the checkpoint itself, since the
promoted copy is renamed to `classifier.pt` and would otherwise be anonymous.

## Adding a class means retraining from scratch

Classes arrive as folders (`trainDataSet/dataSet/noted.md`), and a new folder changes
two things at once: the output layer grows, and every class sorted after the new one
shifts index. So:

- **The old checkpoint cannot be fine-tuned into the new one** by loading its
  `state_dict` — the last layer's shape no longer matches, and the surviving weights
  are wired to the old ordering. Either train fresh, or transplant the old weights
  deliberately by *name*, matching old class names to new indices. `--resume` reads
  from `modelholder/modelVersion/`, by explicit filename: resuming from "whatever is
  running" would make a run unreproducible the next time someone promotes something.
- **Fail loudly on a mismatch.** If `--resume` is given a checkpoint whose class list
  differs from the current scan, stop with a message naming the added and removed
  classes. Silently resuming here is the exact bug that produces a model which is
  confidently wrong about every label.
- **Log the class list at the start of every run**, with the clip count per class.
  It is the cheapest way to catch a stray folder or a typo'd name becoming a class.

## Open questions

- **Validation set.** `dataSpliter.py` makes train and test only. Tuning epochs and
  learning rate against the test set silently inflates the reported score. We want
  train / val / test, or at minimum a val slice carved out of train.
- **Window length T.** A jab is fast, a dodge is slower. One fixed T for all six
  classes is a real constraint — needs a number chosen from the actual clip lengths
  once we have footage.
- **Class imbalance.** If one class ends up with far fewer clips, use a weighted loss.
  Compute the weights from the discovered counts, not from a table written by hand,
  or the table goes stale the first time a folder is added.
- **A newly added class hurting the old ones.** More classes means more ways to be
  wrong, so per-class accuracy on the previous classes should be compared before and
  after. That is the reason `modelVersion/` keeps everything: promote the new version,
  score it, promote the old one back, score it, compare. If `idle/` costs five points
  on `Hook`, the previous model is one `cp` away.
