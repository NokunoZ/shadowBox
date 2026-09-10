# modelholder/ — all model weights

Original note: *this folder holds the models — the YOLO pose model for vertices
detection, and the deep learning model that takes data from YOLO-pose and trains on
it to classify the pose.*

That is the plan, confirmed. Two models, two very different lifecycles:

| file | stage | trained by us? |
|---|---|---|
| `yolo26s-pose.pt` | 1 — video to skeleton vertices | **no**, pretrained, frozen |
| classifier weights | 2 — vertices to boxing move | **yes**, by `code/trainingCode/train.py` |

## Two subfolders: archive and running

Weights are split by **role**, not by stage:

```
modelholder/
  modelVersion/     every trained classifier, one file per run, never overwritten
    classifier_v1.pt  classifier_v2.pt  classifier_v3.pt ...
  runningVersion/   what actually runs right now
    yolo26s-pose.pt   stage 1, frozen
    classifier.pt     a COPY of one file from modelVersion/
```

The rule that makes this worth having:

- **`code/trainingCode/` writes only to `modelVersion/`.** It never touches
  `runningVersion/`, so a training run cannot break the demo that is working.
- **`code/runTest/` reads only from `runningVersion/`.** It never scans
  `modelVersion/` and never picks "the newest" file by itself — a script choosing its
  own model means the score and the demo can silently disagree about which model they
  measured.

Promotion is therefore a deliberate, one-line act: copy the archive file you want
onto `runningVersion/classifier.pt`. Rollback is the same command with an older
version. Nothing in the code changes, no path is edited, and the losing model is
still in `modelVersion/` either way.

```
# promote
cp modelholder/modelVersion/classifier_v3.pt modelholder/runningVersion/classifier.pt
# rolled back in one line if v3 turns out worse
cp modelholder/modelVersion/classifier_v2.pt modelholder/runningVersion/classifier.pt
```

**Copy, never move.** A move empties the archive slot and the version is gone.

### Why the running file has a fixed name

`runningVersion/classifier.pt` is a constant path, so `TestRun.py` and
`getTeststatistic.py` hold no version string and need no `--model` flag for the
normal case. "Which version is that?" is answered by the checkpoint itself, which
stores its own version and the archive filename it was copied from (see below) — not
by the filename, which would otherwise have to be edited in two scripts on every
promotion.

Exactly one classifier lives in `runningVersion/` at a time. If a loader finds two,
that is an error worth raising rather than a guess worth making.

## Weights are gitignored

`*.pt` is in `.gitignore`. A 24 MB binary per training run would bloat the repo fast,
and git cannot diff or merge them. The `noted.md` in each folder is what keeps the
folder — and now both subfolders — in git.

Consequence: **a fresh clone has no weights, and both subfolders are empty.** Whoever
sets up needs to re-download the pose model into `runningVersion/`, and either train
or be handed a classifier. Worth recording the exact source and version of the pose
model in `runningVersion/noted.md` once confirmed — `yolo26s-pose.pt` is currently
just sitting there with no record of where it came from, and "which weights was that
model trained against" becomes unanswerable later.

## Save the checkpoint, not just the tensors

A bare `state_dict` is not enough to run the model. Each stage-2 checkpoint should
carry, in the same file:

- the weights
- the class-index-to-name map
- the preprocessing settings (window length T, normalisation, keypoint layout)
- the architecture version
- **its own version tag and archive filename** (`v3` / `classifier_v3.pt`) — this is
  what survives the copy into `runningVersion/classifier.pt`. Without it the running
  file is anonymous, and a result in `confident.csv` cannot be traced back to the
  archive file that produced it.

Without these, a checkpoint is unusable six months from now — you cannot tell what
input it expects or what its output index 3 means.

The class map matters more here than in most projects, because classes are declared
by folders in `trainDataSet/dataSet/` and that set is meant to grow. A checkpoint is
**bound to the class list it was trained against**: adding `idle/` renumbers every
class after it, so index 3 stops meaning what this file's weights think it means.
The stored list is what makes the checkpoint self-describing instead of dependent on
whatever the dataset folder happens to look like today.

So the loader, wherever it lives, must:

- take the class names from the checkpoint, never from a fresh folder scan
- build the model with `len(checkpoint["classes"])` output units
- raise, not warn, if a caller hands it a class list that differs from the stored one

## Naming

Archive files are versioned (`classifier_v3.pt`) and never overwritten, so a bad
training run cannot destroy the good model. See `code/trainingCode/noted.md` for how
the next number is chosen.

Version bumps on a class change too, not just on a better score — `classifier_v4.pt`
trained on seven classes and `classifier_v3.pt` trained on six are different models,
not an upgrade path, and keeping both in `modelVersion/` is what makes "did the new
class cost us anything?" answerable: promote one, score it, promote the other, score
it, compare.
