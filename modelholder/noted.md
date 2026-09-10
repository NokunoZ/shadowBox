# modelholder/ — all model weights

Original note: *this folder holds the models — the YOLO pose model for vertices
detection, and the deep learning model that takes data from YOLO-pose and trains on
it to classify the pose.*

That is the plan, confirmed. Two models, two very different lifecycles:

| file | stage | trained by us? |
|---|---|---|
| `yolo26s-pose.pt` | 1 — video to skeleton vertices | **no**, pretrained, frozen |
| classifier weights | 2 — vertices to boxing move | **yes**, by `code/trainingCode/train.py` |

## Weights are gitignored

`*.pt` is in `.gitignore`. A 24 MB binary per training run would bloat the repo fast,
and git cannot diff or merge them. This `noted.md` is what keeps the folder in git.

Consequence: **a fresh clone has no weights.** Whoever sets up needs to re-download
the pose model. Worth recording the exact source and version here once confirmed —
`yolo26s-pose.pt` is currently just sitting in the folder with no record of where it
came from, and "which weights was that model trained against" becomes unanswerable
later.

## Save the checkpoint, not just the tensors

A bare `state_dict` is not enough to run the model. Each stage-2 checkpoint should
carry, in the same file:

- the weights
- the class-index-to-name map
- the preprocessing settings (window length T, normalisation, keypoint layout)
- the architecture version

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

Prefer versioned names (`classifier_v3.pt`) over overwriting one file, so a bad
training run cannot destroy the good model. See `code/trainingCode/noted.md`.

Version bumps on a class change too, not just on a better score — `classifier_v4.pt`
trained on seven classes and `classifier_v3.pt` trained on six are different models,
not an upgrade path, and keeping both is what makes "did the new class cost us
anything?" answerable.
