# code/unitTest/ — run tests on each module

Original note: *use this folder to run test on each module.*

## What is worth testing

The model's accuracy is not a unit test — that is `getTeststatistic.py`'s job. What
belongs here is the plumbing around the model, which is where the silent bugs are:

- **`classType.py`** — returns the classes in a stable, sorted order. If this order
  ever changes between training and inference, every prediction is mislabelled and
  nothing crashes. This is the highest-value test in the project. Since a class is
  just a folder, the discovery itself needs covering too, against a temp directory:
  - a folder with clips in it is found; an empty one, a `_scratch/` and a `.hidden/`
    are not
  - adding a folder mid-alphabet renumbers the classes after it — assert this
    explicitly, so the behaviour is recorded rather than discovered later by a model
    that mislabels everything
  - a name that breaks the naming convention is rejected, not quietly accepted as a
    near-duplicate class
- **`dataSpliter.py`** — split ratio is respected, and no clip appears in both train
  and test. Also: a class folder added to the input appears in both output folders
  with no code change, and a re-run after removing a class leaves no stale folder
  behind for training to pick up.
- **normalisation** — feed in a known skeleton, assert the output is centred and
  scaled as expected; feed the same pose shifted and resized, assert it produces the
  same output.
- **windowing** — a clip shorter than T and a clip longer than T both come out at
  length T.
- **checkpoint round-trip** — save a model, load it, get identical outputs, and get
  back the same class list in the same order.
- **class-count mismatch** — loading a checkpoint whose class list disagrees with the
  current folders raises, rather than running with labels shifted by one. Build the
  model for N classes, save, then load against N+1 and assert it fails.
- **model builds for any N** — construct the classifier with 2, 6 and 20 classes and
  assert the output width follows. Catches a hardcoded `6` in the architecture.

## Suggested setup

`pytest`, files named `test_*.py`. Keep a couple of tiny fixture clips (a second or
two each) checked in so tests do not depend on the full dataset — note that `.mp4`
is gitignored, so a fixture would need an explicit exception in `.gitignore`, or
better, generate synthetic keypoint arrays in code and skip video entirely.

Tests must not need a GPU and must not download weights.
