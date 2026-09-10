# modelholder/modelVersion/ — the classifier archive

Every stage-2 model we ever train lands here, one file per training run:

```
classifier_v1.pt  classifier_v2.pt  classifier_v3.pt ...
```

## Rules

- **Written by `code/trainingCode/train.py`, by nothing else.**
- **Never overwritten.** A file here is immutable once written. If a run would
  collide with an existing name, the run stops rather than replacing it — that is the
  whole reason this folder is separate from `runningVersion/`.
- **Never read at run time.** `TestRun.py` and `getTeststatistic.py` load
  `../runningVersion/classifier.pt`. Nothing here is picked up automatically, so a
  finished training run changes nothing about what the demo does until someone
  promotes it on purpose.

## Version numbers

The next number is `max(existing) + 1`, discovered by scanning this folder — not read
from a file that someone has to remember to bump, and not a timestamp, which sorts
badly and says nothing about order of intent.

A bump is not a claim that the model is better. Bump on:

- a better (or worse) score
- **a change to the class list** — `classifier_v4.pt` on seven classes and
  `classifier_v3.pt` on six are different models, not an upgrade path
- a change to preprocessing (window length T, normalisation, keypoint layout) or to
  the architecture

Each of those makes the old file the only way to reproduce the old number, which is
why they all live here rather than replacing each other.

## Which one is any good?

The filename does not say. The checkpoint inside carries its class list, its
preprocessing settings and its own version tag (see `../noted.md`), and
`code/runTest/confident.csv` carries the score — of whichever version was in
`runningVersion/` when it was generated. Keep a line here per version once training
actually starts:

| file | classes | val acc | note |
|---|---|---|---|
| _(nothing trained yet — dataset is empty)_ | | | |

## Gitignored

`*.pt` is gitignored, so this folder is empty in a fresh clone and this `noted.md` is
what keeps it in git. The archive lives on one disk only — if these versions matter,
they need a backup that is not this repo.
