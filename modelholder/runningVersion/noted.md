# modelholder/runningVersion/ — the models that actually run

Everything `code/runTest/` loads lives here, at fixed paths:

| file | stage | where it comes from |
|---|---|---|
| `yolo26s-pose.pt` | 1 — video to skeleton vertices | downloaded, pretrained, frozen |
| `classifier.pt` | 2 — vertices to boxing move | **a copy** of one file from `../modelVersion/` |

## Fixed names on purpose

`TestRun.py` and `getTeststatistic.py` hold these two paths as constants. They do not
scan `../modelVersion/`, do not sort by date, and do not pick "the latest" — if two
scripts each chose their own model, the demo and the score could be measuring
different models while appearing to agree.

Which version is in `classifier.pt` is answered by reading the checkpoint, which
stores its own version tag and the archive filename it was copied from. The filename
here stays constant so promotion never means editing code.

## Promoting and rolling back

```
cp modelholder/modelVersion/classifier_v3.pt modelholder/runningVersion/classifier.pt
```

**Copy, never move** — a move deletes the archived version. Rollback is the same
command with an older file; the model being replaced is not lost, because it was
never only here.

Keep exactly one classifier in this folder. A loader that finds two should raise, not
guess.

## The pose model

`yolo26s-pose.pt` is stage 1 and we never train it, but it lives here because this
folder is "what runs", not "what we produced". It is the one file here that is not a
copy of anything in `../modelVersion/`.

Its provenance is currently unrecorded — it arrived in the project with no note of
source or version. Fill this in once confirmed:

- source / download URL: _unknown, needs confirming_
- version or release tag: _unknown_
- SHA256: _unrecorded_

Without it, "which pose weights produced these keypoints?" is unanswerable, and every
cached `.npy` and every trained classifier downstream inherits that gap.

## Gitignored

`*.pt` is gitignored, so a fresh clone has this folder empty and cannot run anything:
the pose model has to be re-downloaded and a classifier promoted from
`../modelVersion/`. This `noted.md` is what keeps the folder in git.
