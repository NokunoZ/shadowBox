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

Provenance, read out of the checkpoint itself and recorded here so that "which pose
weights produced these keypoints?" stays answerable:

| | |
|---|---|
| file | `yolo26s-pose.pt` |
| size | 24,151,790 bytes |
| SHA256 | `a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646` |
| task | `pose` |
| keypoints | COCO-17, `(x, y, confidence)` — what `code/common/features.py` assumes |
| detects | one class, `person` |
| exported by | ultralytics 8.3.222, 2026-01-11 |
| source URL | **still unrecorded** — the file arrived in the project with no note of where it was downloaded from |

The download URL is the one gap left. Everything else above was recovered from the
file; a URL cannot be, so if this file is ever lost, re-obtaining *these exact* weights
means matching that SHA256 against whatever is downloaded.

The keypoint layout is not a detail: `common/features.py` centres on joints 11 and 12
and scales by the distance to 5 and 6, which is only the torso if the layout really is
COCO-17. `common/pose.py` raises if a pose model returns anything else, rather than
normalising by a nonsense length.

## Gitignored

`*.pt` is gitignored, so a fresh clone has this folder empty and cannot run anything:
the pose model has to be re-downloaded and a classifier promoted from
`../modelVersion/`. This `noted.md` is what keeps the folder in git.
