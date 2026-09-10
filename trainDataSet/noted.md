# trainDataSet/ — data and its splits

```
dataSet/     raw clips, one folder per class   <- the only thing you hand-curate
   |
   |  dataSpliter.py
   v
trainData/   generated, gitignored
testData/    generated, gitignored
```

Both output folders are gitignored: they are reproducible from `dataSet/` by running
the splitter, so versioning them would just duplicate the videos.

## classType.py

*Return all classes of the dataset.*

The single source of truth for labels — training, inference and the splitter all read
from here, so an index can never mean two different things in two places.

**It discovers the classes, it does not declare them.** The project requirement is
that a new class is a new folder in `dataSet/` and nothing else (see
`dataSet/noted.md`), so this file must *scan* `dataSet/` at runtime. A literal list
of six names here would break that rule on day one.

What it must do:

- **Scan `dataSet/` for subdirectories**, no hardcoded names, no hardcoded count.
- **Skip non-classes**: anything starting with `.` or `_`, and any folder holding no
  video files. An empty folder is scaffolding — counting it would add a class the
  model can never learn, and one dead output unit that still eats probability mass.
- **Sort the class names.** If the order ever changes, class 2 stops meaning what the
  trained model thinks it means. Nothing errors; accuracy just drops and the labels
  are wrong. Sorting makes the order deterministic.
- **Validate the folder names** against the naming convention and reject what does
  not match, so `body-shot` and `body_shot` cannot become two classes.
- **Return both directions** — the ordered list and the name-to-index map — so no
  caller has to rebuild the mapping with its own `.index()` call.

### Discovery is for training only

Adding a folder renumbers every class after it alphabetically, so a scan run today
does not necessarily agree with a checkpoint trained last week. The scan therefore
feeds **training and splitting only**. Inference reads its labels out of the
checkpoint. See `modelholder/noted.md` for the checkpoint contract.

## dataSpliter.py

*Create testData and trainData folders from classType.py, then split dataSet into them.*

Three things that decide whether the accuracy number means anything:

1. **Split whole clips, never frames.** Frames from one clip in both train and test
   means the model has effectively seen the test set, and the score is fiction.
2. **Split by person and session too, if we ever have more than one person filmed.**
   Otherwise the model can score well by recognising the boxer rather than the punch.
3. **Seed the shuffle**, so a re-run reproduces the same split.

Copy or symlink? **Hardlink**, by default (`--mode`). A hardlink costs no extra disk
and needs no admin rights on Windows, which a symlink does; it falls back to a copy
automatically when the split folders land on a different volume. `--mode copy` and
`--mode symlink` are there if the default ever gets in the way.

The person/session rule needs a filename convention, since a file cannot say who is in
it: **everything before a double underscore is the person and session** —
`kasi_s01__jab_004.mp4` groups with `kasi_s01__jab_005.mp4` and the two never land on
opposite sides. A clip with no `__` is its own group, which is the right default while
one person is filming.

### It must follow the class list, not repeat it

The splitter loops over whatever `classType.py` discovered and creates the matching
subfolder under `trainData/` and `testData/`. Two consequences of classes arriving
by folder:

- **A new class needs no change here.** Add `dataSet/idle/`, re-run, and
  `trainData/idle/` and `testData/idle/` appear.
- **Regenerate, do not patch.** A re-run should clear the output folders first, or a
  class that was renamed or removed leaves a stale folder behind that the next
  training run happily picks up as a real class.

### When a class has only one person/session

Keeping groups whole is impossible to combine with the ratio if a class has exactly
one group — every clip would land on the same side, leaving train or test empty. The
splitter falls back to splitting whole *clips* in that case and **says so on stderr**.
Rule 1 still holds (no clip is ever on both sides), rule 2 cannot, and the score then
partly measures recognising the boxer. That warning is the signal to film someone else,
or another session.

Per-class minimum: if a class has too few clips to give the test split at least a
couple of examples, say so loudly at split time. A class with one test clip has an
accuracy that is 0% or 100% and means nothing.

## Validation split — settled, and it is not a folder

Only `trainData/` and `testData/` exist on disk, and tuning against the test set
inflates the reported score. The val slice is therefore **carved out of trainData at
training time** (`train.py --val-ratio`, default 0.2, stratified per class and seeded),
rather than becoming a third folder.

Why not a `valData/` folder: the split here is about which *clips* exist, and a third
folder would have to be regenerated and gitignored alongside the others for no gain.
Carving in `train.py` keeps the test split untouched by anything that tunes, which is
the property that actually matters.
