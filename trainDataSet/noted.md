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

Also worth deciding: symlink or copy? Copying duplicates every video on disk.

### It must follow the class list, not repeat it

The splitter loops over whatever `classType.py` discovered and creates the matching
subfolder under `trainData/` and `testData/`. Two consequences of classes arriving
by folder:

- **A new class needs no change here.** Add `dataSet/idle/`, re-run, and
  `trainData/idle/` and `testData/idle/` appear.
- **Regenerate, do not patch.** A re-run should clear the output folders first, or a
  class that was renamed or removed leaves a stale folder behind that the next
  training run happily picks up as a real class.

Per-class minimum: if a class has too few clips to give the test split at least a
couple of examples, say so loudly at split time. A class with one test clip has an
accuracy that is 0% or 100% and means nothing.

## Open question — no validation split

Only train and test exist. Tuning against the test set inflates the reported score.
Suggest a third `valData/` split, or carving a val slice out of train.
