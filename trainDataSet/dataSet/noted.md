# trainDataSet/dataSet/ — raw clips, one folder per class

Original note: *this folder holds video that will be turned into classes of the model,
detected by YOLO-pose. By tracking all the vertices, the deep learning model will be
trained to recognise movement and be able to scale up.*

Confirmed — that is the plan. One subfolder per class, the folder name **is** the
label, read by `classType.py`.

```
dataSet/
  Jab/  Cross/  Hook/  Uppercut/  block/  dodge/
```

## Hard requirement — a new class is a new folder

**Adding a class must mean creating one folder here and dropping clips in it.
Nothing else is edited by hand.** No class list literal anywhere in the project, no
`num_classes = 6`, no `if label == "Jab"`. The six above are a starting set, not a
fixed set — `idle`, `guard`, `overhand`, `body_shot` all arrive the same way.

This is a constraint on every other module, not just on this folder:

| module | what it must do instead of hardcoding |
|---|---|
| `trainDataSet/classType.py` | scan this directory at runtime, return the sorted names |
| `trainDataSet/dataSpliter.py` | loop over whatever it finds, mirror the folders into `trainData/`, `testData/` |
| `code/trainingCode/train.py` | size the output layer from `len(classes)`, store the list in the checkpoint |
| `code/runTest/*` | read labels **from the checkpoint** in `modelholder/runningVersion/`, never rescan the folders |
| `code/unitTest/` | a test that creates a temp folder and asserts it shows up as a class |

### What makes a folder count as a class

Rules to settle now, before the scan is written:

- name matches the convention (see below) — the folder name is the label, so a typo
  is a new class
- holds at least one video file; an empty folder is scaffolding, not a class
- ignore anything starting with `.` or `_`, so `_raw/`, `_rejected/`, `.DS_Store`
  can sit here without becoming classes

The empty-folder rule is why all six currently resolve to **zero** classes. That is
correct behaviour, not a bug: no footage, no class.

### The trap: adding a class renumbers the old ones

Classes are sorted, so the index of every class after the insertion point moves:

```
before:  0 Cross  1 Hook  2 Jab  3 Uppercut  4 block  5 dodge
after adding Idle/:
         0 Cross  1 Hook  2 Idle  3 Jab  4 Uppercut  5 block  6 dodge
                          ^^^^^^ everything from here shifts by one
```

An old checkpoint still outputs "3" and now that reads as `Jab` instead of
`Uppercut`. Nothing crashes; accuracy just quietly rots. Two defences, both required:

1. every checkpoint carries its own class list, and inference maps indices through
   **that list**, never through a fresh scan of this folder
2. loading a checkpoint whose class list differs from the current folders is a loud
   error, not a warning

### After adding a folder

1. drop in the clips (aim for the same count as the other classes)
2. add a `noted.md` to the new folder, like the six existing ones — clips are
   gitignored, so without it the folder does not exist for anyone else who clones
3. re-run `dataSpliter.py` — the split is regenerated, not patched
4. retrain; the old checkpoint cannot be reused, the output layer changed shape
5. it saves as a new `classifier_v<N>.pt` in `modelholder/modelVersion/`, so the
   previous model still exists if the new class hurts
6. promote it into `modelholder/runningVersion/` only after scoring it against the
   old one — until you do, the demo and the score keep using the previous model, and
   that previous model is what the comparison is against

All six are currently empty. This is the blocking item: nothing else in the project
can be built or tested until there is footage here.

## What one clip should be

- **one move per clip**, trimmed to just that move plus a little before and after
- consistent enough framing that the whole body stays in shot — the pose model
  cannot find a hip it cannot see
- vary what should not matter: angle, distance, lighting, clothing, left and right
  stance. If every jab is filmed from the same spot, the model learns the spot.

Videos are gitignored (`*.mp4` and friends), so the clips live on disk only. Each
class folder keeps its own `noted.md` so the structure survives in git.

## Naming and mixed case

`block` and `dodge` are lowercase while `Jab`, `Cross`, `Hook` and `Uppercut` are
capitalised. Harmless on Windows, a real bug on Linux. Pick one convention.

This matters more now that the folder name is the only place a class is declared: if
folder naming is freeform, `Body Shot`, `body-shot` and `bodyShot` are three classes.
Suggest **lowercase, no spaces, underscores between words** (`jab`, `body_shot`), and
have `classType.py` reject anything that does not match rather than silently accept
a near-duplicate. Pretty display names, if we want them, belong in a lookup — not in
the folder name.

## Gaps to settle

- **No `Idle` class.** All six are active moves, so on a live feed the model must
  claim a punch is happening even when the person is standing still. A seventh
  `Idle` / `Guard` class would fix this — and under the rule above that is exactly
  one new folder plus a retrain, no code change.
- **`block` and `dodge` are not punches.** They are defensive, and a dodge is mostly
  head and torso movement while a punch is mostly arm. Grouping all six into one
  flat classifier is reasonable to start, but expect these two to behave differently.
- **How many clips per class?** Roughly 50–100 per class is a sane starting target
  for keypoint-based classification. Keep the counts balanced.
