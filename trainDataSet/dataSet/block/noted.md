# block — class folder

Defensive guard: hands up, forearms absorbing the shot. Little travel, mostly a
holding position.

Put clips of **only this move** here. The folder name is the label.

**Watch out — two problems specific to this class:**

- **It is nearly static.** A model trained on movement has little movement to read
  here. This is also exactly what a person doing nothing looks like, which is the
  argument for adding an `Idle` class — otherwise `block` becomes the default
  prediction for every quiet moment.
- **Where does it start and end?** A punch has a clear beginning and end; a block can
  be held indefinitely. Decide a convention — suggest trimming to the moment the
  guard comes up — and apply it to every clip.

Name is lowercase while `Jab`/`Cross`/`Hook`/`Uppercut` are capitalised. See
`../noted.md`.

Clips are gitignored; this note keeps the folder in git.
