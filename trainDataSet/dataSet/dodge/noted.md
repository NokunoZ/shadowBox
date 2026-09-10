# dodge — class folder

Evasive movement: slip, duck, or lean off the centre line to avoid a punch.

Put clips of **only this move** here. The folder name is the label.

**Watch out:** this is the one class that is mostly **head and torso**, not arms.
Everything else in the dataset is arm-driven. Two consequences:

- Keep the head and hips in frame — cropping to the upper body loses the lean.
- If the classifier ever ends up weighting arm joints heavily, this class suffers
  first. It is the one to check in the confusion matrix.

Consider whether slip / duck / lean should later become separate classes, or stay
merged as one `dodge`. Merged is the right call to start.

Name is lowercase while `Jab`/`Cross`/`Hook`/`Uppercut` are capitalised. See
`../noted.md`.

Clips are gitignored; this note keeps the folder in git.
