# idle — class folder

Standing, guard up or hands down, **not** throwing or evading anything. The quiet
moments between moves.

Put clips of **only this** here. The folder name is the label.

## Why this folder exists

Every other class is an action. Without `idle`, a live webcam forces the model to
claim a punch is happening while the person is just standing there — softmax has to
sum to one, so the probability lands somewhere regardless. `block` would absorb most
of it, being the most static class, and the demo would read "block" all day.

This was flagged as an open question in three notes (`../noted.md`, `../../noted.md`,
`block/noted.md`). Under the project's own rule — a class is a folder — settling it
costs exactly this folder plus clips, and no code change anywhere.

## Filming it

- **Vary it hard.** Idle is not one pose: guard up, guard down, shifting weight,
  bouncing on the toes, resetting after a punch, wiping the face. If every idle clip
  is the same still stance, the model learns that stance, not "no move".
- **Same clip length as the other classes**, cut the same way, so length itself does
  not become the giveaway.
- **Do not include the wind-up or recovery of a real punch.** That footage belongs to
  the punch's own class, and putting it here teaches the two classes to overlap at
  exactly the frames that matter most.

## Watch out

It will likely be the largest and easiest class, since idle footage is trivial to
film. Keep the count balanced with the others (see `../noted.md`) — an idle class
twice the size of the rest teaches the model that "nothing is happening" is the safe
guess, and the punches get quieter.

Clips are gitignored; this note keeps the folder in git. See `../noted.md` for the
shared filming guidance.
