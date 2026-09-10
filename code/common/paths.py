"""Every filesystem path the project cares about, in one file.

Especially the two model folders. Retyping these in three scripts is how training
ends up writing where inference reads.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "trainDataSet" / "dataSet"
TRAIN_DIR = ROOT / "trainDataSet" / "trainData"
TEST_DIR = ROOT / "trainDataSet" / "testData"

MODELHOLDER = ROOT / "modelholder"

# the archive: one file per training run, never overwritten, never read at run time
MODEL_VERSION_DIR = MODELHOLDER / "modelVersion"

# what actually runs: the frozen pose model plus the promoted classifier
RUNNING_VERSION_DIR = MODELHOLDER / "runningVersion"
POSE_WEIGHTS = RUNNING_VERSION_DIR / "yolo26s-pose.pt"
RUNNING_CLASSIFIER = RUNNING_VERSION_DIR / "classifier.pt"

# cached keypoints -- pose estimation runs once per clip, not once per epoch
CACHE_DIR = ROOT / "trainDataSet" / ".keypointCache"

CONFIDENT_CSV = ROOT / "code" / "runTest" / "confident.csv"
