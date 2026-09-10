"""Shared code for training and inference.

trainingCode/ and runTest/ must turn video into keypoints IDENTICALLY -- same weights,
same normalisation, same window length. If the two drift apart the model sees
different inputs at train and at run time and accuracy quietly collapses, with nothing
raising. Every piece of that path lives here exactly once.
"""
