from typing import Literal

import numpy as np
from sklearn.preprocessing import minmax_scale


def build_data(filename: str, split: Literal["train", "val", "test"], seq_len: int, pred_len: int):
    samples = []
    with open(filename) as f:
        rows = iter(f)
        next(rows)  # consume head
        for row in rows:
            cols = row.split(",")
            sample = list(map(float, cols[1:]))
            samples.append(sample)
    samples = samples[:14400]
    n = len(samples)
    if split == "train":
        samples = samples[:int(n * 0.6)]
    elif split == "val":
        samples = samples[int(n * 0.6): int(n * 0.8)]
    elif split == "test":
        samples = samples[int(n * 0.8):]
    else:
        raise ValueError("split should be 'train', 'val', or 'test', but got {!r}.".format(split))
    samples_arr = minmax_scale(samples)
    inputs = np.array(list(zip(*[samples_arr[i: -pred_len] for i in range(seq_len)])), dtype=np.float32)
    targets = np.array(list(zip(*[samples_arr[i + seq_len:] for i in range(pred_len)])), dtype=np.float32)
    return inputs, targets
