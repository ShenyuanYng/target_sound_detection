"""Factory for task-independent DataLoader collation."""

from functools import partial

from utils.collate import collate_seq

"""
就是调collate_seq
"""

def collate_creator(hparams):
    collate_type = getattr(hparams, "collate_type", "sequence")
    if collate_type == "sequence":
        return partial(
            collate_seq,
            pad_back_or_front=hparams.pad_back_or_front,
        )
    raise ValueError(f"Unsupported collate_type: {collate_type}")
