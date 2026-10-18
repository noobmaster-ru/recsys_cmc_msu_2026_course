"""Fit-only indexing and sparse interaction matrices shared by all recommenders."""

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from scipy import sparse


COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def validate_interactions(frame):
    missing = set(COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Empty interactions or missing columns: {sorted(missing)}")
    if frame[COLUMNS].isna().any().any():
        raise ValueError("Missing IDs, ratings or timestamps")
    for column in ("user_id", "item_id"):
        if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{column} must contain nonempty strings")
    if not frame.rating.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Ratings must be integers in 1..5")
    timestamp = frame.timestamp.to_numpy(dtype=np.float64)
    if not np.isfinite(timestamp).all() or not (timestamp > 0).all() or not (timestamp == np.floor(timestamp)).all():
        raise ValueError("Timestamps must be positive integer Unix seconds")
    if frame.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Expected one interaction per user-item pair")


@dataclass
class FitData:
    """Catalog includes EVERY observed item; positive matrix need not cover it."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    user_index: dict
    item_index: dict
    observed: sparse.csr_matrix
    positive: sparse.csr_matrix
    fingerprint: str
    max_timestamp: int
    relevance_threshold: int

    @classmethod
    def from_frame(cls, frame, relevance_threshold=4):
        validate_interactions(frame)
        if relevance_threshold not in (1, 2, 3, 4, 5):
            raise ValueError("Invalid positive-rating threshold")
        user_ids = np.array(sorted(frame.user_id.unique()), dtype=object)
        item_ids = np.array(sorted(frame.item_id.unique()), dtype=object)
        user_index = {value: index for index, value in enumerate(user_ids)}
        item_index = {value: index for index, value in enumerate(item_ids)}
        rows = frame.user_id.map(user_index).to_numpy(dtype=np.int32)
        cols = frame.item_id.map(item_index).to_numpy(dtype=np.int32)
        shape = (len(user_ids), len(item_ids))
        observed = sparse.csr_matrix((np.ones(len(frame)), (rows, cols)), shape=shape)
        mask = frame.rating.to_numpy() >= relevance_threshold
        positive = sparse.csr_matrix((np.ones(int(mask.sum())), (rows[mask], cols[mask])), shape=shape)
        observed.sort_indices()
        positive.sort_indices()
        canonical = frame[COLUMNS].sort_values(["user_id", "item_id"]).reset_index(drop=True)
        digest = hashlib.sha256(pd.util.hash_pandas_object(canonical, index=False).values.tobytes())
        digest.update(str(relevance_threshold).encode())
        return cls(user_ids, item_ids, user_index, item_index, observed, positive,
                   digest.hexdigest(), int(frame.timestamp.max()), relevance_threshold)

    @property
    def positive_item_counts(self):
        return np.asarray(self.positive.sum(axis=0)).ravel()

    @property
    def history_sizes(self):
        return np.diff(self.observed.indptr)

    def seen_items(self, user_index):
        start, end = self.observed.indptr[user_index:user_index + 2]
        return self.observed.indices[start:end]
