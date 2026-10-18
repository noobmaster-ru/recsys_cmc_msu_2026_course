"""Item-based CF: binary cosine, overlap shrinkage, target-item top-k neighbors.

Only NumPy/SciPy matrix operations are used; no pretrained recommender or kNN API.
"""

import numpy as np
from scipy import sparse

from data_utils import FitData


def top_neighbors(matrix, n_neighbors):
    """Prune every TARGET row, resolving equal weights by lexicographic item ID.

    FitData columns are already in lexicographic ID order. The resulting graph
    need not be symmetric; row i lists the source items used to score target i.
    """
    matrix = matrix.tocsr()
    indices, values = [], []
    indptr = [0]
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row:row + 2]
        cols, weights = matrix.indices[start:end], matrix.data[start:end]
        keep = (cols != row) & (weights > 0)
        cols, weights = cols[keep], weights[keep]
        selected = np.lexsort((cols, -weights))[:n_neighbors]
        indices.extend(cols[selected])
        values.extend(weights[selected])
        indptr.append(len(indices))
    graph = sparse.csr_matrix((np.asarray(values, dtype=np.float64),
                               np.asarray(indices, dtype=np.int32),
                               np.asarray(indptr, dtype=np.int64)), shape=matrix.shape)
    graph.sort_indices()
    return graph


class ItemKNN:
    def __init__(self, n_neighbors=100, shrinkage=10.0):
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be finite and nonnegative")
        self.n_neighbors = int(n_neighbors)
        self.shrinkage = float(shrinkage)

    def fit(self, fit_data):
        if not isinstance(fit_data, FitData):
            raise TypeError("Pass shared FitData built only from the current fit period")
        self.data_ = fit_data
        self.item_ids = fit_data.item_ids.copy()
        self.fit_fingerprint = fit_data.fingerprint
        positive = fit_data.positive
        counts = fit_data.positive_item_counts
        cooccurrence = (positive.T @ positive).tocsr()
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()
        # Row i is the TARGET item; column j is a potential history neighbor.
        row_ids = np.repeat(np.arange(len(counts)), np.diff(cooccurrence.indptr))
        denominator = np.sqrt(counts[row_ids] * counts[cooccurrence.indices])
        overlaps = cooccurrence.data.copy()
        cooccurrence.data = np.divide(overlaps, denominator, out=np.zeros_like(overlaps),
                                      where=denominator > 0)
        if self.shrinkage:
            cooccurrence.data *= overlaps / (overlaps + self.shrinkage)
        self.neighbors_ = top_neighbors(cooccurrence, self.n_neighbors)
        return self

    def score_batch(self, user_ids, *, query_timestamp=None):
        if not hasattr(self, "neighbors_"):
            raise RuntimeError("Fit the model before scoring")
        if query_timestamp is not None and query_timestamp <= self.data_.max_timestamp:
            raise ValueError("Query must follow all model training interactions")
        try:
            rows = np.array([self.data_.user_index[user] for user in user_ids], dtype=np.int32)
        except KeyError as error:
            raise ValueError("Unknown user: cold start is outside the primary evaluation") from error
        # score(u,i) = sum_{j in N_i} B[u,j] * S[i,j].
        scores = (self.data_.positive[rows] @ self.neighbors_.T).toarray()
        for position, user_row in enumerate(rows):
            # No evidence for ANY unseen candidate means personalization is unavailable.
            # NaN is an explicit signal to the shared evaluator, not a user exclusion.
            seen = self.data_.seen_items(user_row)
            candidate_scores = scores[position].copy()
            candidate_scores[seen] = 0
            if not np.any(candidate_scores > 0):
                scores[position, :] = np.nan
        return scores
