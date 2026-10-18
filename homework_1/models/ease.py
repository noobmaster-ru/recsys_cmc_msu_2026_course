"""EASE from its constrained ridge objective; no recommender/model library.

min_B ||X-XB||_F^2 + regularization*||B||_F^2, diag(B)=0.
X is the shared binary positive fit matrix. B[source, target] is dense.
"""

import numpy as np
from scipy.linalg.lapack import dpotrf, dpotri

from data_utils import FitData


class EASE:
    def __init__(self, regularization=100.):
        if isinstance(regularization, bool) or not np.isfinite(regularization) or regularization <= 0:
            raise ValueError('regularization must be finite and strictly positive')
        self.regularization = float(regularization)

    def fit(self, fit_data):
        if not isinstance(fit_data, FitData):
            raise TypeError('Pass shared fit-only FitData')
        positive = fit_data.positive
        if not np.all(positive.data == 1):
            raise ValueError('EASE requires binary positive preferences in this experiment')
        self.data_ = fit_data
        self.item_ids = fit_data.item_ids.copy()
        self.fit_fingerprint = fit_data.fingerprint
        # Fortran storage lets LAPACK overwrite one dense buffer throughout.
        gram = (positive.T @ positive).toarray(order='F')
        n_items = gram.shape[0]
        gram[np.diag_indices(n_items)] += self.regularization
        factor, info = dpotrf(gram, lower=0, clean=0, overwrite_a=1)
        if info != 0:
            raise np.linalg.LinAlgError(f'EASE Cholesky failed, LAPACK info={info}')
        precision, info = dpotri(factor, lower=0, overwrite_c=1)
        if info != 0:
            raise np.linalg.LinAlgError(f'EASE SPD inversion failed, LAPACK info={info}')
        # potri fills only the upper triangle. Restore symmetry before scaling.
        for column in range(n_items):
            precision[column+1:, column] = precision[column, column+1:]
        diagonal = precision.diagonal().copy()
        if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
            raise FloatingPointError('Invalid precision diagonal')
        self.precision_diagonal_ = diagonal
        # Divide COLUMN j by -P[j,j]; row scaling or transposition is wrong.
        # P is now exactly symmetric. Its transpose is an equal-valued C-order
        # view, avoiding a full-matrix C-order copy on every sparse scoring batch.
        weights = precision.T
        np.divide(weights, -diagonal[None, :], out=weights)
        np.fill_diagonal(weights, 0.)
        self.weights_ = weights
        negative, nonzero = 0, 0
        for start in range(0, n_items, 128):
            block = self.weights_[:, start:start+128]
            if not np.isfinite(block).all():
                raise FloatingPointError('Nonfinite EASE weights')
            negative += int(np.count_nonzero(block < 0))
            nonzero += int(np.count_nonzero(block))
        self.diagnostics_ = {
            'catalog_items': n_items, 'weight_bytes': self.weights_.nbytes,
            'nonzero_weights': nonzero, 'negative_weights': negative,
            'negative_fraction_of_nonzero': negative / nonzero if nonzero else 0.,
            # Exact optimum objective, derived columnwise using the KKT equations.
            'training_objective': float(np.sum(1./diagonal) - self.regularization*n_items),
        }
        return self

    def score_batch(self, user_ids, *, query_timestamp=None):
        if not hasattr(self, 'weights_'):
            raise RuntimeError('Fit EASE before scoring')
        if query_timestamp is not None and query_timestamp <= self.data_.max_timestamp:
            raise ValueError('Query must follow model training interactions')
        try:
            rows = np.asarray([self.data_.user_index[user] for user in user_ids], dtype=np.int32)
        except KeyError as error:
            raise ValueError('Unknown user is outside the primary warm-start cohort') from error
        scores = np.asarray(self.data_.positive[rows] @ self.weights_)
        for position, row in enumerate(rows):
            unseen_scores = scores[position].copy()
            unseen_scores[self.data_.seen_items(row)] = 0
            if not np.any(unseen_scores != 0):
                scores[position, :] = np.nan
        # Finite negative scores are legitimate; only zero profiles signal fallback.
        return scores
