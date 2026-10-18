"""Exact implicit ALS for binary preferences; NumPy linear algebra, no model API.

Objective: sum_ui (1 + alpha*p_ui)*(p_ui - x_u@y_i)**2
           + regularization*(||X||_F**2 + ||Y||_F**2).
Zeros, including unobserved pairs, have confidence ONE, not zero.
"""

from time import perf_counter

import numpy as np

from data_utils import FitData


def row_batches(matrix, batch_size):
    """Group rows by positive degree to assemble exact normal equations in batches.

    This only changes computation order, not confidence, targets or row weights.
    Each stored neighbor index is used once; no user/item sampling or padding.
    """
    matrix = matrix.tocsr()
    matrix.sort_indices()
    degrees = np.diff(matrix.indptr)
    order = np.argsort(degrees, kind="stable")
    unique, offsets = np.unique(degrees[order], return_index=True)
    plans = []
    for position, degree in enumerate(unique):
        if degree == 0:
            continue
        end = offsets[position + 1] if position + 1 < len(offsets) else len(order)
        group = order[offsets[position]:end]
        for start in range(0, len(group), batch_size):
            rows = group[start:start + batch_size]
            locations = matrix.indptr[rows, None] + np.arange(degree)[None, :]
            plans.append((rows, matrix.indices[locations]))
    return matrix.shape[0], plans


def solve_factors(other, plan, *, alpha, regularization):
    """Exact minimizer of one ALS half-step, including the all-pairs Gram term."""
    n_rows, batches = plan
    dimensions = other.shape[1]
    result = np.zeros((n_rows, dimensions), dtype=np.float64)
    # dot/einsum avoid NumPy 2.2's known spurious Accelerate matmul warnings
    # on some Apple CPUs. Numerical results are checked, not warning-suppressed.
    base = np.dot(other.T, other) + regularization * np.eye(dimensions)
    for rows, neighbors in batches:
        vectors = other[neighbors]
        normal = base[None, :, :] + alpha * np.einsum('bni,bnj->bij', vectors, vectors, optimize=False)
        rhs = (1.0 + alpha) * vectors.sum(axis=1)
        if not np.isfinite(normal).all() or not np.isfinite(rhs).all():
            raise FloatingPointError('Nonfinite ALS normal equations')
        # Explicit final axis is required by NumPy 2's batched solve convention.
        result[rows] = np.linalg.solve(normal, rhs[:, :, None])[:, :, 0]
    # Rows without positive events have rhs=0 and hence the exact optimum x=0.
    return result


def implicit_objective(positive, user_factors, item_factors, *, alpha, regularization):
    """Exact full-catalog objective without allocating a users x items matrix."""
    gram_users = np.dot(user_factors.T, user_factors)
    gram_items = np.dot(item_factors.T, item_factors)
    loss = float(np.sum(gram_users * gram_items))  # sum of scores**2 on ALL pairs
    coo = positive.tocoo()
    for start in range(0, positive.nnz, 16384):
        stop = min(start + 16384, positive.nnz)
        scores = np.einsum("ij,ij->i", user_factors[coo.row[start:stop]],
                           item_factors[coo.col[start:stop]])
        # Replace the baseline score**2 term by (1+alpha)*(1-score)**2.
        loss += float(np.sum((1.0 + alpha) * (1.0-scores)**2 - scores**2))
    loss += regularization * (float(np.sum(user_factors**2)) + float(np.sum(item_factors**2)))
    return loss


class IALS:
    def __init__(self, n_factors=32, alpha=40.0, regularization=1.0,
                 n_iterations=15, seed=42, solver_batch_size=128):
        for name, value in (("n_factors", n_factors), ("n_iterations", n_iterations),
                            ("solver_batch_size", solver_batch_size)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and strictly positive")
        self.n_factors = int(n_factors)
        self.alpha = float(alpha)
        self.regularization = float(regularization)
        self.n_iterations = int(n_iterations)
        self.seed = seed
        self.solver_batch_size = int(solver_batch_size)

    def fit(self, fit_data, *, progress_callback=None):
        if not isinstance(fit_data, FitData):
            raise TypeError("Pass fit-only FitData shared with the evaluator")
        positive = fit_data.positive.tocsr()
        if not np.all(positive.data == 1):
            raise ValueError("This baseline requires binary positive preferences")
        self.data_ = fit_data
        self.item_ids = fit_data.item_ids.copy()
        self.fit_fingerprint = fit_data.fingerprint
        user_plan = row_batches(positive, self.solver_batch_size)
        item_plan = row_batches(positive.T.tocsr(), self.solver_batch_size)
        rng = np.random.default_rng(self.seed)
        self.user_factors_ = np.zeros((len(fit_data.user_ids), self.n_factors))
        self.item_factors_ = rng.normal(0, .01, size=(len(fit_data.item_ids), self.n_factors))
        self.item_factors_[fit_data.positive_item_counts == 0] = 0
        self.positive_history_sizes_ = np.diff(positive.indptr)
        self.loss_history_ = [implicit_objective(positive, self.user_factors_, self.item_factors_,
            alpha=self.alpha, regularization=self.regularization)]
        self.epoch_seconds_ = []
        for iteration in range(1, self.n_iterations + 1):
            started = perf_counter()
            self.user_factors_ = solve_factors(self.item_factors_, user_plan,
                alpha=self.alpha, regularization=self.regularization)
            self.item_factors_ = solve_factors(self.user_factors_, item_plan,
                alpha=self.alpha, regularization=self.regularization)
            loss = implicit_objective(positive, self.user_factors_, self.item_factors_,
                                     alpha=self.alpha, regularization=self.regularization)
            if not np.isfinite(loss) or loss > self.loss_history_[-1] + 1e-8 * max(1., self.loss_history_[-1]):
                raise FloatingPointError("Exact ALS objective increased or became nonfinite")
            self.loss_history_.append(loss)
            self.epoch_seconds_.append(perf_counter() - started)
            if progress_callback is not None:
                progress_callback(iteration, loss, self.epoch_seconds_[-1])
        return self

    def score_batch(self, user_ids, *, query_timestamp=None):
        if not hasattr(self, "user_factors_"):
            raise RuntimeError("Fit the model before scoring")
        if query_timestamp is not None and query_timestamp <= self.data_.max_timestamp:
            raise ValueError("Query must follow model training interactions")
        try:
            rows = np.asarray([self.data_.user_index[user] for user in user_ids], dtype=np.int32)
        except KeyError as error:
            raise ValueError("Unknown user is outside the primary warm-start cohort") from error
        scores = np.dot(self.user_factors_[rows], self.item_factors_.T)
        for position, row in enumerate(rows):
            unseen_scores = scores[position].copy()
            unseen_scores[self.data_.seen_items(row)] = 0
            if self.positive_history_sizes_[row] == 0 or not np.any(unseen_scores != 0):
                scores[position, :] = np.nan
        return scores
