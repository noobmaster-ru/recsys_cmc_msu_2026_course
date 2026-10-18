"""Full-catalog top-N evaluation, independent of a recommender's training target."""

from dataclasses import dataclass
import hashlib
import json
import math
from time import perf_counter

import numpy as np
import pandas as pd

from data_utils import FitData, validate_interactions


@dataclass
class EvaluationContext:
    fit: FitData
    user_ids: np.ndarray
    user_indices: np.ndarray
    relevance: list
    k: int
    query_timestamp: int
    fingerprint: str
    diagnostics: dict

    @classmethod
    def from_frames(cls, fit_frame, holdout, *, fit_data=None, k=10,
                    relevance_threshold=4, query_timestamp=None):
        validate_interactions(fit_frame)
        validate_interactions(holdout)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        data = fit_data if fit_data is not None else FitData.from_frame(fit_frame, relevance_threshold)
        # Verify a supplied index/matrix really belongs to this fit, not another split.
        if fit_data is not None:
            check = FitData.from_frame(fit_frame, relevance_threshold)
            if data.fingerprint != check.fingerprint:
                raise ValueError("FitData belongs to a different fit or positive threshold")
        timestamp = int(holdout.timestamp.min()) if query_timestamp is None else int(query_timestamp)
        if data.max_timestamp >= timestamp or int(holdout.timestamp.min()) < timestamp:
            raise ValueError("Fit, recommendation time and holdout must follow global chronology")
        known_user = holdout.user_id.isin(data.user_index)
        known_item = holdout.item_id.isin(data.item_index)
        relevant_rows = holdout[known_user & known_item & (holdout.rating >= relevance_threshold)]
        groups = relevant_rows.groupby("user_id", sort=True).item_id.agg(set)
        users, indices, relevance = [], [], []
        seen_relevant, too_few = 0, 0
        for user, item_ids in groups.items():
            row = data.user_index[user]
            seen = set(data.seen_items(row))
            truth = {data.item_index[item] for item in item_ids} - seen
            seen_relevant += len(item_ids) - len(truth)
            if not truth:
                continue
            if len(data.item_ids) - len(seen) < k:
                too_few += 1
                continue
            users.append(user)
            indices.append(row)
            relevance.append(frozenset(truth))
        if not users:
            raise ValueError("No eligible evaluation users")
        payload = [data.fingerprint, int(k), timestamp,
                   [[user, sorted(truth)] for user, truth in zip(users, relevance)]]
        fingerprint = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
        diagnostics = {
            "evaluation_users": len(users), "catalog_items": len(data.item_ids),
            "relevant_interactions": sum(map(len, relevance)),
            "seen_relevant_items_removed": seen_relevant,
            "users_excluded_for_fewer_than_k_candidates": too_few,
            "new_user_interactions": int((~known_user).sum()),
            "new_item_interactions": int((~known_item).sum()),
        }
        return cls(data, np.asarray(users, dtype=object), np.asarray(indices, dtype=np.int32),
                   relevance, int(k), timestamp, fingerprint, diagnostics)


def rank_top_k(scores, candidates, k):
    """Exact top-k with ID ties, without sorting the entire catalog.

    Candidate column indices must refer to lexicographically sorted item IDs.
    Boundary ties are explicitly included then resolved by ID, unlike a plain
    argpartition (which does not define which tied elements enter the result).
    """
    scores = np.asarray(scores)
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) < k or k < 1:
        raise ValueError("Not enough candidates for top-k")
    values = scores[candidates]
    if not np.isfinite(values).all():
        raise ValueError("Candidate scores must be finite after shared fallback")
    threshold = np.partition(values, len(values) - k)[len(values) - k]
    better = candidates[values > threshold]
    ties = np.sort(candidates[values == threshold])[:k - len(better)]
    selected = np.concatenate((better, ties))
    return selected[np.lexsort((selected, -scores[selected]))]


def ranking_metrics(recommendations, relevant, k):
    """Binary per-user NDCG/Recall; zero-relevance users are not evaluable."""
    if len(recommendations) != k or len(set(recommendations)) != k:
        raise ValueError("Recommendations must contain exactly k distinct items")
    if not relevant:
        raise ValueError("Recall/NDCG require at least one relevant item")
    hits = np.fromiter((item in relevant for item in recommendations), dtype=np.float64, count=k)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = discounts[:min(k, len(relevant))].sum()
    return float(hits @ discounts / ideal), float(hits.sum() / len(relevant))


@dataclass
class EvaluationResult:
    metrics: dict
    per_user: pd.DataFrame
    recommendations: np.ndarray
    cohort_fingerprint: str
    k: int


def evaluate_scores(score_function, context, *, batch_size=128):
    """score_function(users) returns [batch, entire fit catalog], in FitData order.

    Evaluate all frozen users. The scorer never sees relevance labels. Only
    non-finite CANDIDATE scores trigger the common full-ranking fallback; masked
    historical items do not trigger it. ItemKNN explicitly signals no evidence
    with NaNs. No fallback for a partial finite list or tied zero-score tail.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    started = perf_counter()
    data, k = context.fit, context.k
    n_items, n_users = len(data.item_ids), len(context.user_ids)
    popularity = data.positive_item_counts
    history_sizes = data.history_sizes
    recommendations = np.empty((n_users, k), dtype=np.int32)
    user_records = []
    all_indices = np.arange(n_items)
    for start in range(0, n_users, batch_size):
        stop = min(start + batch_size, n_users)
        scores = np.asarray(score_function(context.user_ids[start:stop]), dtype=np.float64)
        if scores.shape != (stop - start, n_items):
            raise ValueError(f"Scorer returned {scores.shape}; expected {(stop-start, n_items)}")
        for offset, position in enumerate(range(start, stop)):
            user_row = context.user_indices[position]
            mask = np.ones(n_items, dtype=bool)
            mask[data.seen_items(user_row)] = False
            candidates = all_indices[mask]
            values = scores[offset]
            fallback = not np.isfinite(values[candidates]).all()
            chosen = rank_top_k(popularity if fallback else values, candidates, k)
            recommendations[position] = chosen
            ndcg, recall = ranking_metrics(chosen, context.relevance[position], k)
            user_records.append((context.user_ids[position], ndcg, recall, fallback,
                                 len(context.relevance[position]), int(history_sizes[user_row])))
    per_user = pd.DataFrame(user_records, columns=["user_id", "ndcg", "recall", "fallback",
                                                  "n_relevant", "history_size"]).set_index("user_id")
    metrics = {
        f"ndcg_at_{k}": float(per_user.ndcg.mean()),
        f"recall_at_{k}": float(per_user.recall.mean()),
        f"catalog_coverage_at_{k}": len(np.unique(recommendations)) / n_items,
        "fallback_rate": float(per_user.fallback.mean()), "users": n_users,
        "evaluation_seconds": perf_counter() - started,
    }
    return EvaluationResult(metrics, per_user, recommendations, context.fingerprint, k)


def evaluate_model(model, context, *, batch_size=128):
    if model.fit_fingerprint != context.fit.fingerprint:
        raise ValueError("Model was trained on another fit or target-index definition")
    if not np.array_equal(model.item_ids, context.fit.item_ids):
        raise ValueError("Model catalog/order differs from the shared evaluation catalog")
    return evaluate_scores(lambda users: model.score_batch(users, query_timestamp=context.query_timestamp),
                           context, batch_size=batch_size)


def evaluate_popularity(context, *, batch_size=128):
    """Untrained control: positive-fit event count, same candidates and tie rule."""
    counts = context.fit.positive_item_counts
    return evaluate_scores(lambda users: np.broadcast_to(counts, (len(users), len(counts))),
                           context, batch_size=batch_size)


def history_slices(result):
    frame = result.per_user.copy()
    frame["history_slice"] = pd.cut(frame.history_size, [0, 4, 9, np.inf], labels=["1-4", "5-9", "10+"])
    return frame.groupby("history_slice", observed=False).agg(
        users=("ndcg", "size"), ndcg=("ndcg", "mean"), recall=("recall", "mean"),
        fallback_rate=("fallback", "mean"))


def popularity_slices(result, context):
    if result.cohort_fingerprint != context.fingerprint:
        raise ValueError("Evaluation result/context mismatch")
    counts = context.fit.positive_item_counts
    order = np.lexsort((np.arange(len(counts)), -counts))
    head = set(order[:math.ceil(.1 * len(order))])
    rows = []
    for label in ("head_top_10_percent", "tail_other_90_percent"):
        recalls, relevance_count = [], 0
        for recommendations, truth in zip(result.recommendations, context.relevance):
            subset = truth & head if label.startswith("head") else truth - head
            if subset:
                recalls.append(len(set(recommendations) & subset) / len(subset))
                relevance_count += len(subset)
        rows.append({"item_slice": label, "users": len(recalls),
                     "relevant_interactions": relevance_count,
                     f"recall_at_{result.k}": float(np.mean(recalls)) if recalls else None})
    return pd.DataFrame(rows).set_index("item_slice")


def paired_bootstrap(result, baseline, *, resamples=2000, seed=42, confidence=.95,
                     result_name="Model", baseline_name="Baseline"):
    """Resample identical user positions jointly; no model retraining or selection."""
    if (result.cohort_fingerprint != baseline.cohort_fingerprint or result.k != baseline.k
            or not result.per_user.index.equals(baseline.per_user.index)):
        raise ValueError("Paired bootstrap requires exactly the same users, labels and candidates")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap parameters")
    rng = np.random.default_rng(seed)
    arrays = [result.per_user[metric].to_numpy() for metric in ("ndcg", "recall")]
    refs = [baseline.per_user[metric].to_numpy() for metric in ("ndcg", "recall")]
    draws = np.empty((resamples, 2, 3))
    for start in range(0, resamples, 32):
        stop = min(start + 32, resamples)
        indices = rng.integers(0, len(result.per_user), size=(stop-start, len(result.per_user)))
        for metric_index, (values, reference) in enumerate(zip(arrays, refs)):
            a, b = values[indices].mean(axis=1), reference[indices].mean(axis=1)
            draws[start:stop, metric_index, :] = np.column_stack((a, b, a-b))
    alpha = (1-confidence)/2
    rows = []
    for metric_index, metric in enumerate((f"ndcg_at_{result.k}", f"recall_at_{result.k}")):
        a, b = arrays[metric_index].mean(), refs[metric_index].mean()
        for variant, (label, estimate) in enumerate(zip((result_name, baseline_name, "difference"), (a, b, a-b))):
            low, high = np.quantile(draws[:, metric_index, variant], [alpha, 1-alpha])
            rows.append({"metric": metric, "estimate_for": label, "estimate": float(estimate),
                         "ci_low": float(low), "ci_high": float(high)})
    return pd.DataFrame(rows)
