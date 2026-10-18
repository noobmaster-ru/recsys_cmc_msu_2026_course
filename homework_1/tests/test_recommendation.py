"""Hand-computed examples and protocol invariants; no external metric oracle."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from data_utils import FitData
from evaluation import (EvaluationContext, evaluate_model, evaluate_scores, evaluate_popularity,
                        paired_bootstrap, rank_top_k, ranking_metrics)
from models.item_knn import ItemKNN


def example():
    # Positive degrees: A=2, B=2, C=3, D=0, E=0, F=1.
    ratings = [("u1", "A", 5), ("u1", "B", 4), ("u1", "D", 1),
               ("u2", "A", 5), ("u2", "C", 4), ("u3", "B", 5),
               ("u3", "C", 5), ("u4", "C", 5), ("u5", "E", 1), ("u6", "F", 5)]
    fit = pd.DataFrame([(u, i, r, t+1) for t, (u, i, r) in enumerate(ratings)],
                       columns=["user_id", "item_id", "rating", "timestamp"])
    hold = pd.DataFrame([
        ("u1", "E", 4, 100), ("u1", "F", 5, 101), ("u2", "B", 5, 101),
        ("u4", "A", 4, 102), ("u5", "C", 5, 103), ("new", "A", 5, 104),
        ("u1", "G", 5, 104), ("u3", "F", 1, 105), ("u1", "D", 5, 106)],
        columns=fit.columns)
    return fit, hold


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.fit, self.hold = example()
        self.data = FitData.from_frame(self.fit)
        self.context = EvaluationContext.from_frames(self.fit, self.hold, fit_data=self.data, k=2)

    def test_hand_computed_cosine_and_shrinkage(self):
        model = ItemKNN(n_neighbors=2, shrinkage=2).fit(self.data)
        expected = np.zeros((6, 6))
        expected[0, 1] = expected[1, 0] = 1/6
        expected[0, 2] = expected[2, 0] = 1/(3*np.sqrt(6))
        expected[1, 2] = expected[2, 1] = 1/(3*np.sqrt(6))
        np.testing.assert_allclose(model.neighbors_.toarray(), expected)
        np.testing.assert_array_equal(model.neighbors_.diagonal(), 0)

    def test_target_neighborhood_orientation_and_ties(self):
        model = ItemKNN(n_neighbors=1, shrinkage=0).fit(self.data)
        # C ties between A and B: A wins lexicographically. The graph is directed.
        self.assertEqual(model.neighbors_[2, 0], 1/np.sqrt(6))
        self.assertEqual(model.neighbors_[0, 2], 0)
        # u1 likes A,B. Target C must use A -> C's neighborhood, not the transpose.
        scores = model.score_batch(["u1"])[0]
        self.assertAlmostEqual(scores[2], 1/np.sqrt(6))

    def test_no_positive_or_unseen_neighbor_evidence_signals_fallback(self):
        model = ItemKNN(1, 0).fit(self.data)
        self.assertTrue(np.isnan(model.score_batch(["u5", "u6"])).all())
        with self.assertRaises(ValueError):
            model.score_batch(["new"])

    def test_catalog_keeps_negative_only_items_and_masks_them(self):
        self.assertEqual(self.data.item_ids.tolist(), list("ABCDEF"))
        self.assertEqual(self.data.positive_item_counts[3], 0)
        self.assertIn(3, self.data.seen_items(self.data.user_index["u1"]))
        # A future positive re-rating of already known D is excluded from ground truth.
        self.assertEqual(self.context.relevance[0], frozenset([4, 5]))
        self.assertEqual(self.context.diagnostics["seen_relevant_items_removed"], 1)

    def test_global_time_guard_and_wrong_fit(self):
        early = self.hold.copy()
        early.loc[0, "timestamp"] = int(self.fit.timestamp.max())
        with self.assertRaises(ValueError):
            EvaluationContext.from_frames(self.fit, early, k=2)
        with self.assertRaises(ValueError):
            ItemKNN().fit(self.data).score_batch(["u1"], query_timestamp=1)
        changed = self.fit.copy()
        changed.loc[0, "rating"] = 1
        model = ItemKNN().fit(FitData.from_frame(changed))
        with self.assertRaises(ValueError):
            evaluate_model(model, self.context)

    def test_top_k_boundary_ties_are_lexicographic(self):
        scores = np.array([0., 1., 1., 1., 2.])
        # Arbitrary candidate input order still resolves boundary ties by global ID.
        np.testing.assert_array_equal(rank_top_k(scores, [3, 2, 4, 1, 0], 3), [4, 1, 2])
        np.testing.assert_array_equal(rank_top_k(np.zeros(20), np.arange(20), 10), np.arange(10))

    def test_top_k_matches_full_sort_with_many_ties(self):
        rng = np.random.default_rng(7)
        for _ in range(30):
            scores = rng.integers(-2, 3, size=100).astype(float)
            candidates = rng.choice(100, size=60, replace=False)
            reference = sorted(candidates, key=lambda i: (-scores[i], i))[:10]
            np.testing.assert_array_equal(rank_top_k(scores, candidates, 10), reference)

    def test_hand_computed_metrics_multiple_relevance(self):
        ndcg, recall = ranking_metrics([9, 2, 8, 4], {2, 4, 6}, 4)
        expected = (1/np.log2(3) + 1/np.log2(5))/(1 + 1/np.log2(3) + 1/np.log2(4))
        self.assertAlmostEqual(ndcg, expected)
        self.assertAlmostEqual(recall, 2/3)
        with self.assertRaises(ValueError):
            ranking_metrics([1, 1], {1}, 2)
        with self.assertRaises(ValueError):
            ranking_metrics([1, 2], set(), 2)

    def _scores(self, users):
        result = np.zeros((len(users), 6))
        for row, user in enumerate(users):
            if user == "u1":
                result[row, 0] = np.nan  # masked history: MUST NOT cause fallback
            if user == "u2":
                result[row, 3] = np.inf  # unseen candidate: entire ranking uses fallback
            if user == "u5":
                result[row, :] = np.nan
        return result

    def test_shared_fallback_masks_macro_and_no_user_dropping(self):
        result = evaluate_scores(self._scores, self.context, batch_size=2)
        self.assertEqual(result.per_user.index.tolist(), ["u1", "u2", "u4", "u5"])
        np.testing.assert_array_equal(result.recommendations, [[2, 4], [1, 5], [0, 1], [2, 0]])
        self.assertEqual(result.per_user.fallback.tolist(), [False, True, False, True])
        self.assertAlmostEqual(result.metrics["recall_at_2"], .875)  # macro, not micro=.8
        self.assertAlmostEqual(result.metrics["fallback_rate"], .5)
        for user_row, chosen in zip(self.context.user_indices, result.recommendations):
            self.assertTrue(set(chosen).isdisjoint(self.data.seen_items(user_row)))

    def test_batch_size_does_not_change_rankings_or_metrics(self):
        first = evaluate_scores(self._scores, self.context, batch_size=1)
        second = evaluate_scores(self._scores, self.context, batch_size=7)
        pd.testing.assert_frame_equal(first.per_user, second.per_user)
        np.testing.assert_array_equal(first.recommendations, second.recommendations)
        self.assertEqual(first.cohort_fingerprint, second.cohort_fingerprint)

    def test_scorer_shape_errors_are_not_silently_hidden(self):
        with self.assertRaises(ValueError):
            evaluate_scores(lambda users: np.zeros((len(users), 5)), self.context)

    def test_paired_bootstrap_is_reproducible_and_checks_cohorts(self):
        result = evaluate_popularity(self.context)
        one = paired_bootstrap(result, result, resamples=100, seed=42)
        two = paired_bootstrap(result, result, resamples=100, seed=42)
        pd.testing.assert_frame_equal(one, two)
        named = paired_bootstrap(result, result, resamples=5,
                                 result_name="AnotherRecommender", baseline_name="Control")
        self.assertEqual(set(named.estimate_for), {"AnotherRecommender", "Control", "difference"})
        differences = one[one.estimate_for == "difference"]
        self.assertTrue((differences[["estimate", "ci_low", "ci_high"]] == 0).all().all())
        altered = self.hold.copy()
        altered.loc[0, "rating"] = 1
        other = EvaluationContext.from_frames(self.fit, altered, k=2)
        with self.assertRaises(ValueError):
            paired_bootstrap(result, evaluate_popularity(other), resamples=10)


if __name__ == "__main__":
    unittest.main()
