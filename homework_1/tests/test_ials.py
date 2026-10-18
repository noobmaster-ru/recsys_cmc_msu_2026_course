"""Dense weighted least-squares oracles and iALS numerical invariants."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import sparse

from data_utils import FitData
from models.ials import IALS, row_batches, solve_factors, implicit_objective
from evaluation import EvaluationContext, evaluate_model


class IALSTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            ("a", "A", 5, 1), ("a", "B", 4, 2), ("a", "D", 1, 3),
            ("b", "B", 5, 2), ("b", "C", 4, 3), ("c", "A", 5, 4),
            ("c", "C", 5, 5), ("negative", "D", 2, 6)],
            columns=["user_id", "item_id", "rating", "timestamp"])
        self.data = FitData.from_frame(self.frame)

    def test_hand_computed_update_includes_missing_pairs(self):
        preferences = sparse.csr_matrix([[1., 0., 1.], [0., 0., 0.]])
        vectors = np.array([[1., 0.], [0., 2.], [1., 1.]])
        actual = solve_factors(vectors, row_batches(preferences, 8), alpha=2., regularization=.5)
        # A=[[6.5,3],[3,7.5]], b=[6,3], determinant=39.75.
        np.testing.assert_allclose(actual[0], [36/39.75, 1.5/39.75], rtol=1e-13)
        np.testing.assert_array_equal(actual[1], [0, 0])
        wrong_positive_only = np.linalg.solve(3 * vectors[[0, 2]].T @ vectors[[0, 2]] + .5*np.eye(2), [6, 3])
        self.assertFalse(np.allclose(actual[0], wrong_positive_only))

    def test_batched_updates_match_dense_confidence_matrix(self):
        rng = np.random.default_rng(8)
        p = np.array([[1.,0,1,0], [0,0,0,0], [0,1,0,0], [1,1,1,1], [0,0,1,0]])
        y = rng.normal(size=(4,3))
        for alpha in (0., 10.):
            expected = []
            for row in p:
                confidence = np.diag(1+alpha*row)
                expected.append(np.linalg.solve(y.T @ confidence @ y + .7*np.eye(3), y.T @ confidence @ row))
            for batch in (1, 2, 128):
                actual = solve_factors(y, row_batches(sparse.csr_matrix(p), batch), alpha=alpha, regularization=.7)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_sparse_objective_matches_explicit_all_pair_sum(self):
        x = np.array([[.2,-.1], [.5,.3]])
        y = np.array([[1.,2.], [-1.,.4], [.3,.8]])
        p = np.array([[1.,0,1], [0,1,0]])
        expected = 0.
        for user in range(2):
            for item in range(3):
                expected += (1+4*p[user,item])*(p[user,item]-x[user]@y[item])**2
        expected += .3*(np.sum(x*x)+np.sum(y*y))
        actual = implicit_objective(sparse.csr_matrix(p), x, y, alpha=4, regularization=.3)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_exact_als_loss_is_monotonic_and_factors_finite(self):
        model = IALS(n_factors=3, alpha=10, regularization=.2, n_iterations=8).fit(self.data)
        self.assertEqual(len(model.loss_history_), 9)
        self.assertTrue(np.all(np.diff(model.loss_history_) <= 1e-9))
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])
        self.assertTrue(np.isfinite(model.user_factors_).all())
        self.assertTrue(np.isfinite(model.item_factors_).all())

    def test_zero_profiles_and_negative_only_items_have_zero_optimum(self):
        model = IALS(n_factors=2, n_iterations=4).fit(self.data)
        np.testing.assert_array_equal(model.user_factors_[self.data.user_index['negative']], [0,0])
        np.testing.assert_array_equal(model.item_factors_[self.data.item_index['D']], [0,0])
        self.assertTrue(np.isnan(model.score_batch(['negative'])).all())
        self.assertEqual(model.item_ids.tolist(), ['A','B','C','D'])

    def test_seed_and_solver_batch_size_reproduce_scores(self):
        first = IALS(n_factors=3, n_iterations=5, seed=42, solver_batch_size=1).fit(self.data)
        second = IALS(n_factors=3, n_iterations=5, seed=42, solver_batch_size=128).fit(self.data)
        np.testing.assert_allclose(first.score_batch(['a','b','c']),second.score_batch(['a','b','c']),
                                   rtol=1e-11, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(first.loss_history_,second.loss_history_,rtol=1e-12)

    def test_common_evaluator_mask_and_fallback_keep_same_users(self):
        hold = pd.DataFrame([('a','C',5,10),('b','A',4,10),('negative','A',5,10)],columns=self.frame.columns)
        context = EvaluationContext.from_frames(self.frame,hold,k=1,query_timestamp=10)
        model = IALS(n_factors=2,n_iterations=4).fit(self.data)
        result = evaluate_model(model,context)
        self.assertEqual(result.metrics['users'],3)
        self.assertTrue(result.per_user.loc['negative','fallback'])
        for row, recommendations in zip(context.user_indices,result.recommendations):
            self.assertTrue(set(recommendations).isdisjoint(self.data.seen_items(row)))
        with self.assertRaises(ValueError): model.score_batch(['a'],query_timestamp=5)
        with self.assertRaises(ValueError): model.score_batch(['new'])

    def test_negative_latent_scores_are_valid_not_missing_personalization(self):
        model = IALS(n_factors=2,n_iterations=1).fit(self.data)
        model.user_factors_[:]=1
        model.item_factors_[:]=-1
        self.assertTrue(np.isfinite(model.score_batch(['a'])).all())
        self.assertTrue((model.score_batch(['a']) < 0).all())


if __name__ == '__main__':
    unittest.main()
