"""Small synthetic checks for fit-only cohort selection; no recommender training."""
import unittest
import pandas as pd
from prepare import QUERY, select_fit


class FitSelectionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([(f'u{u}', f'i{i}', 4, QUERY - 100) for u in range(6) for i in range(6)],
                            columns=['user_id', 'item_id', 'rating', 'timestamp'])

    def test_item_cap_and_whole_user_budget_preserve_fit_core(self):
        data = self.frame()
        selected, diagnostics = select_fit(data, max_items=5, max_events=25)
        self.assertEqual(len(selected), 25)
        self.assertEqual(set(selected.item_id), {f'i{i}' for i in range(5)})
        self.assertTrue(diagnostics['user_budget_applied'])
        self.assertTrue((selected.groupby('user_id').size() >= 5).all())
        self.assertTrue((selected.groupby('item_id').size() >= 5).all())
        again, _ = select_fit(data.sample(frac=1, random_state=19), max_items=5, max_events=25)
        pd.testing.assert_frame_equal(selected, again)

    def test_future_rows_rejected_and_ratings_do_not_control_selection(self):
        data = self.frame()
        expected, _ = select_fit(data, max_items=5)
        changed = data.copy()
        changed['rating'] = 1
        actual, _ = select_fit(changed, max_items=5)
        pd.testing.assert_frame_equal(expected.drop(columns='rating'), actual.drop(columns='rating'))
        data.loc[0, 'timestamp'] = QUERY
        with self.assertRaisesRegex(ValueError, 'strictly before 2019'):
            select_fit(data)


if __name__ == '__main__':
    unittest.main()
