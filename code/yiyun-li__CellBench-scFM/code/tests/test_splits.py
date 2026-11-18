"""Donor hold-out splits must not leak the same donor into train and test."""
import unittest
import numpy as np

from bio.data.splits import split_indices


class SplitTests(unittest.TestCase):
    def test_donor_holdout_keeps_donors_disjoint(self):
        donors = np.repeat(['d1', 'd2', 'd3', 'd4', 'd5'], 8)
        labels = np.tile(['T', 'B'], 20)
        train, val, test, info = split_indices(len(donors), labels, groups=donors, seed=41, test_size=0.2, val_size=0.2)
        self.assertEqual(info['mode'], 'donor')
        self.assertEqual(set(donors[train]) & set(donors[test]), set())
        self.assertEqual(set(donors[train]) & set(donors[val]), set())
        self.assertEqual(set(donors[val]) & set(donors[test]), set())
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertGreater(len(val), 0)

    def test_single_donor_falls_back_to_cell_split(self):
        donors = np.repeat(['only'], 20)
        labels = np.tile(['T', 'B'], 10)
        train, val, test, info = split_indices(20, labels, groups=donors, seed=41, test_size=0.2)
        self.assertEqual(info['mode'], 'cell')
        self.assertEqual(set(donors[train]), {'only'})
        self.assertEqual(len(train) + len(val) + len(test), 20)

    def test_test_labels_are_seen_in_train(self):
        donors = np.array(['a'] * 10 + ['b'] * 10)
        labels = np.array(['T'] * 10 + ['B'] * 10)
        with self.assertRaisesRegex(ValueError, 'no test cells'):
            split_indices(20, labels, groups=donors, seed=0, test_size=0.5)

    def test_condition_holdout_keeps_knockouts_disjoint(self):
        from bio.data.splits import split_conditions
        conditions = np.repeat(['KO1', 'KO2', 'KO3', 'KO4', 'KO5'], 6)
        train, val, test, info = split_conditions(conditions, seed=41, test_size=0.2, val_size=0.2)
        self.assertEqual(info['mode'], 'unseen_condition')
        self.assertEqual(set(conditions[train]) & set(conditions[test]), set())
        self.assertEqual(set(conditions[train]) & set(conditions[val]), set())
        self.assertEqual(set(conditions[val]) & set(conditions[test]), set())
        self.assertGreater(info['n_test_conditions'], 0)
        self.assertGreater(info['n_train_conditions'], 0)

    def test_condition_split_is_deterministic(self):
        from bio.data.splits import split_conditions
        conditions = np.array(['A'] * 4 + ['B'] * 4 + ['C'] * 4)
        a = split_conditions(conditions, seed=3, test_size=0.3, val_size=0.3)
        b = split_conditions(conditions, seed=3, test_size=0.3, val_size=0.3)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])

    def test_norman_underscore_is_a_gene_combo(self):
        from bio.data.pert_genes import parse_perturbation_genes, is_valid_perturbation
        self.assertEqual(parse_perturbation_genes('AHR_FEV'), ['AHR', 'FEV'])
        self.assertEqual(parse_perturbation_genes('AHR+FEV'), ['AHR', 'FEV'])
        self.assertEqual(parse_perturbation_genes('AHR'), ['AHR'])
        self.assertFalse(is_valid_perturbation('*'))
        self.assertFalse(is_valid_perturbation('ctrl'))
        self.assertTrue(is_valid_perturbation('AHR_FEV'))

    def test_combo_holdout_trains_singles_and_tests_doubles(self):
        from bio.data.splits import split_perturbations
        singles = [f'G{i}' for i in range(12)]
        combos = [f'G{i}_G{i+1}' for i in range(12)]
        conditions = np.array(singles * 3 + combos * 3)
        train, val, test, info = split_perturbations(conditions, seed=41, mode='auto')
        self.assertEqual(info['mode'], 'unseen_combo')
        self.assertEqual(set(conditions[train]), set(singles))
        self.assertTrue(set(conditions[test]).issubset(set(combos)))
        self.assertEqual(set(conditions[train]) & set(conditions[test]), set())

    def test_default_cell_split_is_exact_70_10_20(self):
        labels = np.tile(['T', 'B'], 50)
        train, val, test, _ = split_indices(100, labels)
        self.assertEqual(tuple(map(len, (train, val, test))), (70, 10, 20))
        self.assertEqual(len(set(train) | set(val) | set(test)), 100)
        self.assertFalse(set(train) & set(val) or set(train) & set(test) or set(val) & set(test))

    def test_default_donor_split_holds_out_10_and_20_percent(self):
        donors = np.repeat([f'd{i}' for i in range(10)], 10)
        labels = np.tile(['T', 'B'], 50)
        train, val, test, info = split_indices(100, labels, groups=donors)
        self.assertEqual(tuple(map(len, (train, val, test))), (70, 10, 20))
        self.assertEqual((info['n_train_donors'], info['n_val_donors'], info['n_test_donors']), (7, 1, 2))
        self.assertFalse(set(donors[train]) & set(donors[val]))
        self.assertFalse(set(donors[train]) & set(donors[test]))
        self.assertFalse(set(donors[val]) & set(donors[test]))

    def test_default_condition_split_is_70_10_20_conditions(self):
        from bio.data.splits import split_perturbations
        conditions = np.repeat([f'G{i}' for i in range(10)], 10)
        train, val, test, info = split_perturbations(conditions)
        self.assertEqual(tuple(map(len, (train, val, test))), (70, 10, 20))
        self.assertEqual((info['n_train_conditions'], info['n_val_conditions'], info['n_test_conditions']), (7, 1, 2))
        self.assertFalse(set(conditions[train]) & set(conditions[val]))
        self.assertFalse(set(conditions[train]) & set(conditions[test]))
        self.assertFalse(set(conditions[val]) & set(conditions[test]))

    def test_split_is_deterministic(self):
        donors = np.repeat(['d1', 'd2', 'd3'], 6)
        labels = np.tile(['T', 'B', 'M'], 6)
        a = split_indices(18, labels, groups=donors, seed=7, test_size=0.3)
        b = split_indices(18, labels, groups=donors, seed=7, test_size=0.3)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])


if __name__ == '__main__':
    unittest.main()
