import unittest

import numpy as np
import pandas as pd
from anndata import AnnData


class OrthologTests(unittest.TestCase):
    def test_mouse_matrix_is_renamed_to_human_ensembl(self):
        from bio.data import orthologs as module
        module._MAPPING = {'ENSMUSG00000000001': 'ENSG00000000001',
                           'ENSMUSG00000000002': 'ENSG00000000002'}
        obs = pd.DataFrame({'celltype': ['T', 'B']})
        var = pd.DataFrame(index=['ENSMUSG00000000001', 'ENSMUSG00000000099', 'ENSMUSG00000000002'])
        data = AnnData(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), obs=obs, var=var)
        out = module.project_mouse_to_human_ensembl(data)
        self.assertEqual(list(out.var_names), ['ENSG00000000001', 'ENSG00000000002'])
        np.testing.assert_array_equal(out.X, np.array([[1, 3], [4, 6]], dtype=np.float32))
        self.assertEqual(list(data.var_names), ['ENSMUSG00000000001', 'ENSMUSG00000000099', 'ENSMUSG00000000002'])

    def test_human_matrix_is_unchanged(self):
        from bio.data.orthologs import project_mouse_to_human_ensembl
        var = pd.DataFrame(index=['ENSG00000123456', 'CD3D'])
        data = AnnData(np.ones((2, 2), dtype=np.float32), var=var)
        out = project_mouse_to_human_ensembl(data)
        self.assertIs(out, data)
