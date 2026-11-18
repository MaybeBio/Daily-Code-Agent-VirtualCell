"""Shared split helpers used by downstream tasks."""
import numpy as np


class TaskSupportMixin:
    def evaluation_split(self, cell_ids, task):
        from bio.data.splits import evaluation_indices
        options = self.cfgs.get('evaluation_cfg', {})
        return evaluation_indices(
            self.adata_all[np.asarray(cell_ids)], task=task, seed=self.seed,
            split_by=options.get('split_by', 'donor'),
            perturbation_split=options.get('perturbation_split', 'auto'))

    def donor_groups(self, cell_ids):
        if self.cfgs.get('evaluation_cfg', {}).get('split_by', 'donor') != 'donor':
            return None
        if 'donor_id' not in self.adata_all.obs:
            return None
        return self.adata_all.obs['donor_id'].astype(str).to_numpy()[np.asarray(cell_ids)]
