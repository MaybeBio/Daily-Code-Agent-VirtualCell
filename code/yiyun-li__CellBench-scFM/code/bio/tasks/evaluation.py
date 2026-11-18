"""Task orchestration; model encoders and metrics are supplied by the adapter."""
from bio.io.results import log_metrics


class MetricMixin:
    def metric(self, input, task):
        from bio import metrics as evaluation
        options = self.cfgs.get('evaluation_cfg', {})
        if task == 'clustering':
            result, _ = evaluation.clustering(input['cell_emb'], input['label'], options, self.seed)
        elif task == 'correction':
            result = evaluation.correction(input['embedding'], input['batch_labels'],
                                           input['cell_type_labels'], options, self.seed)
            result['representation_protocol'] = options.get('correction_mode', 'native')
        elif task == 'annotation':
            result = evaluation.annotation(input['pred'], input['label'], input.get('probabilities'),
                                           input.get('classes'))
        elif task == 'reconstruction':
            result = evaluation.reconstruction(input['recon'], input['truth'],
                                               options.get('reconstruction_top_k', 50))
        elif task == 'perturbation':
            result = evaluation.perturbation(input['recon'], input['truth'], input['ctrl'], input['groups'],
                                             options.get('perturbation_top_k', 50),
                                             options.get('min_effect', 0.0),
                                             control_cells=input.get('control_cells'),
                                             control_labels=input.get('control_labels'),
                                             fdr=options.get('des_fdr', 0.05))
            result['control_reference'] = 'celltype_mean_of_observed_controls'
        else:
            raise ValueError(f'Unknown evaluation task: {task}')
        if input.get('split'):
            result.update({k: input['split'][k] for k in input['split']})
        result['hvg_source'] = options.get('hvg_source', 'none')
        result['metric_version'] = evaluation.METRIC_VERSION
        result['expression_scale'] = self.cfgs['data_cfg'].get('evaluation_expression_scale', 'dataset_defined')
        log_metrics(self.cfgs, result, task=task)
        return result
