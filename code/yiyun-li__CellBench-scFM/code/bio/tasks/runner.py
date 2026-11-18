"""Dispatch configured tasks without knowing model-specific preprocessing."""
TASK_METHODS = {
    'embedding_extraction': 'embedding_extraction',
    'clustering': 'no_finetuning_cluster',
    'correction': 'no_finetuning_correction',
    'reconstruction': 'adapter_finetuning_reconstruction',
    'perturbation': 'adapter_finetuning_perturbation',
}


def run_model(cfgs, training='adapter_finetuning'):
    from bio.models import get_model
    task = cfgs['model_cfg']['tasks']
    if task == 'annotation':
        if training not in ('adapter_finetuning', 'full_finetuning'):
            raise ValueError(f'Unknown training phase: {training}')
        method = training + '_annotation'
    elif task in TASK_METHODS:
        method = TASK_METHODS[task]
    else:
        raise ValueError(f'Unknown task: {task}')
    model = get_model(cfgs['model_cfg']['model'])(cfgs, training)
    return getattr(model, method)(cfgs)

