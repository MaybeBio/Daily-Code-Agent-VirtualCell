"""Command line entry point; importing this module does not start an experiment."""
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description='CellBench single-cell representation benchmark')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--cfgs', help='Model YAML; omit in all-task mode to run all registered models')
    parser.add_argument('--phase', default='adapter_finetuning', choices=['adapter_finetuning', 'full_finetuning'])
    parser.add_argument('--task', choices=['embedding_extraction', 'clustering', 'annotation', 'reconstruction', 'perturbation', 'correction', 'all'])
    parser.add_argument('--topk', default=0, type=int,
                        help='Deprecated. Task heads always use all training cells; only 0 is accepted.')
    parser.add_argument('--seed', default='41,42,43',
                        help='One seed or comma-separated seeds (default 41,42,43)')
    parser.add_argument('--hvg', default=None, type=int)
    parser.add_argument('--dataset_name', default=None)
    parser.add_argument('--dataset', help='Run all applicable tasks for this dataset unless --task is specified')
    parser.add_argument('--panel', choices=['organ', 'disease', 'species', 'perturbation'],
                        help='Run every dataset in resources/experiment_panels.json for this section')
    parser.add_argument('--extra-cfg', help='YAML merged after the model config')
    parser.add_argument('--dry-run', action='store_true', help='Preview the all-task plan without running models')
    parser.add_argument('--fail-fast', action='store_true', help='Stop the suite after its first failed task')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--log_to_file', action='store_true', help='Retained for existing command files')
    parser.add_argument('--iter', default=0, help='Retained for existing command files')
    return parser


def _apply_extra_cfg(cfgs, extra_cfg):
    if not extra_cfg:
        return cfgs
    import yaml
    from bio.config import MergeCfgsDict
    from bio.io.paths import project_path
    extra = yaml.safe_load(project_path(extra_cfg).read_text()) or {}
    MergeCfgsDict(extra, cfgs)
    return cfgs


def main(argv=None):
    parser = build_parser()
    opt = parser.parse_args(argv)
    if opt.dataset:
        if opt.dataset_name and opt.dataset_name != opt.dataset:
            parser.error('--dataset and --dataset_name disagree')
        opt.dataset_name = opt.dataset
        opt.task = opt.task or 'all'
    if opt.panel:
        opt.task = opt.task or 'all'
    if opt.task != 'all' and (opt.dry_run or opt.fail_fast) and not opt.panel and not opt.dataset:
        parser.error('--dry-run and --fail-fast require --task all, --dataset, or --panel')
    if opt.topk not in (0, None):
        parser.error('few-shot --topk was removed; task heads always use all training cells')
    from bio.config import config_loader
    from bio.runtime.suite import parse_seeds, run_suite
    if opt.task == 'all' or opt.panel:
        from bio.io.paths import project_path
        if opt.cfgs:
            opt.cfgs = str(project_path(opt.cfgs))
        try:
            return run_suite(opt)
        except ValueError as exc:
            parser.error(str(exc))
    # Suite controllers keep the original mask; only leaf workers select a GPU.
    # Resolve logical indices before importing torch or any adapter.
    if opt.gpu is not None:
        import os
        from bio.runtime.devices import cuda_selector
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_selector(opt.gpu)
        except ValueError as exc:
            parser.error(str(exc))
    seeds = parse_seeds(opt.seed)
    opt.cfgs = opt.cfgs or 'configs/pca.yaml'
    status = 0
    for seed in seeds:
        cfgs = config_loader(opt.cfgs)
        _apply_extra_cfg(cfgs, getattr(opt, 'extra_cfg', None))
        from bio.runtime.seeds import init_seeds
        from bio.io.results import start_run
        from bio.tasks.runner import run_model
        cfgs.update(topk=0, seed=seed, phase=opt.phase)
        if opt.task is not None:
            cfgs['model_cfg']['tasks'] = opt.task
        if opt.dataset_name is not None:
            cfgs['data_cfg']['dataset_name'] = opt.dataset_name
        if opt.hvg is not None:
            cfgs['data_cfg']['hvg'] = opt.hvg
        start_run(cfgs)
        init_seeds(cfgs['seed'])
        run_model(cfgs, opt.phase)
    return status
