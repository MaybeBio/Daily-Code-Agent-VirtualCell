"""GPU scheduling options and selectors within the caller's visible devices."""
import os


def add_gpu_options(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', type=int, help='One visible GPU index (default 0)')
    group.add_argument('--gpus', help='Comma-separated visible GPU indices, e.g. 0,1,2,3')
    parser.add_argument('--max-workers', '--jobs', type=int,
                        help='Maximum concurrent tasks (default one per selected GPU)')
    parser.add_argument('--tasks-per-gpu', type=int, default=1,
                        help='Maximum tasks sharing one GPU; default 1')


def gpu_schedule(gpu=None, gpus=None, max_workers=None, tasks_per_gpu=1):
    if gpu is not None and gpus is not None:
        raise ValueError('Use either --gpu or --gpus')
    try:
        ids = [0 if gpu is None else int(gpu)] if gpus is None else [int(p.strip()) for p in gpus.split(',')]
    except (ValueError, AttributeError):
        raise ValueError('--gpus must be comma-separated integers') from None
    if not ids or any(i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('GPU indices must be nonnegative and unique')
    workers = len(ids) if max_workers is None else max_workers
    if workers < 1 or tasks_per_gpu < 1:
        raise ValueError('--max-workers and --tasks-per-gpu must be positive')
    if workers > len(ids) * tasks_per_gpu:
        raise ValueError('--max-workers exceeds GPU slots; select more GPUs or explicitly increase --tasks-per-gpu')
    return ids, workers


def cuda_selector(index, environ=None):
    """Map a logical index through an existing mask, including GPU/MIG UUIDs."""
    env = os.environ if environ is None else environ
    if index < 0:
        raise ValueError('GPU index must be nonnegative')
    if 'CUDA_VISIBLE_DEVICES' not in env:
        return str(index)
    visible = [p.strip() for p in env['CUDA_VISIBLE_DEVICES'].split(',') if p.strip()]
    if (not visible or visible == ['-1']) and index == 0:
        return ''  # Keep an explicitly CPU-only allocation masked.
    if not visible or visible == ['-1'] or index >= len(visible):
        raise ValueError(f'GPU index {index} is outside CUDA_VISIBLE_DEVICES')
    return visible[index]
