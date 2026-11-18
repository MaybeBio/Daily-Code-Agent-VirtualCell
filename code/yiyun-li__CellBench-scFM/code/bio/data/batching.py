"""Training batches that retain every sample without singleton BatchNorm input."""
from torch.utils.data import BatchSampler


class MergeSingletonBatchSampler(BatchSampler):
    """Merge a final one-sample batch into its predecessor (at most size + 1)."""

    def __init__(self, sampler, batch_size):
        if batch_size < 2:
            raise ValueError('BatchNorm training requires batch_size >= 2')
        if len(sampler) < 2:
            raise ValueError('BatchNorm training requires at least two training samples')
        super().__init__(sampler, batch_size=batch_size, drop_last=False)

    def __iter__(self):
        previous = None
        for batch in super().__iter__():
            if previous is not None:
                if len(batch) == 1:
                    yield previous + batch
                    return
                yield previous
            previous = batch
        if previous is not None:
            yield previous

    def __len__(self):
        count = len(self.sampler)
        return super().__len__() - int(count % self.batch_size == 1)
