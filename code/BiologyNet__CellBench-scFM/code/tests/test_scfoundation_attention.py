import copy
from types import SimpleNamespace
import unittest

import torch

from bio.models.compat.scfoundation_attention import configure_query_blocks


class ScFoundationAttentionTests(unittest.TestCase):
    def test_query_blocks_match_native_weights_masks_and_residuals(self):
        torch.manual_seed(41)
        for norm_first in [False, True]:
            reference = torch.nn.TransformerEncoderLayer(
                24, 4, dim_feedforward=48, batch_first=True, norm_first=norm_first).double().eval()
            blocked = copy.deepcopy(reference)
            before = {k: v.clone() for k, v in blocked.state_dict().items()}
            configure_query_blocks(SimpleNamespace(transformer_encoder=[blocked]), chunk_size=5)
            x = torch.randn(2, 17, 24, dtype=torch.float64)
            mask = torch.zeros(2, 17, dtype=torch.bool)
            mask[0, -3:] = True
            for padding in [None, mask, torch.zeros_like(mask, dtype=torch.float64).masked_fill(mask, -float('inf'))]:
                with torch.inference_mode():
                    expected = reference(x, src_key_padding_mask=padding)
                    actual = blocked(x, src_key_padding_mask=padding)
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            for name, value in blocked.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
