"""CUDA checks for the stable logical-to-physical staging table."""

import pytest
import torch

from freetoken.kvcache.kv_staging import KVStagingBinding


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_staging_binding_updates_and_rejects_stale_generations() -> None:
    binding = KVStagingBinding(4, torch.device("cuda"))
    binding.update((1, 3), (2, 5), (7, 11))
    binding.validate((1, 3), (7, 11))
    with pytest.raises(RuntimeError, match="cold or stale"):
        binding.validate((1, 3), (7, 12))
    binding.clear((1,))
    with pytest.raises(RuntimeError, match="cold or stale"):
        binding.validate((1,), (7,))
