import pytest
import torch

_num_gpus = torch.cuda.device_count()

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires CUDA")
needs_cudnn = pytest.mark.skipif(not torch.backends.cudnn.is_available(), reason="This test requires CUDNN")
needs_multiple_gpus = pytest.mark.skipif(_num_gpus < 2, reason="This test needs at least 2 CUDA-enabled GPUs")
modes = pytest.mark.modes
cpu_only = modes("cpu")
cuda_only = modes("cuda")
dp_only = modes("dp")
slow = pytest.mark.slow
cpu = pytest.mark.cpu
cuda = pytest.mark.cuda
dp = pytest.mark.dp
report_metrics = pytest.mark.report_metrics

# TODO: Add a mark for skipping test for which a required file doesn't exist
