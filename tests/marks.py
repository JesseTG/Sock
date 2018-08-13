import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires CUDA")
needs_cudnn = pytest.mark.skipif(not torch.backends.cudnn.is_available(), reason="This test requires CUDNN")
cpu_only = pytest.mark.cpu_only
cuda_only = pytest.mark.cuda_only
