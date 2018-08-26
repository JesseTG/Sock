import pytest
import torch

from torch import Tensor

from .marks import *

SIZE = torch.Size([100, 100, 100])


@pytest.fixture(scope="module")
def data_cpu():
    return torch.rand(SIZE, dtype=torch.float64, device="cpu")


@pytest.fixture(scope="module")
def data_cuda():
    return torch.rand(SIZE, dtype=torch.float64, device="cuda")


@pytest.mark.benchmark(group="test_bench_pin")
def test_bench_pin(benchmark, data_cpu: Tensor):
    result = benchmark(Tensor.pin_memory, data_cpu)  # type: Tensor

    assert result is not None
    assert result.is_pinned()


@needs_cuda
@pytest.mark.benchmark(group="test_bench_copy_cpu_to_cuda")
@pytest.mark.parametrize("pin", [False, True], ids=("unpinned", "pinned"))
@pytest.mark.parametrize("non_blocking", [False, True], ids=("blocking", "non_blocking"))
def test_bench_copy_cpu_to_cuda_new_tensor(benchmark, data_cpu: Tensor, pin: bool, non_blocking: bool):
    data = data_cpu.pin_memory() if pin else data_cpu

    result = benchmark(data.cuda, non_blocking=non_blocking)

    assert result is not None
    assert result.is_cuda


@needs_cuda
@pytest.mark.benchmark(group="test_bench_copy_cpu_to_cuda")
@pytest.mark.parametrize("pin", [False, True], ids=("unpinned", "pinned"))
@pytest.mark.parametrize("non_blocking", [False, True], ids=("blocking", "non_blocking"))
def test_bench_copy_cpu_to_cuda_existing_tensor(benchmark, data_cpu: Tensor, pin: bool, non_blocking: bool):
    data = data_cpu.pin_memory() if pin else data_cpu
    destination = torch.empty_like(data_cpu, device="cuda")
    destination_addr = destination.data_ptr()
    result = benchmark(destination.copy_, data, non_blocking=non_blocking)

    assert result is not None
    assert result.data_ptr() == destination_addr
    assert result.is_cuda


@needs_cuda
@pytest.mark.benchmark(group="test_bench_copy_cuda_to_cpu")
@pytest.mark.parametrize("non_blocking", [False, True], ids=("blocking", "non_blocking"))
def test_bench_copy_cuda_to_cpu_new_tensor(benchmark, data_cuda: Tensor, non_blocking: bool):
    result = benchmark(data_cuda.to, device="cpu", non_blocking=non_blocking)

    assert result is not None
    assert not result.is_cuda


@needs_cuda
@pytest.mark.benchmark(group="test_bench_copy_cuda_to_cpu")
@pytest.mark.parametrize("pin", [False, True], ids=("unpinned", "pinned"))
@pytest.mark.parametrize("non_blocking", [False, True], ids=("blocking", "non_blocking"))
def test_bench_copy_cuda_to_cpu_existing_tensor(benchmark, data_cuda: Tensor, pin: bool, non_blocking: bool):
    destination = torch.empty_like(data_cuda, device="cpu")
    destination = destination.pin_memory() if pin else destination
    destination_addr = destination.data_ptr()

    result = benchmark(destination.copy_, data_cuda, non_blocking=non_blocking)

    assert result is not None
    assert result.data_ptr() == destination_addr
    assert not result.is_cuda
    assert result.is_pinned() == pin
