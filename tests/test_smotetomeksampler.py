
import pytest
import numpy
from numpy import array, arange, ndarray, ones, zeros, int8
from sockpuppet.model.sampling.smote import SMOTETomekSampler, SamplerMode

INDICES = arange(1000)
CLASSES = numpy.concatenate([zeros(200, int8), ones(800, int8)])


@pytest.fixture(scope="module")
def smotetomek_sampler():
    return SMOTETomekSampler(INDICES, CLASSES)


@pytest.fixture(scope="module")
def smotetomek_sampler_with_labels():
    return SMOTETomekSampler(INDICES, CLASSES, SamplerMode.INDICES_AND_LABELS)


def test_sampler_created(smotetomek_sampler: SMOTETomekSampler):
    assert smotetomek_sampler is not None


def test_sampler_requires_lists_of_same_size():
    with pytest.raises(ValueError):
        SMOTETomekSampler([1, 2, 3], [0, 1, 0, 0])


@pytest.mark.skip
def test_sampler_is_random(smotetomek_sampler: SMOTETomekSampler):
    samples1 = [s for s in smotetomek_sampler]
    samples2 = [s for s in smotetomek_sampler]

    assert samples1 != samples2


@pytest.mark.skip
def test_sampler_yields_indices(smotetomek_sampler: SMOTETomekSampler):
    for s in smotetomek_sampler:
        assert numpy.can_cast(s, int)


@pytest.mark.skip
def test_sampler_yields_indices_and_labels(smotetomek_sampler_with_labels: SMOTETomekSampler):
    for s in smotetomek_sampler_with_labels:
        assert isinstance(s, tuple)
        assert numpy.can_cast(s[0], int)
        assert numpy.can_cast(s[1], int)


@pytest.mark.skip
def test_sampler_splits_classes_evenly(smotetomek_sampler_with_labels: SMOTETomekSampler):
    labels = [s[1] for s in smotetomek_sampler_with_labels]

    zeroes = len([s for s in labels if s == 0])
    ones = len([s for s in labels if s == 1])
    assert zeroes == pytest.approx(ones, abs=1)
    # plus or minus one class is acceptable


@pytest.mark.skip
def test_sampler_matches_classes(smotetomek_sampler_with_labels: SMOTETomekSampler):
    samples = [s for s in smotetomek_sampler_with_labels]

    assert all(label == CLASSES[index] for index, label in samples)
