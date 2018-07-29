# write a subclass of torch.data.Sampler based on SMOTE

# it just has to iterate over the parts of the data set I want
# http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html

# use the smote sampler just like i did when training sklearn models
# then at the end just iterate over them inside my sampler subclass

from enum import Enum
from typing import Sequence
import numpy
from numpy import ndarray
from torch.utils.data.sampler import Sampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek


class SamplerMode(Enum):
    INDICES = 0
    INDICES_AND_LABELS = 1


# NOTE: I might have misread the paper.  I don't think I need SMOTE at all!
# NOTE: If I were to use SMOTE, it would actually be better as a DataSet subclass, since it generates samples

class SMOTETomekSampler(Sampler):
    def __init__(self, indices: Sequence[int], classes: Sequence[int], mode: SamplerMode=SamplerMode.INDICES):
        if len(indices) != len(classes):
            raise ValueError(
                f"lengths of indices ({len(indices)}) and classes ({len(classes)}) must match"
            )

        self.indices = numpy.reshape(indices, (-1, 1))
        self.classes = numpy.reshape(classes, (-1, 1))
        self.min_index = self.indices.min()
        self.max_index = self.indices.max()
        self.mode = mode

    def __len__(self) -> int:
        # TODO: Give proper length, see the SMOTETomek docs
        return len(self.indices)

    def __iter__(self):
        smotetomek = SMOTETomek()
        samples = smotetomek.fit_sample(self.indices, self.classes)
        resampled_indices = samples[0].ravel().round().astype(int).clip(min=self.min_index, max=self.max_index)
        # TODO: Must these be int?  If they're numpy.int64, does that help performance?

        if self.mode == SamplerMode.INDICES:
            return iter(resampled_indices)
        elif self.mode == SamplerMode.INDICES_AND_LABELS:
            resampled_labels = samples[1].astype(int)  # type: ndarray
            return iter(zip(resampled_indices, resampled_labels))
