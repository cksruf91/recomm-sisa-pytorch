from abc import ABCMeta, abstractmethod

import numpy as np


class Metrics(metaclass=ABCMeta):

    @abstractmethod
    def __str__(self):
        """ return display name """
        pass


class Accuracy(Metrics):

    def __str__(self):
        return "acc"

    def __call__(self, output: list, label: list) -> float:
        """get accuracy
        Args:
            output : model prediction, dim: [total data size]
            label : label, dim: [total data size]
        Returns:
            float: accuracy
        """
        assert len(output) == len(label)

        total = len(output)
        label_array = np.array(label)
        output_array = np.array(output)

        match = np.sum(label_array == output_array)
        return match / total
