from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch


class LogBuffer:
    """A buffer that stores the history of values and outputs the average."""

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()

    def clear(self):
        """Clear the history."""
        self.val_history.clear()
        self.n_history.clear()

    def update(self, vars, count=1):
        """Update the history with the given variables.

        Args:
            vars (dict): A dictionary of variables.
            count (int): The number of samples.
        """
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            if isinstance(var, torch.Tensor):
                var = var.item()
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n=0):
        """Calculate the average of the history.

        Args:
            n (int): The number of latest values to be averaged. If n is 0,
                average all values.
        """
        assert n >= 0
        output = OrderedDict()
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            output[key] = avg
        return output

    def summary(self, n=0):
        """Calculate the average of the history and return the output.

        Args:
            n (int): The number of latest values to be averaged. If n is 0,
                average all values.
        Returns:
            dict: A dictionary of the average values.
        """
        output = self.average(n)
        self.clear()
        return output
