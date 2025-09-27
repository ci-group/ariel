import torch
from torch.ao.quantization.observer import ObserverBase as ObserverBase

class ModelReportObserver(ObserverBase):
    """This observer is used to record additional information regarding keeping track
    of S = average_batch_activation_range/epoch_activation_range.

    The purpose of this information is to prepare a report to present to users on whether
    Dynamic or Static Quantization is more appropriate for their model given the general
    distributions of their data.

    Args:
        ch_axis (int, optional): The channel axis for which the range and outlier stats are computed
            Default: 1
        comp_percentile (float, optional): The percentile to compare against 100 percentile to find outliers
            Should be between 0 and 1 exclusive
            Default: 0.9

    * :attr:`num_batches_tracked` specifies number of batches passed through the observer

    * :attr:`average_batch_activation_range` defines average across the ranges of each batch passed through

    * :attr:`epoch_activation_min` defines the minimum value passed through the observer

    * :attr:`epoch_activation_max` defines the maximum value passed through the observer

    * :attr:`ch_axis` defines the channel being used to compute per channel min max stats

    * :attr:`min_val` defines the per channel minimum values passed through

    * :attr:`max_val` defines the per channel maximum values passed through

    * :attr:`comp_percentile` defines comparison percentile to find outliers

    * :attr:`average_percentile_ratio` defines the per channel average percentile ratios

    * :attr:`percentile_batches_tracked` defines the number of percentile batches tracked for each channel

    * :attr:`constant_channels` defines the number of batches that aren't constant channels per channel

    Note: this tool is meant for FX Graph Mode Quantization
    """
    epoch_activation_min: torch.Tensor
    epoch_activation_max: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor
    comp_percentile: torch.Tensor
    average_percentile_ratio: torch.Tensor
    percentile_batches_tracked: torch.Tensor
    constant_channels: torch.Tensor
    num_batches_tracked: int
    average_batch_activation_range: torch.Tensor
    ch_axis: int
    def __init__(self, ch_axis: int = 1, comp_percentile: float = 0.9) -> None: ...
    def forward(self, x): ...
    def _calculate_range_stats(self, x_copy):
        """Calculates and stores range stats with forward values.

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
    def _calculate_min_max_stats(self, x_copy):
        """Calculates and stores the per_channel min, max stats with forward values.
        Does calculation based on channel axis: self.ch_axis

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
    def _calculate_percentile_stats(self, x_copy):
        """Calculates and stores the per_channel percentile stats with forward values.
        Does calculation based on channel axis: self.ch_axis

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
    @torch.jit.export
    def get_batch_to_epoch_ratio(self): ...
    @torch.jit.export
    def reset_batch_and_epoch_values(self) -> None: ...
    @torch.jit.export
    def calculate_qparams(self) -> None: ...
