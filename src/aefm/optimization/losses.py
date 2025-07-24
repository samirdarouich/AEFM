import torch

__all__ = [
    "TimeWeight",
    "SigmoidTimeWeight",
]


class TimeWeight(torch.nn.Module):
    def forward(self, t: torch.Tensor):
        raise NotImplementedError


class SigmoidTimeWeight(TimeWeight):
    """
    SigmoidTimeWeight applies a sigmoid function to a given tensor of time values.

    Args:
        switch_time (float): The time at which the sigmoid function switches.
        rate (float): The steepness of the sigmoid function.
    """

    def __init__(self, switch_time: float, rate: float) -> None:
        super().__init__()
        self.switch_time = switch_time
        self.rate = rate

    def forward(self, t: torch.Tensor):
        """
        Forward pass to apply the sigmoid function to the input tensor.

        Args:
            t (torch.Tensor): A tensor of time values.

        Returns:
            torch.Tensor: A tensor with the sigmoid function applied.
        """
        return 1 / (1 + torch.exp(-self.rate * (t - self.switch_time)))
