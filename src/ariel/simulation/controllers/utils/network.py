"""Utility neural-network helpers for simulation controllers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import Tanh


class Network(nn.Module):
    """Simple MLP policy network used in evolutionary controller examples."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.tanh = Tanh()

        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, state: torch.Tensor | list[float]) -> torch.Tensor:
        x = torch.as_tensor(state, dtype=torch.float32)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.tanh(self.fc_out(x)) * (torch.pi / 2)


@torch.no_grad()
def fill_parameters(
    net: nn.Module,
    vector: torch.Tensor,
) -> None:
    """Fill model parameters from a flattened vector."""
    address = 0
    for parameter in net.parameters():
        flattened = parameter.data.view(-1)
        n_values = len(flattened)
        flattened[:] = torch.as_tensor(
            vector[address : address + n_values],
            device=flattened.device,
        )
        address += n_values

    if address != len(vector):
        raise IndexError("The parameter vector is larger than expected")
