"""Neural developmental encoding.

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ]

"""

# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
import numpy.typing as npt
import torch
from rich.console import Console
from rich.traceback import install
from torch import nn

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)

# Global constants
# Global functions
# Warning Control
# Type Checking
# Type Aliases

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- TERMINAL OUTPUT SETUP --- #
install(show_locals=False)
console = Console()


class NeuralDevelopmentalEncoding(nn.Module):
    def __init__(self, number_of_modules: int, genotype_size: int = 64) -> None:
        super().__init__()
        """
        Neural developmental encoder.

        Given a genotype (list of chromosomes), output the phenotype
        (probability matrices corresponding to module types, connections, rotations).

        Parameters
        ----------
        number_of_modules : int
            Number of modules in the robot.
        genotype_size : int, optional
            Size of each genotype chromosome, by default 64.
        """

        # Hidden Layers
        self.fc1 = nn.Linear(genotype_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)

        # ------------------------------------------------------------------- #
        # OUTPUTS
        self.type_p_shape = (number_of_modules, NUM_OF_TYPES_OF_MODULES)
        self.type_p_out = nn.Linear(
            128,
            number_of_modules * NUM_OF_TYPES_OF_MODULES,
        )

        self.conn_p_shape = (number_of_modules, number_of_modules, NUM_OF_FACES)
        self.conn_p_out = nn.Linear(
            128,
            number_of_modules * number_of_modules * NUM_OF_FACES,
        )

        self.rot_p_shape = (number_of_modules, NUM_OF_ROTATIONS)
        self.rot_p_out = nn.Linear(
            128,
            number_of_modules * NUM_OF_ROTATIONS,
        )

        self.output_layers = [
            self.type_p_out,
            self.conn_p_out,
            self.rot_p_out,
        ]
        self.output_shapes = [
            self.type_p_shape,
            self.conn_p_shape,
            self.rot_p_shape,
        ]
        # ------------------------------------------------------------------- #

        # Activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        genotype: list[npt.NDArray[np.float32]],
    ) -> list[npt.NDArray[np.float32]]:
        """
        Forward pass through the neural developmental encoder.

        Parameters
        ----------
        genotype : list[npt.NDArray[np.float32]]
            List of chromosomes (numpy arrays).
        Returns
        -------
        list[npt.NDArray[np.float32]]
            List of phenotype outputs (numpy arrays).
        """

        outputs: list[npt.NDArray[np.float32]] = []
        for idx, chromosome in enumerate(genotype):
            with torch.no_grad():  # double safety
                np_chromosome = np.array(chromosome)
                x = torch.from_numpy(np_chromosome).to(torch.float32)

                x = self.fc1(x)
                x = self.relu(x)

                x = self.fc2(x)
                x = self.tanh(x)

                x = self.fc3(x)
                x = self.relu(x)

                x = self.fc4(x)
                x = self.relu(x)

                x = self.output_layers[idx](x)
                x = self.sigmoid(x)

                x = x.view(self.output_shapes[idx])
                outputs.append(x.detach().numpy())
        return outputs


if __name__ == "__main__":
    """Usage example."""
    nde = NeuralDevelopmentalEncoding(number_of_modules=20)

    genotype_size = 64
    type_p_genes = RNG.random(genotype_size)
    conn_p_genes = RNG.random(genotype_size)
    rot_p_genes = RNG.random(genotype_size)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    outputs = nde.forward(genotype)
    for output in outputs:
        console.log(output.shape)
