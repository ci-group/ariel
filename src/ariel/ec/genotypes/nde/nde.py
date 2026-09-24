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

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

install(show_locals=False)
console = Console()


def reflect_to_unit_interval(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Reflect values into the interval [0, 1].

    Reflection avoids the boundary pile-up produced by clipping.

    Examples
    --------
    -0.2 -> 0.2
     1.2 -> 0.8
     2.2 -> 0.2
    """
    reflected = np.mod(values, 2.0)
    reflected = np.where(
        reflected > 1.0,
        2.0 - reflected,
        reflected,
    )
    return reflected.astype(np.float32)


class NeuralDevelopmentalEncoding(nn.Module):
    def __init__(self, number_of_modules: int, genotype_size: int = 64) -> None:
        super().__init__()
        """
        Neural developmental encoder.

        Given a genotype (list of chromosomes), output the phenotype
        (probability matrices corresponding to module types, connections,
        rotations, and optionally directly encoded variable brick lengths).

        Parameters
        ----------
        number_of_modules : int
            Number of modules in the robot.
        genotype_size : int, optional
            Size of each genotype chromosome, by default 64.
        """

        if number_of_modules > genotype_size:
            msg = (
                "number_of_modules cannot exceed genotype_size when variable "
                "brick lengths are encoded directly."
            )
            raise ValueError(msg)

        self.number_of_modules = number_of_modules
        self.genotype_size = genotype_size

        self.fc1 = nn.Linear(genotype_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)

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

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        genotype: list[npt.NDArray[np.float32]],
    ) -> list[npt.NDArray[np.float32]]:
        """Forward pass through the neural developmental encoder.

        Parameters
        ----------
        genotype : list[npt.NDArray[np.float32]]
            The first three chromosomes encode type, connection, and rotation.
            An optional fourth chromosome directly encodes normalized
            per-module brick lengths.

        Returns
        -------
        list[npt.NDArray[np.float32]]
            Phenotype outputs.
        """
        if len(genotype) > 4:
            msg = f"Expected at most 4 chromosomes, got {len(genotype)}"
            raise ValueError(msg)

        if len(genotype) < 3:
            msg = f"Expected at least 3 chromosomes, got {len(genotype)}"
            raise ValueError(msg)

        outputs: list[npt.NDArray[np.float32]] = []

        for idx, chromosome in enumerate(genotype[:3]):
            with torch.no_grad():
                np_chromosome = np.asarray(
                    chromosome,
                    dtype=np.float32,
                )

                if np_chromosome.shape != (self.genotype_size,):
                    msg = (
                        f"Chromosome {idx} must have shape "
                        f"({self.genotype_size},), got {np_chromosome.shape}"
                    )
                    raise ValueError(msg)

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

        if len(genotype) == 4:
            length_chromosome = np.asarray(
                genotype[3],
                dtype=np.float32,
            )

            if length_chromosome.shape != (self.genotype_size,):
                msg = (
                    "Length chromosome must have shape "
                    f"({self.genotype_size},), got {length_chromosome.shape}"
                )
                raise ValueError(msg)

            length_p = reflect_to_unit_interval(
                length_chromosome[: self.number_of_modules]
            )

            outputs.append(length_p)

        return outputs


if __name__ == "__main__":
    nde = NeuralDevelopmentalEncoding(number_of_modules=20)

    genotype_size = 64
    type_p_genes = RNG.random(genotype_size)
    conn_p_genes = RNG.random(genotype_size)
    rot_p_genes = RNG.random(genotype_size)
    length_p_genes = RNG.random(genotype_size)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
        length_p_genes,
    ]

    outputs = nde.forward(genotype)

    for output in outputs:
        console.log(output.shape)