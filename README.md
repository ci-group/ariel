![ariel-header](./docs/resources/ariel_logo.svg)

# ARIEL: Autonomous Robots through Integrated Evolution and Learning

[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-%E2%89%A53.3.6-orange.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)](./LICENSE)
[![Docs](https://img.shields.io/badge/docs-ci--group.github.io%2Fariel-informational.svg)](https://ci-group.github.io/ariel/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Status](https://img.shields.io/badge/status-beta-orange)](https://github.com/ci-group/ariel)

ARIEL is a research framework, developed by the [CI Group](https://cs.vu.nl/ci/) at the Vrije Universiteit Amsterdam, for the joint evolution and learning of modular robots in simulation.

<!-- ## Requirements

* [vscode](https://code.visualstudio.com/)
  * [containers ext](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
  * [container tools ext](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers)

* Container manager:
  * [podman desktop](https://podman.io/)
  * [docker desktop](https://www.docker.com/products/docker-desktop/)
  
* [vscode containers tut](https://code.visualstudio.com/docs/devcontainers/tutorial)

--- -->

## Overview

**ARIEL** is a research framework for the evolution and learning of modular robots. It couples an evolutionary computation (EC) engine with a [MuJoCo](https://mujoco.org/)-based simulation stack, allowing robot *bodies* (morphologies) and *brains* (controllers) to be co-evolved and optimised within a single, consistent pipeline.

The framework is organised around a few core ideas:

- **Genotypes**: encodings that generate robot bodies and/or brains (tree genomes, CPPN/NEAT, and a Neural Developmental Encoding "NDE").
- **Phenotypes**: the RoboGen-lite modular body construction system that turns a genotype into a buildable robot made of cores, bricks and hinges.
- **Controllers**: neural and central-pattern-generator (CPG) controllers that drive the robot's joints.
- **Simulation**: terrains/environments, tasks (locomotion, gait learning, turning) and a MuJoCo worker that evaluates an individual and reports its
  fitness.
- **EC engine**: reusable building blocks (populations, individuals, selection/crossover, archive) for assembling evolutionary algorithms.
- **Visualisation**: dashboards and analysis tools for inspecting results.

## Features

- 🧬 **Body–brain co-evolution**: evolve robot morphologies and controllers jointly, or learn a controller for a fixed body.
- 🤖 **Modular bodies**: RoboGen-lite construction from cores, bricks and hinges, plus a library of prebuilt robots (gecko, spider, …).
- 🧠 **Multiple genotypes**: tree genomes, CPPN/NEAT, and a Neural Developmental Encoding (NDE).
- 🌍 **MuJoCo simulation**: ready-made terrains (flat, rugged, crater, tilted, Olympic arena) and tasks (targeted locomotion, gait learning,
  turning in place).
- 🔁 **Reusable EC engine**: composable populations, individuals, operators, and an archive, decoupled from the simulation layer.
- 💾 **Built-in logging**: runs are persisted to a SQLite database for later analysis and replay.
- 📊 **Visualisation**: dashboards and plotting utilities for results.

## Documentation
[ARIEL main documentation page](https://ci-group.github.io/ariel/)

## Quickstart

After [installing](#installation-and-running), build a prebuilt robot, drop it into a world, and launch the MuJoCo viewer:

```python
import mujoco
from mujoco import viewer

from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# 1. Create a world (a flat terrain)
world = SimpleFlatWorld()

# 2. Build a modular robot body and spawn it into the world
robot = gecko()                       # returns a CoreModule
world.spawn(robot.spec, position=[0, 0, 0.1])

# 3. Compile to a MuJoCo model + data
model = world.spec.compile()
data = mujoco.MjData(model)

# 4. Watch it in the interactive viewer
viewer.launch(model, data)
```

From here, see [`examples/`](examples/) for adding controllers, defining fitness, and running full evolutionary loops.

## Requirements

- **Python** ≥ 3.12
- **MuJoCo** ≥ 3.3.6
- **[uv](https://docs.astral.sh/uv/)** for environment and dependency management

All Python dependencies are declared in [`pyproject.toml`](pyproject.toml) and are installed automatically by `uv sync`.

## Installation and Running

This project uses [uv](https://docs.astral.sh/uv/).

To run the code examples, please do:
1. Clone the repository
```bash
git clone https://github.com/ci-group/ariel.git
cd ariel
```
2. Create a uv virtual environment inside the repository folder
```bash
  uv venv
```
3. Sync the virtual environment with the requirements
```bash
uv sync
```
4. Run an example, in this case, brain evolution (aka learning) using:
```bash
uv run examples/re_book/1_brain_evolution.py
```

## Repository Structure

```
ariel/
├── src/ariel/                      # Main package
│   ├── body_phenotypes/            # Genotype → robot body construction
│   │   ├── robogen_lite/           # RoboGen-lite modular body system
│   │   │   ├── modules/            #   Body parts: core, brick, hinge
│   │   │   ├── decoders/           #   Genotype-to-body decoders (hi-prob, CPPN, vector)
│   │   │   ├── cppn_neat/          #   CPPN/NEAT genome implementation
│   │   │   ├── prebuilt_robots/    #   Ready-made bodies (gecko, spider, ...)
│   │   │   ├── constructor.py      #   Assembles modules into a MuJoCo spec
│   │   │   └── config.py
│   │   └── lynx_mjspec/            # Lynx robot arm body + evolve/replay pipeline
│   ├── ec/                         # Evolutionary computation engine
│   │   ├── genotypes/              #   Encodings: tree, cppn, nde (neural dev. encoding)
│   │   ├── population.py           #   Population container
│   │   ├── individual.py           #   Individual (genotype + fitness + state)
│   │   ├── archive.py              #   Archive of historical individuals
│   │   ├── crossover.py            #   Variation operators
│   │   ├── generators.py           #   Genotype generators + mutators
│   │   └── ea.py                   #   EA orchestration
│   ├── simulation/                 # MuJoCo simulation stack
│   │   ├── environments/           #   Terrains/worlds (flat, rugged, crater, arena, ...)
│   │   ├── tasks/                  #   Tasks (targeted locomotion, gait, turning)
│   │   ├── controllers/            #   Controllers (CPG variants, neural)
│   │   └── mujoco_worker.py        #   Evaluates an individual, returns fitness
│   ├── parameters/                 # Shared types, module defs, MuJoCo params
│   ├── utils/                      # Renderers, trackers, video, optimizers, descriptors
│   └── visualisation/              # Dashboards and analysis tooling
├── examples/                       # Runnable examples
│   ├── a_mujoco/                   #   MuJoCo basics (launcher, rendering, cameras)
│   ├── b_robots/                   #   Building robots from graphs/decoders
│   ├── c_genotypes/                #   Body/brain evolution with genotypes
│   ├── re_book/                    #   "Robot Evolution" book walkthrough examples
│   └── z_ec_course/                #   EC course assignment templates
├── tests/                          # Unit and functional tests
├── docs/                           # Sphinx documentation sources
├── wiki/                           # Project wiki (Obsidian vault)
├── pyproject.toml                  # Project metadata and dependencies (uv)
└── noxfile.py                      # Automation sessions (tests, docs, compiled build)
```

Simulation runs write output to a local `__data__/` directory (created on first run).

## Examples

The [`examples/`](examples/) folder is the best entry point for learning the framework. A good reading order is:

1. [`examples/a_mujoco/`](examples/a_mujoco/): MuJoCo fundamentals: launching a simulation, rendering frames, recording video, and cameras.
2. [`examples/b_robots/`](examples/b_robots/): turning genotypes/graphs into robot bodies and placing them on terrains.
3. [`examples/re_book/`](examples/re_book/): end-to-end brain evolution, then body-brain evolution, learning, and waypoint-following tasks.
4. [`examples/c_genotypes/`](examples/c_genotypes/): body/brain joint evolution, multiprocessing, and replaying results from a database.

## Citation

ARIEL has been accepted for publication at **ALIFE 2026** and **PPSN 2026**. Citation entries for those papers will be added here once they are published.

In the meantime, if you use ARIEL in your research, please cite the repository:

```bibtex
@unpublished{ariel,
    author = {Di Matteo, Jacopo Michele and Grigoriadis, Ioannis and Aron Richard Ferencz and Lilly Schwarzenbach and Agoston Eiben},
    title  = {ARIEL: Autonomous Robots through Integrated Evolution and Learning},
    year   = {2026},
    note   = {https://github.com/ci-group/ariel}
}
```

<!-- TODO: replace with the official ALIFE 2026 and PPSN 2026 citations once published. -->

## Contributing

Contributions are welcome! Please see the [Contributor Guide](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md) before opening an issue or pull request.

## License

Distributed under the terms of the [GPL-3.0 license](LICENSE). ARIEL is free and open source software.

<!-- ## TODO: Installation

## Notes

### This project is managed using `uv`

### Python Code Style Guide

This repository uses the `numpydoc` documentation standard.
For more information checkout: [numpydoc-style guide](https://numpydoc.readthedocs.io/en/latest/format.html#) -->

<!-- ### Units

To ensure that Ariel uses a consistent set of units for all simulations, we use [SI units](https://www.wikiwand.com/en/articles/International_System_of_Units), and (astropy)[https://docs.astropy.org/en/stable/index.html] to enforce it (we automatically convert where we can).

For more information, see: [astropy: units and quantities](https://docs.astropy.org/en/stable/units/index.html) and [astropy: standard units](https://docs.astropy.org/en/stable/units/standard_units.html#standard-units). -->

<!-- ### MuJoCo

#### Attachments

Robot parts should be attached using the `site` functionality (from body to body), while robots should be added to a world using the `frame` functionality (from spec to spec).

- [Python → Attachment](https://mujoco.readthedocs.io/en/stable/python.html#attachment)
- [mjsFrame](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjsframe)

NOTE: when attaching a body, only the contents of `worldbody` get passed, meaning that, for example, `compiler` options are not!

---

## IMPORTANT!!

Change the default configuration of vscode dev containers to accept podman!

Either by the settings gui:

![vscode-podman-settings](./docs/resources/vscode-podman-settings.png)


## Running the code

* In general you can run the currently open python script via the command palette (`cmd+shift+p`): 
  * `Tasks: Run Task` -> `Run script: uv run {$file}`

### Run GUI: 

* via terminal

```bash
uv run src/ariel/gui_code/litegraph/main.py
```

### Run EA Example

```bash
uv run src/ariel/ec/a004.py
```

### Run MuJoCo Example(s)a

Any from the `examples/` folder

For example (pun intended):

```bash
uv run examples/_hi_prob_dec.py
```

## Neat commands!

### Grab `requirements.tex` automatically
```bash
uv add tool pipreqs
pipreqs path/to/parse --mode no-pin --force
uv add -r requirements.txt
``` -->


<!-- # Ariel

[![PyPI](https://img.shields.io/pypi/v/ariel.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/ariel.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/ariel)][pypi status]
[![License](https://img.shields.io/pypi/l/ariel)][license]

[![Read the documentation at https://ariel.readthedocs.io/](https://img.shields.io/readthedocs/ariel/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/Jacopo-DM/ariel/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/Jacopo-DM/ariel/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Ruff codestyle][ruff badge]][ruff project]

[pypi status]: https://pypi.org/project/ariel/
[read the docs]: https://ariel.readthedocs.io/
[tests]: https://github.com/Jacopo-DM/ariel/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/Jacopo-DM/ariel
[pre-commit]: https://github.com/pre-commit/pre-commit
[ruff badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff project]: https://github.com/charliermarsh/ruff

## Features

- TODO

## Requirements

- TODO

## Installation

You can install _Ariel_ via [pip] from [PyPI]. The package is distributed as a pure Python package, but also with pre-compiled wheels for major platforms, which include performance optimizations.

```console
$ pip install ariel
```

The pre-compiled wheels are built using `mypyc` and will be used automatically if your platform is supported. You can check the files on PyPI to see the list of available wheels.

## Usage

Please see the [Command-line Reference] for details.

## Development

To contribute to this project, please see the [Contributor Guide].

### Mypyc Compilation

This project can be compiled with `mypyc` to produce a high-performance version of the package. The compilation is optional and is controlled by an environment variable.

To build and install the compiled version locally, you can use the `tests_compiled` nox session:

```console
$ nox -s tests_compiled
```

This will set the `ARIEL_COMPILE_MYPYC=1` environment variable, which triggers the compilation logic in `setup.py`. The compiled package will be installed in editable mode in a new virtual environment.

You can also build the compiled wheels for distribution using the `cibuildwheel` workflow, which is configured to run on releases. If you want to build the wheels locally, you can use `cibuildwheel` directly:

```console
$ pip install cibuildwheel
$ export ARIEL_COMPILE_MYPYC=1
$ cibuildwheel --output-dir wheelhouse
```

This will create the compiled wheels in the `wheelhouse` directory.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [GPL 3.0 license][license],
_Ariel_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [uv hypermodern python cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[uv hypermodern python cookiecutter]: https://github.com/bosd/cookiecutter-uv-hypermodern-python
[file an issue]: https://github.com/Jacopo-DM/ariel/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->
<!-- 
[license]: https://github.com/Jacopo-DM/ariel/blob/main/LICENSE
[contributor guide]: https://github.com/Jacopo-DM/ariel/blob/main/CONTRIBUTING.md
[command-line reference]: https://ariel.readthedocs.io/en/latest/usage.html -
->
