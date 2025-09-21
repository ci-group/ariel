# ARIEL Notes

## VS Code

By default, ARIEL ships with a `.vscode/` folder, which contains various
configuration files which I (Jacopo) think could be useful for most users; and
in general smoothen the experience of using VS Code with ARIEL.

* `.vscode/tasks.json`: contains various useful terminal tasks, such as running
tests, building docs, etc.
* `.vscode/settings.json`: contains various default settings, such as python
interpreter path, linting settings, etc.

If you do not use VS Code, you can ignore this folder. If you use VS Code, but
do not want to use these configurations, you can delete this folder.

## Installation: More Info on `uv`

In this project we use `uv` to interact with python for most tasks. When
installing ARIEL via `uv`, you will use commands which are different than the
usual `pip` ones:

Such as:
```bash
uv venv # create a virtual environment
uv sync # install all packages defined in the project into the venv
```

To install an new package, you can run the usual command (with a small change):
```bash
uv pip install NAME_OF_PACKAGE # install package `NAME_OF_PACKAGE`
```

If you want to add the package as a dependency you may run:
```bash
uv add # add a new dependency
```

In general `uv add` should only really be used by people who want to contribute
to the ARIEL code base.

On that note, `uv` stores dependencies in two broad categories: `default` and
`dev`. This functionality separates what developers of a library need to
install versus what users may need to install.

To install to the `dev` dedicated group, you should run:
```bash
uv add --dev NAME_OF_PACKAGE
```

In ARIEL, we have **four** categories (called `dependency-groups` in `uv`):
* default: what every user should install
* dev: what developers of ARIEL should install
* docs: installs required to build and view the documentation
* physical_robot: installs required **only** for the robots

To install to the other dedicated group (not `dev` or `default`), you should run:
```bash
uv add --group=docs NAME_OF_PACKAGE
```

This is reflected in the `project.toml`:
```toml
[project]
# ... some unrelated lines ...
dependencies = ["DEPENDENCY_0"] # for all groups

# ... some unrelated lines ...

[dependency-groups]
dev = [
    "DEPENDENCY_1",
    "DEPENDENCY_2",
    "DEPENDENCY_3", # in both dev and docs
]
docs = [
    "DEPENDENCY_4",
    "DEPENDENCY_5",
    "DEPENDENCY_3", # in both dev and docs
]
```

## Docs About Docs

### Python Docstring Headers

In most Python files written by a human, you will find a block of docstring
similar to the following:

```python
"""Nox sessions.

Author:     jmdm
Date:       2025-09-13
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro
Status:     Completed ✅
"""
```

This block of code is created by customizing a snippet I (Jacopo) use on all my
python files:

```python
"""TODO(jmdm): description of script.

Author:     jmdm
Date:       yyyy-mm-dd
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro

Status:     In progress ⚙️
Status:     Paused ⏸️
Status:     Completed ✅
Status:     Incomplete ❌
Status:     Broken ⚠️
Status:     To Improve ⬆️

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
```

This is a [`numpydoc` style] header, with some extra info; such as
the `Status:` of the file, and the original author(s). 

The purpose of this header is two fold, to indicate if the file you are looking
at is considered complete, to see how long ago it was created, and if you have
any questions regarding it, you know who to ask.

Of course, the same (and better) information can be obtained via `git` (for the
history) and running tests on the code (for the completeness). However, knowing
that this code must be as accessible to students, we assume they may not know
how to use those tools, or where to look. Also, using this snippet as a habit,
encourages coders to always write a docstring (which is good practice).
Moreover, we also include which system the code was written in (which maybe
harder to discover).

I do not include this header in `__init__.py` files.

[`numpydoc` style]: https://numpydoc.readthedocs.io/en/latest/format.html