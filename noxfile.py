"""Nox sessions.

References
----------
    * `nox <https://github.com/wntrblm/nox>`_
"""

# Standard library
import shutil
from pathlib import Path

# Third-party libraries
import nox

# --- NOX SETUP ---
nox.options.default_venv_backend = "uv"
package = "ariel"
python_versions = ["3.12", "3.13"]


@nox.session(python=python_versions[0])
def docs(session: nox.Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    # Remove existing build dir
    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # Remove existing apidocs
    apidocs_dir = Path("docs", "_autoapi")
    if apidocs_dir.exists():
        shutil.rmtree(apidocs_dir)

    session.run(
        "uv",
        "sync",
        "--group",
        "docs",
        external=True,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.run("sphinx-autobuild", *args)
