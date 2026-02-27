Contributing guide
==================
If you intend to contribute you may find this guide helpful. Contributions are highly appreciated.

----------------------------
Publishing your contribution
----------------------------
If you added something to Ariel that you would like to share with other people, you can do so by creating a `pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_ (PR) on `GitHub <https://github.com/ci-group/Ariel/pulls>`_.
The first time you make a contribution to a Ariel package your PR should add your name to the ``authors`` list in that package's ``pyproject.toml``.
**Only open PRs on the** ``development`` **-branch!** And make sure that only your commits are present, otherwise do a rebase.

Note that a general heuristic is, if your addition adds a dependency of another ARIEL package to the existing dependencies, you might not want to structure it that way.
For a guideline what can depend on what, look at the package diagram on the main page.

**Important Information before merging your PRs:**

- For merging into the ``development``-branch: Always use `Squash and Merge`.
- For merging into the ``master``-branch: Always use  `Rebase and Merge`.

----------------------
Developer installation
----------------------
The normal installation guide applies. You should definitely use editable mode.
**Add correct developer installation.**

----------------------
Continuous integration
----------------------
Github Actions is used for continuous integration(CI). You can find plenty of resources about the CI online. It is located in the Ariel directory at ``.github/workflows``.
You cannot directly run the CI configuration locally, but scripts are available to run the tools.

----------
Code tools
----------
WORK IN PROGRESS

-------------
Documentation
-------------
This documentation is automatically built by the CI and uploaded to github pages.
Code is analyzed and an API reference is generated programmatically.
You can compile the documentation using the Makefile available at ``./docs`` using ``make -C docs html``.

---------------
Version control
---------------
The codebase is managed through Git. We strictly enforce a `linear history <https://www.bitsnbites.eu/a-tidy-linear-git-history/>`_.
Releases are according to `Semantic Versioning <https://semver.org/>`_.
