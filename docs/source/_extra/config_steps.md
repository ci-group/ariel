# Configuration Steps

All steps taken when configuring the project.

## Setup Steps

Some text with `code`, ``code``, **bold**, *italic*, and [a link](https://www.sphinx-doc.org/en/master/).

[some other link](https://356d3c33-0ced-454f-ab27-fc4a14842ad7.mdnplay.dev/en-US/docs/Web/CSS/:visited/runner.html?uuid=356d3c33-0ced-454f-ab27-fc4a14842ad7&state=bVDBTsMwDP0VKxyQppWKaxnblTsnpFzcxl3DMqdK3JWq6r%2BTVGwgjVxivef3%2FOxZdXJ2qlI7hC5Q%2B6rVg1CU4mKjFTKFs3zSav%2BGF4LJD%2FCDg3Q2QiZhIjnsStzv6gDlXvOvU9J9%2BOExKdEFQjPdq5%2ByUrPaqibGFANh1gxQbuC9p8a2E7DnQgJy7DEQCxhqcXASQTw0FAQtQx98n0pLcZvV6aFzfrR8TJPonFtrgiiTS7NHK12GobqmiYJCsCmztsbmdAx%2BYFM03vlQwdilppeV88FQQp77L4jeWXPjFs2a8WY4%2F%2B80UU71x%2BpKdF76dIyVuYOW9Tyf%2BTrpz%2FtQKg2Gk1q%2BAQ%3D%3D#test-visited-link)

[some other link](https:impossible.com)

```{code-block} python
:linenos:
:caption: text for caption
:emphasize-lines: 2

print("Hello, ...")
print("World!")
print("(with highlights!!)")
```

- [ ]


### Init

```bash
uv init . --python=3.12 --lib
git add .
git commit -m "Add: project init"

# <<< UPDATE GITIGNORE WITH TEMPLATE >>>
git add .gitignore
git commit -m "Update: gitignore from template"

uv add mypy ruff rich 
touch mypy.ini pyrightconfig.json ruff.toml
git add mypy.ini pyrightconfig.json ruff.toml
git commit -m "Add: configuration files for linters"

# <<< UPDATE CONFIG FILES WITH TEMPLATES >>>
git add mypy.ini pyrightconfig.json ruff.toml
git commit -m "Update: config files from template"
```

### Docs (Files)

```bash
mkdir -p docs/source docs/_build 
uv add --group=docs shibuya myst-parser sphinx_design sphinx-autodoc2
uv pip compile --group=docs -o docs/requirements.txt
touch docs/conf.py
# <<< FILL `conf.py` WITH CONTENT >>>
git add docs/conf.py docs/requirements.txt
git commit -m "Add: configuration files for docs"

touch docs/index.md
# <<< FILL `index.md` WITH CONTENT >>>
git add docs/index.md
git commit -m "Add: index page (landing page)"

mkdir -p docs/_templates/partials/
touch docs/_templates/partials/webfonts.html
# <<< FILL `webfonts.html` WITH CUSTOMIZATIONS >>>
git add docs/_templates/partials/webfonts.html
git commit -m "Add: custom fonts and colors"
```

* fill: `docs/conf.py`
  * https://shibuya.lepture.com/install/
  * https://www.sphinx-doc.org/en/master/tutorial/getting-started.html
  * https://myst-parser.readthedocs.io/en/stable/intro.html


#### Change Base Font

* Following: https://shibuya.lepture.com/customisation/fonts/

```bash
mkdir -p docs/_templates/partials/
touch  docs/_templates/partials/webfonts.html
```

### Nox

```bash
uv add --dev nox
touch noxfile.py
# <<< FILL `noxfile.py` WITH CONTENT >>>
nox --session=docs # test
git add noxfile.py
git commit -m "Add: nox configuration file (noxfile.py)"
```

* fill: `noxfile.py`
  * https://nox.thea.codes/en/stable/


### VSCode Configurations

#### Tasks

```bash
mkdir .vscode
touch .vscode/tasks.json
# <<< UPDATE TASKS FILES WITH TEMPLATE >>>
git add .vscode/tasks.json
git commit -m "Add: common terminal tasks"
```


### GitHub Workflows

#### Docs

```bash
mkdir -p .github/workflows/
touch .github/workflows/sphinx.yml
# <<< FILL `sphinx.yml` FROM TEMPLATES >>>
```

* create: `.github/workflows/sphinx.yml`
  * https://coderefinery.github.io/documentation/gh_workflow/
  * https://www.sphinx-doc.org/en/master/tutorial/deploying.html
