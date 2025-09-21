"""Sphinx configuration."""

# --------- BASICS --------- #
html_theme = "shibuya"

project = "ARIEL"

copyright = "Computational Intelligence Group, Vrije Universiteit Amsterdam & Contributors"
author = "Computational Intelligence Group, Vrije Universiteit Amsterdam & Contributors"

extensions = [
    # "autodoc2",
    "autoapi.extension",
    "jupyter_sphinx",
    "myst_parser",
    "nbsphinx",
    "sphinx_click",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
]

add_module_names = False
autosectionlabel_prefix_document = True

# --------- SHIBUYA --------- #
templates_path = ["_templates"]
# html_title =
# html_logo = 
html_favicon = "resources/ariel_favicon.ico"

# --------- MYST --------- #
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]


# --------- AUTODOC2 --------- #
# autodoc2_packages = [
#     "../src/ariel",
# ]

# --------- NBSPHINX --------- #
# Myst notebook settings
nbsphinx_execute = "never"  # Do not run notebooks during build

# --------- AUTOAPI --------- #
autoapi_add_toctree_entry = True
autoapi_dirs = ["../src/ariel/"]
autoapi_template_dir = "_build/autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "special-members",
    "show-inheritance",
    "show-inheritance-diagram",
    "imported-members",
    "show-module-summary",
    "titles_only=True",
]
add_module_names = False
autoapi_keep_module_path = False
autoapi_template_dir = "_templates/autoapi"

# --------- AUTOSUMMARY --------- #
autosummary_generate = True
autosummary_generate_overwrite = True

# --------- AUTODOC --------- #
autodoc_default_options: dict[str, bool | str | list[str]] = {
    # "autodoc_preserve_defaults": True,
    # "autodoc_type_aliases": False,
    # "autodoc_typehints": "signature",
    # "autolink_concat_ids": "short",
    # "class-doc-from": "both",
    # "ignore-module-all": False,
    # "imported-members": False,
    # "inherited-members": True,
    # "member-order": "bysource",
    # "members": True,
    # "module-first": True,
    # "no-index-entry": True,
    # "no-index": True,
    # "no-value": True,
    # "private-members": True,
    # "show-inheritance": True,
    # "special-members": False,
    # "undoc-members": True,
    # "exclude-members": [",
}

# --------- NAPOLEON --------- #
napoleon_attr_annotations = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
napoleon_type_aliases: None = None
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# Auto section label settings
autosectionlabel_prefix_document = True
