"""Sphinx configuration."""

# --------- BASICS --------- #
html_theme = "furo"

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

# --------- FURO THEME OPTIONS --------- #
html_theme_options = {
    # -- Sidebar --
    "sidebar_hide_name": False,  # Show project name in sidebar
    
    # -- Top of page announcement banner --
    "announcement": "<b>🚀 New release coming soon!</b> Check out what's new in v0.1",

    # -- Light / Dark mode custom colors --
    "light_css_variables": {
        "color-brand-primary": "#4A90D9",
        "color-brand-content": "#4A90D9",
        "color-admonition-background": "rgba(74, 144, 217, 0.05)",
        "font-stack": "Inter, system-ui, sans-serif",
        "font-stack--monospace": "JetBrains Mono, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#79B8FF",
        "color-brand-content": "#79B8FF",
        "color-admonition-background": "rgba(121, 184, 255, 0.05)",
    },

    # -- Footer icons (e.g. GitHub link) --
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ci-group/ariel",
            "html": """
                <svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16" height="1em" width="1em">
                  <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                  0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                  -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                  .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                  -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27
                  .68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
                  .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
                  0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
            """,
            "class": "",
        },
    ],
}

# --------- HTML OUTPUT --------- #
html_title = "ARIEL"                        # Browser tab + sidebar title
html_last_updated_fmt = "%b %d, %Y"        # "Last updated: Jan 01, 2025" in footer
html_show_sourcelink = False               # Hide "View page source" link
html_copy_source = False                   # Don't copy .rst/.md source into build

# --------- STATIC FILES (optional custom CSS) --------- #
# html_static_path = ["_static"]
# html_css_files = ["custom.css"]