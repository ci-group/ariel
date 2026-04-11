You are an autonomous wiki maintainer for the **ariel** EvoDevo robotics project. Ingest the provided source and write structured wiki pages to `.claude/wiki/`.

## Your task

The user has provided: **$ARGUMENTS**

---

### Step 1 — Fetch the source

- **URL** (`http://` or `https://`): use WebFetch
- **arXiv or PDF link**: use WebFetch; focus on abstract, method sections, algorithm listings, parameter tables
- **Local file** (`.py`, `.txt`, `.md`, or any path): use Read from the given path
- **Plain text / paste**: use it directly

---

### Step 2 — Identify source type and set extraction strategy

Classify the source as one of the following before proceeding:

| Source type | Extraction focus | Page types to create |
|---|---|---|
| **Library docs** (MuJoCo, Nevergrad, EvoTorch, DEAP, dm-control, PyTorch) | Every public class, function, config object. Exact signatures, parameter types, return types, code examples. | `api_reference` per entity + one `source_summary` |
| **Academic paper** (algorithm or theory) | Algorithm pseudocode, all equations, parameter table with defaults, key findings, practical recommendations, limitations | `algorithm_reference` per distinct algorithm + `concept_reference` per named scientific concept + one `source_summary` |
| **Ariel source code** (`src/ariel/**/*.py`) | Class hierarchy, public method signatures, constructor parameters, typical usage from call sites in the same file | `ariel_reference` per class/function + one `source_summary` |
| **Concept / blog / notes** | Core definition, theory, practical rules, relevance to the ariel project | `concept_reference` + one `source_summary` |

---

### Step 3 — Read the existing wiki

```
Glob .claude/wiki/*.md
```

Skim all existing pages. Note which topics are already covered and which `[[wikilinks]]` are already in use. **Do not create a new page for a topic that already has one** — instead update the existing page by appending new information under a new `## From: <source>` subsection.

Then read `.claude/SCHEMA.md`. Follow its page type definitions, naming conventions, section requirements, and frontmatter rules exactly.

---

### Step 4 — Extract content

Apply your chosen extraction strategy from Step 2. The rules below are non-negotiable regardless of source type:

- Preserve exact function/class signatures character-for-character
- Include every parameter with its type and meaning
- Extract all default values and valid ranges from the source
- Copy code examples verbatim — do not paraphrase or simplify them
- Preserve equation notation exactly; use code blocks if LaTeX is unavailable
- For ariel source: read the actual `.py` file rather than relying on memory

For **library docs**: create one page per class or logical function group. Do not merge unrelated classes into one page.

For **papers**: create one `algorithm_reference` page per distinct algorithm described. Create one `concept_reference` page per named scientific concept that is general enough to appear in other contexts (e.g. fitness shaping, morphological complexity, evolutionary pressure).

For **ariel source**: document the public interface. Note the file path under `## Location`. Include a real usage example from a call site in the codebase (search for one with Grep if needed).

---

### Step 5 — Write the files

Use the Write tool for new files, Edit for updates to existing pages. Every file must:

- Have YAML frontmatter matching SCHEMA.md
- Use the correct naming convention for its page type (SCHEMA.md §Naming)
- Use `[[wikilinks]]` for all cross-references to other wiki pages
- Have H1 matching the filename without `.md`

Also create `.claude/wiki/Source - <Title>.md` listing every page created or updated.

---

### Step 6 — Update the log

Append to `log.md` in the project root:

```
## [YYYY-MM-DD] Ingest | <source title>
- Files created: file1.md, file2.md, ...
- Files updated: file3.md, ...
- Model: Claude (subscription)
```

---

Do not explain steps — just execute them in order. Start by fetching the content now.
