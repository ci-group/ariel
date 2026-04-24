---
type: source_summary
tags: [source, mujoco, ui]
source: https://mujoco.readthedocs.io/en/stable/programming/ui.html
author: DeepMind / Google
date_ingested: 2026-04-13
---
# Source - MuJoCo UI Framework

Official MuJoCo documentation for the native UI framework (`mjui.h`). Covers the full C API for building interactive simulation viewers with mouse/keyboard input, themed panel layout, and OpenGL rendering. Relevant to ariel when building custom simulation viewers beyond the default `mujoco.viewer`.

## Entity Pages Created

- [[mjUI]] — Complete reference for `mjUI`, `mjuiSection`, `mjuiItem`, `mjuiDef`, `mjuiThemeSpacing`, `mjuiThemeColor` structs; all `mjtItem`, `mjtSection` enumerations; capacity constants and key codes
- [[mjuiState]] — Reference for `mjuiState` global input struct; `mjtButton` and `mjtEvent` enumerations; usage pattern for platform integration
- [[mujoco_ui_functions]] — All eight `mjui_*` API functions with exact C signatures, parameter tables, and initialization/render-loop examples
