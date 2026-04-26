---
type: source_summary
tags: [source, mujoco, visualization, rendering]
source: https://mujoco.readthedocs.io/en/stable/programming/visualization.html
author: Google DeepMind
date_ingested: 2026-04-13
---
# Source - MuJoCo Visualization API

MuJoCo's official visualization programming page and C header files (`mujoco.h`, `mjvisualize.h`) covering the full `mjv_*` abstract visualizer and `mjr_*` OpenGL renderer APIs, including all structs, enums, and function signatures. Content retrieved from GitHub source headers due to 403 on readthedocs.

## Entity Pages Created

- [[MjvPerturb]] — New: struct for mouse-driven body selection and perturbation; `mjPERT_TRANSLATE/ROTATE` bitmask; full field table and usage pattern
- [[MjvGeom]] — New: abstract scene geometry element; custom geom injection via `mjv_initGeom` / `mjv_connector`; debug arrow/line examples
- [[MjvFigure]] — New: 2D figure/plot overlay; `mjr_figure`; up to 100 line series; full attribute table
- [[mujoco_visualization_functions]] — New: complete reference for all `mjv_*` and `mjr_*` function signatures

## Entity Pages Updated

- [[MjvCamera]] — Added `orthographic` attribute
- [[MjvOption]] — Added `flexgroup`, `skingroup`, `bvh_depth`, `flex_layer`; expanded `mjtVisFlag` list to 31 flags
- [[MjvScene]] — Added `geomorder`, `camera[2]`, stereo/transform fields, `framewidth`, `framergb`; full attribute table
- [[mujoco_enumerations]] — Added `mjtPertBit`, `mjtLabel` (17 values), `mjtFrame` (8 values); expanded `mjtRndFlag` to 11 flags with descriptions; updated `mjtVisFlag` to 31 values
