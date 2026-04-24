---
type: api_reference
tags: [mujoco, simulation, visualization, rendering, opengl]
source: https://mujoco.readthedocs.io/en/stable/programming/samples.html
date_ingested: 2026-04-13
---
# mujoco_code_samples

MuJoCo ships five C++ code samples (`testspeed`, `simulate`, `compile`, `basic`, `record`) that illustrate core programming patterns: simulation loops, interactive visualization, model I/O, and offscreen rendering.

---

## testspeed

Times passive-dynamics simulation of a model over many steps across one or more threads.

### Command-line usage

```
testspeed modelfile [nstep nthread ctrlnoise npoolthread]
```

### Parameters

| Argument | Default | Meaning |
|---|---|---|
| `modelfile` | (required) | Path to model file |
| `nstep` | 10000 | Number of steps per rollout |
| `nthread` | 1 | Number of threads running parallel rollouts |
| `ctrlnoise` | 0.01 | Scale of pseudo-random noise injected into actuators |
| `npoolthread` | 1 | Number of threads in engine-internal `mjThreadPool` |

### Notes

- When `nthread > 1`: allocates one `mjModel` + per-thread `mjData`, runs identical simulations in parallel. Optimal `nthread` = number of logical cores. Models RL sample-collection scenarios.
- When `npoolthread > 1`: creates an `mjThreadPool` inside the engine to parallelize large-scene simulation. `nthread` and `npoolthread` serve mutually exclusive use-cases.
- If a keyframe named `"test"` exists in the model, it is used as the initial state instead of the reference configuration.
- `ctrlnoise > 0` prevents models settling to a static state where warmstarts inflate speed.
- Statistics printed: contact count, scalar constraint count, CPU time from internal profiling.
- For repeatable stats: use the "performance" CPU governor on Linux or "High Performance" power plan on Windows. Use `taskset` (Linux) or `start /affinity` (Windows) to pin to same core type (P-cores vs E-cores).

### Simulating controlled dynamics

Either install a [[mujoco_callbacks|`mjcb_control`]] callback or set `data.ctrl` values manually before calling `mj_step`.

---

## simulate

Fully-featured interactive simulator with OpenGL window, profiler, sensor plots, and drag-and-drop model loading.

### Features

- **Window**: opened via GLFW (platform-independent)
- **Help**: press `F1` for built-in command summary
- **Object selection**: left double-click
- **Force/torque application**: `Ctrl` + drag on selected object
- **Camera control**: drag to rotate, right-drag to translate vertically, shift+right-drag to translate horizontally, scroll/middle-drag to zoom
- **Keyboard shortcuts**: pause, reset, reload model file
- **Profiler**: reads diagnostic fields from `mjData`, uses [[MjrRect|`mjr_figure`]] for 2D plots with grids, annotation, axis scaling
- **Sensor plot**: bar graph of all sensor outputs defined in the model
- **Drag-and-drop**: load model at runtime without restarting

### Architecture

| Component | Description |
|---|---|
| `main()` | Initializes MuJoCo + GLFW, opens window, installs mouse/keyboard callbacks |
| Main loop | Handles UI events and rendering (no render callback — user-driven) |
| Background thread | Runs simulation; synchronized with main thread |
| Mouse/keyboard callbacks | Invoke MuJoCo abstract visualization functions directly |

### Key visualization functions used

- `mjv_moveCamera` — camera and perturbation control from mouse events
- `mjr_figure` — profiler and sensor data 2D plots
- [[MjvCamera]], [[MjvOption]], [[MjvScene]], [[MjrContext]]

### Windows timing note

`simulate.cc` sleeps to maintain real-time rate. On Windows, this can lower CPU frequency and distort profiler timing. Fix: set the minimum processor state to 100% in the Windows power plan.

---

## compile

Converts model files between formats: (MJCF, URDF, MJB) → (MJCF, MJB, TXT).

### Supported conversions

| Input | Output |
|---|---|
| MJCF | MJCF (canonical subset — output differs from input) |
| MJCF | MJB (binary compiled model) |
| MJCF | TXT (human-readable road-map, not loadable) |
| URDF | MJCF / MJB / TXT |
| MJB | MJCF / TXT |

### Notes

- TXT output is in one-to-one correspondence with compiled `mjModel`. Useful during model development.
- `mj_printData` (not in this sample) produces a TXT in one-to-one correspondence with `mjData`.
- MJCF→MJCF conversion uses the canonical subset of the format; output will generally differ from input.
- If input is MJCF and output path is empty, compilation is performed and timed twice to measure the compiler's asset cache impact.

---

## basic

Minimal interactive simulator. Renders at 60 fps, advances simulation in real-time. Single-file illustration of the [[mujoco_rendering_pipeline|Visualization programming guide]] concepts.

### Usage

```
basic modelfile
```

### Controls

| Input | Action |
|---|---|
| Left drag | Rotate camera |
| Right drag | Translate camera (vertical plane) |
| Shift + right drag | Translate camera (horizontal plane) |
| Scroll / middle drag | Zoom |
| Backspace | Reset simulation |

### Architecture

Uses GLFW for window + context. Advances simulation in real-time while rendering at 60 fps. No threading, no UI controls — the minimal scaffold for custom simulators.

---

## record

Simulates passive dynamics, renders offscreen, and writes raw RGB (+ optional depth) pixel data to a file.

### Usage

```
record modelfile duration fps rgbfile [adddepth]
```

### Parameters

| Argument | Default | Meaning |
|---|---|---|
| `modelfile` | (required) | Path to model file |
| `duration` | (required) | Recording duration in seconds |
| `fps` | (required) | Frames per second to render |
| `rgbfile` | (required) | Path to output raw file |
| `adddepth` | 1 | Overlay depth image in lower-left corner (0 = none) |

### Example

```bash
# Record 5 seconds at 60 fps
record humanoid.xml 5 60 rgb.out

# Convert raw output to MP4 (model uses 2560x1440 offscreen resolution)
ffmpeg -f rawvideo -pixel_format rgb24 -video_size 2560x1440 \
       -framerate 60 -i rgb.out -vf "vflip,format=yuv420p" video.mp4
```

The `video_size` passed to `ffmpeg` must exactly match `visual/global/offwidth` × `visual/global/offheight` from the model XML.

### Offscreen resolution

Controlled by model XML attributes:

| Attribute | Description |
|---|---|
| `visual/global/offwidth` | Offscreen buffer width in pixels |
| `visual/global/offheight` | Offscreen buffer height in pixels |
| `visual/quality/offsamples` | Number of multi-samples for anti-aliasing |

### OpenGL context options

`record.cc` can be compiled with three OpenGL context backends. Select at compile time:

| Symbol | Backend | Platform |
|---|---|---|
| (default) | GLFW with invisible window | All platforms |
| `MJ_OSMESA` | OSMesa (software) | Linux only |
| `MJ_EGL` | EGL (GPU-accelerated) | Linux only |

The MuJoCo rendering code is identical regardless of which backend is used — only `initOpenGL` / `closeOpenGL` differ. See [[mujoco_rendering_pipeline]] for backend setup details.

---

## See Also

- [[mujoco_rendering_pipeline]] — OpenGL backend setup (EGL, OSMesa, GLFW), render loop
- [[MjvCamera]] — camera control
- [[MjvOption]] — visualization options
- [[MjvScene]] — scene graph
- [[MjrContext]] — rendering context
- [[mujoco_callbacks]] — `mjcb_control` for injecting control in `testspeed`
- [[mujoco_simulation_functions]] — `mj_step`, `mj_forward`
