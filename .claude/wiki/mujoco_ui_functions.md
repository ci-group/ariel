---
type: api_reference
tags: [mujoco, ui, c-api, opengl, visualization]
source: https://mujoco.readthedocs.io/en/stable/programming/ui.html
date_ingested: 2026-04-13
---
# mujoco_ui_functions

The eight C API functions that comprise MuJoCo's UI framework. They manage theme creation, panel construction, layout computation, event handling, and rendering. All operate on [[mjUI]] and [[mjuiState]] structs.

## Signatures

```c
// Get built-in UI theme spacing (ind: 0 or 1).
MJAPI mjuiThemeSpacing mjui_themeSpacing(int ind);

// Get built-in UI theme color (ind: 0–3).
MJAPI mjuiThemeColor mjui_themeColor(int ind);

// Add definitions from a mjuiDef table to the UI (appends to last section).
MJAPI void mjui_add(mjUI* ui, const mjuiDef* def);

// Add definitions from a mjuiDef table to a specific section by index.
MJAPI void mjui_addToSection(mjUI* ui, int sect, const mjuiDef* def);

// Compute UI layout sizes (call after adding items or resizing window).
MJAPI void mjui_resize(mjUI* ui, const mjrContext* con);

// Update UI rendering; section=-1, item=-1 means update everything.
MJAPI void mjui_update(int section, int item, const mjUI* ui,
                       const mjuiState* state, const mjrContext* con);

// Process one input event; returns pointer to changed item, or NULL.
MJAPI mjuiItem* mjui_event(mjUI* ui, mjuiState* state, const mjrContext* con);

// Blit UI offscreen buffer to the current OpenGL framebuffer.
MJAPI void mjui_render(mjUI* ui, const mjuiState* state, const mjrContext* con);
```

## Function Reference

### `mjui_themeSpacing`

```c
mjuiThemeSpacing mjui_themeSpacing(int ind);
```

Returns one of the two built-in layout themes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ind` | `int` | Theme index: `0` = compact, `1` = spacious |

**Returns:** `mjuiThemeSpacing` struct (by value).

---

### `mjui_themeColor`

```c
mjuiThemeColor mjui_themeColor(int ind);
```

Returns one of the four built-in color themes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ind` | `int` | Color theme index: `0`–`3` |

**Returns:** `mjuiThemeColor` struct (by value).

---

### `mjui_add`

```c
void mjui_add(mjUI* ui, const mjuiDef* def);
```

Parses a `NULL`-terminated `mjuiDef` table and appends sections and items to the UI. Table entries with `type == mjITEM_SECTION` start a new section; `type == mjITEM_END` terminates the table.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ui` | `mjUI*` | UI to populate |
| `def` | `const mjuiDef*` | Definition table, terminated by `{mjITEM_END}` |

**Example:**

```c
mjuiDef defSimulate[] = {
  {mjITEM_SECTION,  "Simulation",  1,  NULL,        "AS"},
  {mjITEM_RADIO,    "Run Mode",    2,  &run_mode,   "Pause\0Step\0Run"},
  {mjITEM_BUTTON,   "Reset",       2,  NULL,        ""},
  {mjITEM_SLIDERNUM,"Speed",       3,  &speed,      "0.1 10 100"},
  {mjITEM_END}
};
mjui_add(&ui, defSimulate);
```

---

### `mjui_addToSection`

```c
void mjui_addToSection(mjUI* ui, int sect, const mjuiDef* def);
```

Like `mjui_add`, but appends items to an existing section by index rather than creating a new one.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ui` | `mjUI*` | UI to modify |
| `sect` | `int` | Target section index (0-based, must be < `ui->nsect`) |
| `def` | `const mjuiDef*` | Definition table of items only (no `mjITEM_SECTION` entries) |

---

### `mjui_resize`

```c
void mjui_resize(mjUI* ui, const mjrContext* con);
```

Recomputes layout rectangles for all items and sections. Must be called:
- After `mjui_add` / `mjui_addToSection`
- After window resize
- Before first render

| Parameter | Type | Description |
|-----------|------|-------------|
| `ui` | `mjUI*` | UI to lay out |
| `con` | `const mjrContext*` | Rendering context (for font metrics) |

---

### `mjui_update`

```c
void mjui_update(int section, int item, const mjUI* ui,
                 const mjuiState* state, const mjrContext* con);
```

Redraws UI elements into the offscreen auxiliary buffer. Call when:
- User data pointed to by `pdata` changes
- Item enable/disable state changes
- Use `section=-1, item=-1` to redraw everything

| Parameter | Type | Description |
|-----------|------|-------------|
| `section` | `int` | Section to update; `-1` = all sections |
| `item` | `int` | Item within section to update; `-1` = all items in section |
| `ui` | `const mjUI*` | UI panel |
| `state` | `const mjuiState*` | Current input state (needed for hover highlighting) |
| `con` | `const mjrContext*` | Rendering context |

---

### `mjui_event`

```c
mjuiItem* mjui_event(mjUI* ui, mjuiState* state, const mjrContext* con);
```

Processes the event described in `state->type` and updates UI internal state (scroll position, active text edit, etc.). If the event caused an item's value to change, returns a pointer to that item (so the caller can react to it). Returns `NULL` if no item changed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ui` | `mjUI*` | UI panel (mutable — updates scroll, edit state, etc.) |
| `state` | `mjuiState*` | Input state (mutable — updates `mouserect`, `dragrect`) |
| `con` | `const mjrContext*` | Rendering context |

**Returns:** `mjuiItem*` pointing to the item that changed, or `NULL`.

**Typical dispatch loop:**

```c
mjuiItem* it = mjui_event(&ui, &uistate, &con);
if (it) {
    switch (it->sectionid) {
      case SECT_SIMULATION:
        handleSimulationChange(it);
        break;
      // ...
    }
    mjui_update(-1, -1, &ui, &uistate, &con);
}
```

---

### `mjui_render`

```c
void mjui_render(mjUI* ui, const mjuiState* state, const mjrContext* con);
```

Copies the UI's offscreen OpenGL buffer to the current window framebuffer. Call once per frame in the render loop, after `mjr_render`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ui` | `mjUI*` | UI panel to blit |
| `state` | `const mjuiState*` | Input state (used to position the panel within the window) |
| `con` | `const mjrContext*` | Rendering context |

## Typical Initialization Sequence

```c
// 1. Allocate and zero the UI
mjUI ui;
memset(&ui, 0, sizeof(ui));

// 2. Set theme and callback
ui.spacing = mjui_themeSpacing(0);
ui.color   = mjui_themeColor(0);
ui.predicate = myEnableCallback;
ui.userdata  = myAppData;

// 3. Assign a rectangle slot
ui.rectid = 1;  // rect[1] in mjuiState

// 4. Set auxiliary buffer id (from rendering context setup)
ui.auxid = 0;

// 5. Add sections and items
mjui_add(&ui, myDefTable);

// 6. Compute layout (after window + context are ready)
mjui_resize(&ui, &con);

// 7. Initial draw
mjui_update(-1, -1, &ui, &uistate, &con);
```

## Render Loop

```c
// Each frame:
mjr_render(viewport, &scene, &con);  // render 3D scene
mjui_render(&ui, &uistate, &con);    // overlay UI
```

## Notes

- The `mjrContext` must have been initialized with `mjr_makeContext` before any `mjui_*` calls.
- `mjui_update` only redraws to the offscreen buffer; `mjui_render` blits to screen. Both must be called to see changes.
- Calling `mjui_update(-1, -1, ...)` on every frame is safe but wasteful; prefer targeted updates when only specific items change.
- Platform integration (window resize, mouse/keyboard callbacks) is provided by `GlfwAdapter` in simulate.cc; replicate its pattern when building a custom viewer.
- See [[mjUI]] for all struct definitions, [[mjuiState]] for the input event struct, and [[mujoco_rendering_pipeline]] for context setup.
