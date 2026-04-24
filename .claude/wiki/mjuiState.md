---
type: api_reference
tags: [mujoco, ui, input, events, c-api]
source: https://mujoco.readthedocs.io/en/stable/programming/ui.html
date_ingested: 2026-04-13
---
# mjuiState

Global input state struct for MuJoCo's UI framework. Tracks mouse position, button state, keyboard modifiers, key presses, file drops, and the bounding rectangles of all UI panels. Passed to all [[mujoco_ui_functions]] calls.

## Enumerations

### `mjtButton` — mouse buttons

```c
typedef enum mjtButton_ {
  mjBUTTON_NONE   = 0,
  mjBUTTON_LEFT,
  mjBUTTON_RIGHT,
  mjBUTTON_MIDDLE
} mjtButton;
```

### `mjtEvent` — input event types

```c
typedef enum mjtEvent_ {
  mjEVENT_NONE      = 0,
  mjEVENT_MOVE,        // mouse moved
  mjEVENT_PRESS,       // mouse button pressed
  mjEVENT_RELEASE,     // mouse button released
  mjEVENT_SCROLL,      // mouse wheel scrolled
  mjEVENT_KEY,         // keyboard key pressed
  mjEVENT_RESIZE,      // window resized
  mjEVENT_REDRAW,      // window needs redraw (e.g. expose)
  mjEVENT_FILESDROP    // files dropped onto window
} mjtEvent;
```

## Struct: `mjuiState`

```c
struct mjuiState_ {
  // Rectangle registry
  int nrect;                    // number of registered UI rectangles
  mjrRect rect[mjMAXUIRECT];    // bounding rectangles (index 0 = window; 1+ = UI panels)

  // Application data
  void* userdata;               // user-supplied pointer passed through event handlers

  // Current event
  int type;                     // mjtEvent value for the current event

  // Mouse button state
  int left;                     // 1 if left button currently held
  int right;                    // 1 if right button currently held
  int middle;                   // 1 if middle button currently held
  int doubleclick;              // 1 if current press is a double-click
  int button;                   // mjtButton value for the last pressed button
  double buttontime;            // timestamp of last button press (seconds)

  // Mouse position
  double x;                     // cursor x position (pixels from left)
  double y;                     // cursor y position (pixels from top)
  double dx;                    // x displacement since last event
  double dy;                    // y displacement since last event
  double sx;                    // horizontal scroll amount
  double sy;                    // vertical scroll amount

  // Keyboard modifiers
  int control;                  // 1 if Ctrl is held
  int shift;                    // 1 if Shift is held
  int alt;                      // 1 if Alt is held

  // Key event
  int key;                      // key code (GLFW or mjKEY_* constant)
  double keytime;               // timestamp of last key press (seconds)

  // Hit-testing
  int mouserect;                // index of rect containing mouse (-1 = none)
  int dragrect;                 // index of rect where drag started (-1 = none)
  int dragbutton;               // mjtButton that initiated current drag

  // File drop
  int dropcount;                // number of dropped files
  const char** droppaths;       // array of dropped file paths (valid during mjEVENT_FILESDROP)
};
typedef struct mjuiState_ mjuiState;
```

## Parameters

| Field | Type | Description |
|-------|------|-------------|
| `nrect` | `int` | Number of active UI rectangles. Rect 0 is the full window. |
| `rect` | `mjrRect[25]` | Bounding boxes. Set by the platform layer; read by `mjui_event`. |
| `userdata` | `void*` | Passed to `mjfItemEnable` callback and available to event handlers. |
| `type` | `int` | `mjtEvent` value — set by platform before calling `mjui_event`. |
| `left/right/middle` | `int` | Current held state of mouse buttons (1 = held). |
| `doubleclick` | `int` | Set to 1 by platform layer on rapid successive press. |
| `button` | `int` | `mjtButton` value for most-recently pressed button. |
| `buttontime` | `double` | Time of last press; used for double-click detection. |
| `x`, `y` | `double` | Current cursor position in window pixels. |
| `dx`, `dy` | `double` | Motion delta since last event. |
| `sx`, `sy` | `double` | Scroll delta for the current event. |
| `control/shift/alt` | `int` | Modifier key hold state. |
| `key` | `int` | Key code for `mjEVENT_KEY`; 0 otherwise. |
| `keytime` | `double` | Timestamp of last key event. |
| `mouserect` | `int` | Which registered rectangle the mouse is currently over. |
| `dragrect` | `int` | Rectangle where the current drag started. |
| `dragbutton` | `int` | `mjtButton` that initiated the drag. |
| `dropcount` | `int` | Number of files in a drop event. |
| `droppaths` | `const char**` | File paths from a drop event (valid only during that event). |

## Usage Pattern

The platform layer (e.g. GLFW callbacks) fills `mjuiState` before calling into the UI framework:

```c
// In a GLFW mouse button callback:
uistate.type    = mjEVENT_PRESS;
uistate.button  = (button == GLFW_MOUSE_BUTTON_LEFT) ? mjBUTTON_LEFT : mjBUTTON_RIGHT;
uistate.left    = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
// ... then:
mjuiItem* changed = mjui_event(&ui, &uistate, &con);
```

## Notes

- `rect[0]` is conventionally the full window rectangle; `rect[1..n]` are UI panel rectangles. `mjUI.rectid` specifies which slot a given `mjUI` occupies.
- `mouserect` is updated by `mjui_event` to route events to the correct panel.
- The `droppaths` pointer is only valid during the `mjEVENT_FILESDROP` handler call; copy paths immediately if needed after.
- See [[mjUI]] for the UI panel struct and [[mujoco_ui_functions]] for the full API.
