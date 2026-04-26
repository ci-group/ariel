---
type: api_reference
tags: [mujoco, ui, visualization, opengl, c-api]
source: https://mujoco.readthedocs.io/en/stable/programming/ui.html
date_ingested: 2026-04-13
---
# mjUI

The primary struct for MuJoCo's native UI framework. Holds sections, items, theme settings, focus tracking, and edit state for a single UI panel. Used alongside [[mjuiState]] and the [[mujoco_ui_functions]] API.

## Design Overview

- **Static allocation**: a single `mjUI` C struct with fixed capacity — no dynamic object creation/deletion.
- **Offscreen rendering**: UI elements render to auxiliary OpenGL buffers; GPU copies pixels to the window framebuffer.
- **Minimal state**: pointers reference user data directly instead of replicating it.
- **Theme-based appearance**: no per-element customization — all visual properties come from `mjuiThemeSpacing` and `mjuiThemeColor`.
- **Automated enable/disable**: items carry an integer category; the `mjfItemEnable` callback decides enabled/disabled state.

## Capacity Constants

```c
#define mjMAXUISECT     10    // max sections per UI
#define mjMAXUIITEM     200   // max items per section
#define mjMAXUITEXT     300   // max text length (edit buffer, other fields)
#define mjMAXUINAME     40    // max name/label length
#define mjMAXUIMULTI    35    // max choices in radio/select
#define mjMAXUIEDIT     7     // max fields in edit item
#define mjMAXUIRECT     25    // max rectangles tracked in mjuiState

#define mjSEPCLOSED     1000  // special state: separator starts closed
#define mjPRESERVE      2000  // special state: preserve existing state on add
```

## Key Codes

GLFW-compatible key codes for `mjuiState.key`:

```c
#define mjKEY_ESCAPE     256
#define mjKEY_ENTER      257
#define mjKEY_TAB        258
#define mjKEY_BACKSPACE  259
#define mjKEY_INSERT     260
#define mjKEY_DELETE     261
#define mjKEY_RIGHT      262
#define mjKEY_LEFT       263
#define mjKEY_DOWN       264
#define mjKEY_UP         265
#define mjKEY_PAGE_UP    266
#define mjKEY_PAGE_DOWN  267
#define mjKEY_HOME       268
#define mjKEY_END        269
#define mjKEY_F1–F12     290–301
#define mjKEY_NUMPAD_0   320
#define mjKEY_NUMPAD_9   329
```

## Enumerations

### `mjtItem` — UI element types

```c
typedef enum mjtItem_ {
  mjITEM_END        = -2,  // sentinel: end of mjuiDef table
  mjITEM_SECTION    = -1,  // sentinel: section header in mjuiDef table
  mjITEM_SEPARATOR  =  0,  // horizontal separator line
  mjITEM_STATIC,           // non-interactive text label
  mjITEM_BUTTON,           // clickable button
  mjITEM_CHECKINT,         // checkbox backed by int (0/1)
  mjITEM_CHECKBYTE,        // checkbox backed by byte (0/1)
  mjITEM_RADIO,            // radio group (vertical layout)
  mjITEM_RADIOLINE,        // radio group (horizontal layout)
  mjITEM_SELECT,           // dropdown selection list
  mjITEM_SLIDERINT,        // integer slider
  mjITEM_SLIDERNUM,        // double slider
  mjITEM_EDITINT,          // integer text edit field(s)
  mjITEM_EDITNUM,          // double text edit field(s)
  mjITEM_EDITFLOAT,        // float text edit field(s)
  mjITEM_EDITTXT,          // string text edit field
  mjNITEM                  // count
} mjtItem;
```

### `mjtSection` — section expand state

```c
typedef enum mjtSection_ {
  mjSECT_CLOSED = 0,  // section is collapsed
  mjSECT_OPEN,        // section is expanded
  mjSECT_FIXED        // section is always visible, cannot be toggled
} mjtSection;
```

## Struct: `mjuiThemeSpacing`

Layout dimensions for a UI theme (pixel units):

```c
struct mjuiThemeSpacing_ {
  int total;        // total width
  int scroll;       // scrollbar width
  int label;        // label width
  int section;      // section indentation
  int cornersect;   // section corner radius
  int cornersep;    // separator corner radius
  int itemside;     // item side padding
  int itemmid;      // item middle gap
  int itemver;      // item vertical spacing
  int texthor;      // text horizontal padding
  int textver;      // text vertical padding
  int linescroll;   // line scroll amount
  int samples;      // antialiasing samples
};
```

## Struct: `mjuiThemeColor`

RGB color values (each field is `float[3]`):

```c
struct mjuiThemeColor_ {
  float master[3];              // master/background color
  float thumb[3];               // scrollbar thumb
  float secttitle[3];           // section title (unchecked, gradient start)
  float secttitle2[3];          // section title (unchecked, gradient end)
  float secttitleuncheck[3];    // section title unchecked state
  float secttitleuncheck2[3];
  float secttitlecheck[3];      // section title checked state
  float secttitlecheck2[3];
  float sectfont[3];            // section title font
  float sectsymbol[3];          // expand/collapse arrow
  float sectpane[3];            // section content background
  float separator[3];           // separator line (gradient start)
  float separator2[3];          // separator line (gradient end)
  float shortcut[3];            // keyboard shortcut hint text
  float fontactive[3];          // active/enabled item text
  float fontinactive[3];        // disabled item text
  float decorinactive[3];       // disabled decoration
  float decorinactive2[3];
  float button[3];              // button fill
  float check[3];               // checkbox fill
  float radio[3];               // radio button fill
  float select[3];              // select dropdown fill
  float select2[3];
  float slider[3];              // slider track
  float slider2[3];             // slider fill
  float edit[3];                // edit field background
  float edit2[3];               // edit field border
  float cursor[3];              // text cursor
};
```

Two built-in themes are available via `mjui_themeColor(0)` and `mjui_themeColor(1)` (dark), and `mjui_themeSpacing(0)` / `mjui_themeSpacing(1)`.

## Struct: `mjuiItemSingle`

For `mjITEM_BUTTON` and `mjITEM_SEPARATOR`:

```c
struct mjuiItemSingle_ {
  int modifier;   // required modifier key (0 = none)
  int shortcut;   // keyboard shortcut key code
};
```

## Struct: `mjuiItemMulti`

For `mjITEM_RADIO`, `mjITEM_RADIOLINE`, `mjITEM_SELECT`:

```c
struct mjuiItemMulti_ {
  int nelem;                              // number of choices
  char name[mjMAXUIMULTI][mjMAXUINAME];  // choice labels
};
```

## Struct: `mjuiItemSlider`

For `mjITEM_SLIDERINT`, `mjITEM_SLIDERNUM`:

```c
struct mjuiItemSlider_ {
  double range[2];     // [min, max]
  double divisions;    // number of discrete steps
};
```

## Struct: `mjuiItemEdit`

For `mjITEM_EDITINT`, `mjITEM_EDITNUM`, `mjITEM_EDITFLOAT`:

```c
struct mjuiItemEdit_ {
  int nelem;                       // number of edit fields (1–mjMAXUIEDIT)
  double range[mjMAXUIEDIT][2];    // [min, max] per field
};
```

## Struct: `mjuiItem`

A single UI element:

```c
struct mjuiItem_ {
  int type;              // mjtItem value
  char name[mjMAXUINAME]; // display label
  int state;             // 0=disabled, 1=enabled, 2=hide; or mjPRESERVE
  void *pdata;           // pointer to user data (read/written by UI)
  int sectionid;         // which section this item belongs to
  int itemid;            // index within section
  int userid;            // application-assigned identifier
  union {
    struct mjuiItemSingle_ single;   // for BUTTON, SEPARATOR
    struct mjuiItemMulti_  multi;    // for RADIO, SELECT
    struct mjuiItemSlider_ slider;   // for SLIDERS
    struct mjuiItemEdit_   edit;     // for EDIT fields
  };
  mjrRect rect;          // bounding rectangle (set by mjui_resize)
  int skip;              // internal layout flag
};
```

## Struct: `mjuiSection`

A collapsible section container:

```c
struct mjuiSection_ {
  char name[mjMAXUINAME]; // section header label
  int state;              // mjtSection value
  int modifier;           // modifier key for section toggle shortcut
  int shortcut;           // key code for section toggle shortcut
  int checkbox;           // 1 if section has a checkbox, 0 otherwise
  int nitem;              // number of items currently in section
  mjuiItem item[mjMAXUIITEM]; // item array
  mjrRect rtitle;         // bounding rect of title bar
  mjrRect rcontent;       // bounding rect of content area
  int lastclick;          // timestamp of last title click (double-click detection)
};
```

## Struct: `mjUI`

The top-level UI struct:

```c
struct mjUI_ {
  mjuiThemeSpacing spacing;   // layout theme
  mjuiThemeColor color;       // color theme
  mjfItemEnable predicate;    // callback: int predicate(int category, void* data)
  void* userdata;             // passed to predicate callback
  int rectid;                 // index in mjuiState.rect[] for this UI panel
  int auxid;                  // OpenGL auxiliary buffer id
  int radiocol;               // number of columns in radio/select layout
  int width;                  // panel width in pixels (set by theme)
  int height;                 // total panel height (set by mjui_resize)
  int maxheight;              // visible height limit (0 = unlimited)
  int scroll;                 // current scroll offset in pixels
  int mousesect;              // section under mouse (-1 = none)
  int mouseitem;              // item under mouse (-1 = none)
  int mousehelp;              // item showing tooltip (-1 = none)
  int mouseclicks;            // click count for double-click detection
  int mousesectcheck;         // section checkbox under mouse
  int editsect;               // section with active text edit (-1 = none)
  int edititem;               // item with active text edit (-1 = none)
  int editcursor;             // cursor position in edit text
  int editscroll;             // horizontal scroll in edit text
  char edittext[mjMAXUITEXT]; // current edit buffer contents
  mjuiItem* editchanged;      // pointer to item that just changed (after mjui_event)
  int nsect;                  // number of sections currently in UI
  mjuiSection sect[mjMAXUISECT]; // section array
};
```

## Struct: `mjuiDef`

Definition table entry — used to build UIs declaratively with `mjui_add()`:

```c
struct mjuiDef_ {
  int type;              // mjtItem value, or mjITEM_SECTION / mjITEM_END
  char name[mjMAXUINAME]; // item or section label
  int state;             // initial state (0/1/2 or mjPRESERVE/mjSEPCLOSED)
  void* pdata;           // pointer to backing data (NULL for buttons/separators)
  char other[mjMAXUITEXT]; // type-specific: choice labels ("|"-separated), range spec, etc.
  int otherint;          // additional integer parameter
};
```

### `other` field format by item type

| Type | `other` content |
|------|----------------|
| `mjITEM_RADIO` / `mjITEM_SELECT` | Choice names separated by `\|`, e.g. `"opt1\|opt2\|opt3"` |
| `mjITEM_SLIDERINT` / `mjITEM_SLIDERNUM` | Range as `"min max divisions"` |
| `mjITEM_EDITINT` / `mjITEM_EDITNUM` | Range(s) as `"min1 max1 min2 max2 ..."` |
| `mjITEM_BUTTON` | Shortcut key name (optional) |
| Others | Typically empty |

## Callback Type

```c
typedef int (*mjfItemEnable)(int category, void* data);
```

Called by `mjui_update` to determine whether items are enabled. Return 1 = enabled, 0 = disabled. `category` is the item's `state` field value when used as a category integer; `data` is `mjUI.userdata`.

## Notes

- The `mjUI` struct is large due to static allocation (`mjMAXUIITEM` items × `mjMAXUISECT` sections). Typically heap-allocated.
- `rectid` must be set to the index in `mjuiState.rect[]` where this UI's bounding rectangle is stored, before calling any rendering/event functions.
- `auxid` is assigned by the rendering context; obtain it via `mjr_addAux()` before calling `mjui_resize`.
- After modifying sections/items at runtime, call `mjui_update(-1, -1, ...)` to recompute layout.
- See [[mujoco_ui_functions]] for the full API, and [[mjuiState]] for the input event struct.
