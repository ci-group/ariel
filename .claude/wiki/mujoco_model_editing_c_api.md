---
type: api_reference
tags: [mujoco, mjspec, model-building, c-api]
source: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html
date_ingested: 2026-04-13
---
# mujoco_model_editing_c_api

C API for programmatic MuJoCo model construction and editing via [[MjSpec]] (`mjSpec`) structs. The Python bindings (`mujoco.MjSpec`) wrap these functions.

## Signature

```c
// All functions declared in mujoco/mujoco.h; structs in mujoco/mjspec.h
```

## Lifecycle Functions

```c
// Create empty spec
mjSpec* mj_makeSpec(void);

// Free spec (frees all owned elements)
void mj_deleteSpec(mjSpec* s);

// Parse XML file → spec (vfs and error are nullable)
mjSpec* mj_parseXML(const char* filename, const mjVFS* vfs,
                    char* error, int error_sz);

// Parse XML string → spec
mjSpec* mj_parseXMLString(const char* xml, const mjVFS* vfs,
                          char* error, int error_sz);

// Compile spec → mjModel (vfs nullable)
mjModel* mj_compile(mjSpec* s, const mjVFS* vfs);

// Recompile spec in-place, preserving simulation state. Returns 0 on success.
int mj_recompile(mjSpec* s, const mjVFS* vfs, mjModel* m, mjData* d);

// Copy real-valued arrays from model back to spec. Returns 1 on success.
int mj_copyBack(mjSpec* s, const mjModel* m);

// Save spec to XML file. Returns 0 on success.
int mj_saveXML(const mjSpec* s, const char* filename,
               char* error, int error_sz);

// Save spec to XML string. Returns 0 on success, -1 on failure.
// If output buffer too small, returns required size.
int mj_saveXMLString(const mjSpec* s, char* xml, int xml_sz,
                     char* error, int error_sz);

// Error from last compile attempt
const char* mjs_getError(mjSpec* s);

// Enable/disable deep copy during attachment (1=deep, 0=shallow)
int mjs_setDeepCopy(mjSpec* s, int deepcopy);
```

## Tree Element Constructors (body-owned)

All take optional `def` (default class, nullable):

```c
// Add child body
mjsBody*  mjs_addBody(mjsBody* body, const mjsDefault* def);

// Add site to body
mjsSite*  mjs_addSite(mjsBody* body, const mjsDefault* def);

// Add joint to body
mjsJoint* mjs_addJoint(mjsBody* body, const mjsDefault* def);

// Add free joint (no default needed)
mjsJoint* mjs_addFreeJoint(mjsBody* body);

// Add geom to body
mjsGeom*  mjs_addGeom(mjsBody* body, const mjsDefault* def);

// Add camera to body
mjsCamera* mjs_addCamera(mjsBody* body, const mjsDefault* def);

// Add light to body
mjsLight* mjs_addLight(mjsBody* body, const mjsDefault* def);

// Add frame to body (parentframe nullable)
mjsFrame* mjs_addFrame(mjsBody* body, mjsFrame* parentframe);

// Remove element and all referencing elements. Returns 0 on success.
int mjs_delete(mjSpec* spec, mjsElement* element);
```

## Non-Tree Element Constructors (spec-level)

```c
mjsActuator* mjs_addActuator(mjSpec* s, const mjsDefault* def);
mjsSensor*   mjs_addSensor(mjSpec* s);
mjsFlex*     mjs_addFlex(mjSpec* s);
mjsPair*     mjs_addPair(mjSpec* s, const mjsDefault* def);
mjsExclude*  mjs_addExclude(mjSpec* s);
mjsEquality* mjs_addEquality(mjSpec* s, const mjsDefault* def);
mjsTendon*   mjs_addTendon(mjSpec* s, const mjsDefault* def);

// Tendon wrapping
mjsWrap* mjs_wrapSite(mjsTendon* tendon, const char* name);
mjsWrap* mjs_wrapGeom(mjsTendon* tendon, const char* name, const char* sidesite);
mjsWrap* mjs_wrapJoint(mjsTendon* tendon, const char* name, double coef);
mjsWrap* mjs_wrapPulley(mjsTendon* tendon, double divisor);
```

## Asset Constructors

```c
mjsMesh*     mjs_addMesh(mjSpec* s, const mjsDefault* def);
mjsHField*   mjs_addHField(mjSpec* s);
mjsSkin*     mjs_addSkin(mjSpec* s);
mjsTexture*  mjs_addTexture(mjSpec* s);
mjsMaterial* mjs_addMaterial(mjSpec* s, const mjsDefault* def);
```

## Metadata Constructors

```c
mjsNumeric* mjs_addNumeric(mjSpec* s);
mjsText*    mjs_addText(mjSpec* s);
mjsTuple*   mjs_addTuple(mjSpec* s);
mjsKey*     mjs_addKey(mjSpec* s);
mjsPlugin*  mjs_addPlugin(mjSpec* s);
mjsDefault* mjs_addDefault(mjSpec* s, const char* classname,
                            const mjsDefault* parent);  // parent nullable
```

## Actuator Shorthand Setters

```c
// Set actuator type to motor
const char* mjs_setToMotor(mjsActuator* actuator);

// Position servo (returns error string or NULL)
const char* mjs_setToPosition(mjsActuator* actuator, double kp,
                               double kv[1], double dampratio[1],
                               double timeconst[1], double inheritrange);

// Integrated velocity servo
const char* mjs_setToIntVelocity(mjsActuator* actuator, double kp,
                                  double kv[1], double dampratio[1],
                                  double timeconst[1], double inheritrange);

// Velocity servo
const char* mjs_setToVelocity(mjsActuator* actuator, double kv);

// Damper
const char* mjs_setToDamper(mjsActuator* actuator, double kv);

// Cylinder (hydraulic/pneumatic)
const char* mjs_setToCylinder(mjsActuator* actuator, double timeconst,
                               double bias, double area, double diameter);

// Muscle
const char* mjs_setToMuscle(mjsActuator* actuator, double timeconst[2],
                             double tausmooth, double range[2], double force,
                             double scale, double lmin, double lmax,
                             double vmax, double fpmax, double fvmax);

// Adhesion
const char* mjs_setToAdhesion(mjsActuator* actuator, double gain);
```

## Search / Navigation

```c
mjsBody*   mjs_findBody(mjSpec* s, const char* name);
mjsFrame*  mjs_findFrame(mjSpec* s, const char* name);
mjsElement* mjs_findElement(mjSpec* s, mjtObj type, const char* name);
mjsBody*   mjs_findChild(mjsBody* body, const char* name);
mjSpec*    mjs_findSpec(mjSpec* spec, const char* name);

// Parent of any element (returns NULL for worldbody)
mjsBody*   mjs_getParent(mjsElement* element);

// Frame attached to element
mjsFrame*  mjs_getFrame(mjsElement* element);

// Spec owning an element
mjSpec*    mjs_getSpec(mjsElement* element);

// Compiled id (valid only after compilation)
int        mjs_getId(mjsElement* element);

// Iterate elements
mjsElement* mjs_firstChild(mjsBody* body, mjtObj type, int recurse);
mjsElement* mjs_firstElement(mjSpec* s, mjtObj type);
mjsElement* mjs_nextElement(mjSpec* s, mjsElement* element);
```

## Default Class Functions

```c
// Get global default for a spec
mjsDefault* mjs_getSpecDefault(mjSpec* s);

// Get default associated with an element
mjsDefault* mjs_getDefault(mjsElement* element);

// Find named default class
mjsDefault* mjs_findDefault(mjSpec* s, const char* classname);

// Assign default class to element
void mjs_setDefault(mjsElement* element, const mjsDefault* def);
```

## Attribute Getters / Setters (C handles)

```c
// Name
int        mjs_setName(mjsElement* element, const char* name);
mjString*  mjs_getName(mjsElement* element);

// String I/O
void        mjs_setString(mjString* dest, const char* text);
const char* mjs_getString(const mjString* source);

// StringVec (append or set by index)
void    mjs_setStringVec(mjStringVec* dest, const char* text);
mjtByte mjs_setInStringVec(mjStringVec* dest, int i, const char* text);

// Numeric arrays
void mjs_setInt(mjIntVec* dest, const int* array, int size);
void mjs_setFloat(mjFloatVec* dest, const float* array, int size);
void mjs_setDouble(mjDoubleVec* dest, const double* array, int size);
const double* mjs_getDouble(const mjDoubleVec* source, int* size);

// Raw buffer
void mjs_setBuffer(mjByteVec* dest, const void* array, int size);

// Frame
int mjs_setFrame(mjsElement* dest, mjsFrame* frame);

// User values (arbitrary key/value metadata per element)
void        mjs_setUserValue(mjsElement* element, const char* key, const void* data);
const void* mjs_getUserValue(mjsElement* element, const char* key);
void        mjs_deleteUserValue(mjsElement* element, const char* key);
```

## Type-Cast Functions (mjsElement* → typed pointer)

```c
mjsBody*     mjs_asBody(mjsElement* element);
mjsGeom*     mjs_asGeom(mjsElement* element);
mjsJoint*    mjs_asJoint(mjsElement* element);
mjsSite*     mjs_asSite(mjsElement* element);
mjsCamera*   mjs_asCamera(mjsElement* element);
mjsLight*    mjs_asLight(mjsElement* element);
mjsFrame*    mjs_asFrame(mjsElement* element);
mjsActuator* mjs_asActuator(mjsElement* element);
mjsSensor*   mjs_asSensor(mjsElement* element);
mjsTendon*   mjs_asTendon(mjsElement* element);
mjsMesh*     mjs_asMesh(mjsElement* element);
```

## Initializer Defaults

Call these to reset a struct to global defaults (same values as `user_init.c`):

```c
void mjs_defaultSpec(mjSpec* spec);
void mjs_defaultBody(mjsBody* body);
void mjs_defaultFrame(mjsFrame* frame);
void mjs_defaultJoint(mjsJoint* joint);
void mjs_defaultGeom(mjsGeom* geom);
void mjs_defaultSite(mjsSite* site);
void mjs_defaultCamera(mjsCamera* camera);
void mjs_defaultLight(mjsLight* light);
void mjs_defaultFlex(mjsFlex* flex);
void mjs_defaultMesh(mjsMesh* mesh);
void mjs_defaultHField(mjsHField* hfield);
void mjs_defaultSkin(mjsSkin* skin);
void mjs_defaultTexture(mjsTexture* texture);
void mjs_defaultMaterial(mjsMaterial* material);
void mjs_defaultPair(mjsPair* pair);
void mjs_defaultEquality(mjsEquality* equality);
void mjs_defaultTendon(mjsTendon* tendon);
void mjs_defaultActuator(mjsActuator* actuator);
void mjs_defaultSensor(mjsSensor* sensor);
void mjs_defaultNumeric(mjsNumeric* numeric);
void mjs_defaultText(mjsText* text);
void mjs_defaultOrientation(mjsOrientation* orient);
```

## Attachment API

Move or copy a subtree from one spec into another:

```c
// Attach child element to parent element.
// prefix/suffix are prepended/appended to all child names.
// Returns attached element on success, NULL on failure.
mjsElement* mjs_attach(mjsElement* parent, const mjsElement* child,
                       const char* prefix, const char* suffix);
```

Supported parent/child combinations:
- frame → body or mjSpec
- site → body or mjSpec
- body → frame or mjSpec

Default behavior is **shallow copy** (move by reference). Enable deep copy with `mjs_setDeepCopy(spec, 1)`.

**Known limitations:**
- All child assets are copied regardless of whether they are referenced.
- Circular references are not checked (infinite loop risk).
- Keyframes from prior attachments are lost if a second attach occurs before compilation.

## Examples

```c
// Minimal: add box geom to world body
mjSpec* spec = mj_makeSpec();
mjsBody* world = mjs_findBody(spec, "world");
mjsGeom* geom = mjs_addGeom(world, NULL);
geom->type = mjGEOM_BOX;
geom->size[0] = geom->size[1] = geom->size[2] = 0.5;
mjModel* model = mj_compile(spec, NULL);
mj_deleteModel(model);
mj_deleteSpec(spec);

// Default class usage
mjsDefault* main = mjs_getSpecDefault(spec);
main->geom.type = mjGEOM_BOX;
mjsGeom* geom = mjs_addGeom(mjs_findBody(spec, "world"), main);

// Attachment: attach body from child spec to frame in parent spec
mjSpec* parent = mj_makeSpec();
mjSpec* child  = mj_makeSpec();
mjsElement* frame = mjs_addFrame(mjs_findBody(parent, "world"), NULL)->element;
mjsElement* body  = mjs_addBody(mjs_findBody(child,  "world"), NULL)->element;
mjsBody* attached = mjs_asBody(mjs_attach(frame, body, "robot-", "-0"));

// In-place recompile (during simulation)
mj_recompile(spec, NULL, model, data);  // returns 0 on success
```

## Notes

- **Memory**: elements are owned by the spec; only call `mj_deleteSpec`. Never free elements directly.
- **Default timing**: defaults apply only at element initialization via the constructor. Changing a default afterward does NOT retroactively update already-initialized elements.
- **Compiler flags per subtree**: when attaching, parent and child compiler flags coexist. Child is compiled with child flags; parent with parent flags.
- **After attachment**: a child attached by reference cannot be compiled independently.
- See [[MjSpec]] for the Python-facing API. See [[mjsBody]], [[mjsGeom]], [[mjsJoint]], [[mjsActuator]], [[mjsSensor]] for struct field references.
