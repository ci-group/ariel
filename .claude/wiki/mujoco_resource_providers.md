---
type: api_reference
tags: [mujoco, extension, asset-loading, c-api, resource]
source: https://mujoco.readthedocs.io/en/stable/programming/extension.html
date_ingested: 2026-04-13
---
# mujoco_resource_providers

Resource providers extend MuJoCo's asset loading beyond the OS filesystem and Virtual File System (VFS), enabling custom loading mechanisms such as HTTP downloads, in-memory stores, or data URIs.

## Overview

A resource provider is registered via `mjp_registerResourceProvider` with a `mjpResourceProvider` struct. Once registered, it is immediately available to all MuJoCo loading functions. When an asset filename has a prefix matching a registered provider's scheme, that provider handles loading.

## `mjpResourceProvider` Struct

```c
struct mjpResourceProvider {
  const char* prefix;              // URI scheme prefix (case-insensitive, e.g. "http", "data")
  mjfOpenResource     open;        // required: open/validate resource
  mjfReadResource     read;        // required: return resource bytes
  mjfCloseResource    close;       // required: free resource memory
  mjfGetResourceDir   getdir;      // optional: extract directory from resource name
  mjfResourceModified modified;    // optional: check if resource changed
  void* data;                      // opaque provider-specific data (constant within a model)
};
```

## `mjResource` Struct

```c
struct mjResource {
  const char* name;   // full resource name (e.g. "http://example.com/mesh.obj")
  void* data;         // provider-managed slot for per-resource data
  // ... additional internal fields
};
```

## Prefix / URI Matching

- Prefix must use valid URI scheme syntax (letters, digits, `+`, `-`, `.`; case-insensitive).
- A resource `{prefix}:{filename}` matches a provider with scheme `prefix`.
- Example: prefix `http` matches `http://www.example.com/myasset.obj` but **not** `https://...` (different scheme).
- The colon in the resource name is required for matching.

## Callback Signatures

### Required Callbacks

```c
// mjfOpenResource
// Validate resource existence; populate resource->data with per-resource info.
// Returns 1 (true) on success, 0 (false) on failure.
int open_callback(mjResource* resource);

// mjfReadResource
// Point *buffer at the resource bytes; return byte count.
// Returns -1 on failure.
int read_callback(mjResource* resource, const void** buffer);

// mjfCloseResource
// Free memory allocated in open_callback (stored in resource->data).
void close_callback(mjResource* resource);
```

### Optional Callbacks

```c
// mjfGetResourceDir
// Extract directory portion from resource name.
// e.g. "http://www.example.com/myasset.obj" -> "http://www.example.com/"
void getdir_callback(mjResource* resource, const char** dir, int* ndir);

// mjfResourceModified
// Return 1 if resource has changed since it was opened, else 0.
int modified_callback(const mjResource* resource);
```

## Registration

```c
// Returns positive number on success, 0 on failure
int mjp_registerResourceProvider(const mjpResourceProvider* provider);
```

## Complete Example: Data URI Provider

Implements a provider for the [data URI scheme](https://en.wikipedia.org/wiki/Data_URI_scheme), allowing base64-encoded assets inline in MJCF.

### Callbacks

```c
int str_open_callback(mjResource* resource) {
  if (!is_valid_data_uri(resource->name)) {
    return 0;  // failure
  }
  resource->data = mju_malloc(get_data_uri_size(resource->name));
  if (resource->data == NULL) {
    return 0;  // failure
  }
  get_data_uri(resource->name, &resource->data);
  return 1;   // success (implicit in original example)
}

int str_read_callback(mjResource* resource, const void** buffer) {
  *buffer = resource->data;
  return get_data_uri_size(resource->name);
}

void str_close_callback(mjResource* resource) {
  mju_free(resource->data);
}
```

### Registration

```c
mjpResourceProvider resourceProvider = {
  .prefix = "data",
  .open   = str_open_callback,
  .read   = str_read_callback,
  .close  = str_close_callback,
};

if (!mjp_registerResourceProvider(&resourceProvider)) {
  // handle failure
}
```

### MJCF Usage

```xml
<asset>
  <texture name="grid" file="grid.png" type="2d"/>
  <mesh content-type="model/obj"
        file="data:model/obj;base64,I215IG9iamVjdA0KdiAxIDAgMA0KdiAwIDEgMA0KdiAwIDAgMQ=="/>
</asset>
```

## Notes

- The `data` pointer in `mjpResourceProvider` is constant within a model — do not use it for per-resource mutable state; use `mjResource.data` instead.
- Providers are global once registered; there is no per-model scoping.
- Memory allocated in `open` must be freed in `close` — MuJoCo will not manage it.
- See [[mujoco_engine_plugins]] for the plugin extension system.
