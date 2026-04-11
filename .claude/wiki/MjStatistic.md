---
type: api_reference
tags: [mujoco, python, class]
source: https://mujoco.readthedocs.io/en/stable/python.html
---
# MjStatistic

Static model statistics. Accessed as `model.stat` on an [[MjModel]] instance.

## Access

```python
print(model.stat.meanmass)
print(model.stat.extent)
print(model.stat.center)
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `meaninertia` | float | mean diagonal inertia |
| `meanmass` | float | mean body mass |
| `meansize` | float | mean body size |
| `extent` | float | spatial extent of the model |
| `center` | `(3,)` | geometric center of the model |
