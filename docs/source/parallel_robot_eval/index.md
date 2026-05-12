# Parallelising Robot Simulations

Evaluating a population of robots is the main bottleneck in evolutionary robotics — every individual needs to be simulated before selection can proceed, and simulations are fully independent of each other.
ARIEL supports two complementary approaches to parallelising this step.

---

## Choosing an approach

| | MuJoCo Worker | Ray |
|---|---|---|
| **Best for** | Single-machine physics simulation | Multi-machine clusters or general Python tasks |
| **Parallelism mechanism** | `multiprocessing.Pool` with `spawn` context | Distributed actor/task system |
| **Dependencies** | Python standard library only | Requires `ray` |
| **MuJoCo isolation** | ✅ Spawn context avoids OpenGL/thread conflicts | ⚠️ Requires manual care around MuJoCo state |
| **Cluster deployment** | ❌ One machine | ✅ Multi-machine and cloud |
| **Per-task overhead** | Low — direct OS process fork | Higher — Ray object store serialisation |
| **Setup complexity** | Subclass one class, pass to `pool.imap` | `ray.init()` + `@ray.remote` decorator |
| **Debugging** | Standard Python tracebacks | Ray dashboard and distributed logs |

**If you are running experiments on a single workstation**, use the MuJoCo Worker approach — it has less overhead, plays nicely with MuJoCo's internal threading, and requires no additional services.

**If you need to scale beyond one machine**, or want to distribute heterogeneous tasks (simulation, neural-network training, data processing) across a cluster, use Ray.

---

## Pages

```{toctree}
:maxdepth: 2

mujoco_worker
parallel_ackley
```
