# Project dependencies

Inter-library links for XSigma (`Library/*`). Third-party packages are
summarized here and detailed in
[readme/third-party-dependencies.md](readme/third-party-dependencies.md).

CMake optional edges are `TARGET Xxx::Xxx` checks (or `MEMORY_ENABLE_*`).
Bazel hard-links the same edges. There is **no cycle**: Profiler must not
depend on Memory or Core. Logging is a **required** dependency of Memory,
Vectorization, and Core. Memory is a **required** dependency of Vectorization
and Core.

## Graphs

Arrows mean **links against** (A → B = A depends on B). Solid = always in a
full-tree build. Dashed = CMake-only optional (`HAS_*=0` if the target is
missing).

### Full tree (`Library/*`)

```mermaid
flowchart TB
  subgraph foundations["No Library/* deps"]
    Logging
    Profiler
    Models
  end

  subgraph mid["Optional / gated consumers"]
    Memory
    Parallel
    Vectorization
  end

  subgraph top["Always-on consumers"]
    Core
    Graph
  end

  Memory --> Logging
  Memory -.->|MEMORY_HAS_PROFILER| Profiler
  Parallel -.->|PARALLEL_HAS_PROFILER| Profiler
  Vectorization --> Logging
  Vectorization --> Memory
  Vectorization -.->|VECTORIZATION_HAS_PROFILER| Profiler
  Core --> Logging
  Core --> Memory
  Graph --> Parallel

  classDef leaf fill:#e8f5e9,stroke:#2e7d32
  classDef gated fill:#fff8e1,stroke:#f9a825
  classDef always fill:#e3f2fd,stroke:#1565c0
  class Logging,Profiler,Models leaf
  class Memory,Parallel,Vectorization gated
  class Core,Graph always
```

In a **full CMake configure** or **any Bazel build**, the dashed edges are
present (defaults ON / targets exist). `--project.memory` still builds Logging
(required) and drops the Profiler edge. `--project.profiler` builds Profiler alone.

`Models` has no `Library/*` link. `Graph` requires `Parallel::Parallel` and
inherits its Profiler dependency when enabled. Its current executor uses
Parallel's standard-thread callback queue. Financial Models integration is
future work; see the [Graph guide](graph/README.md).

### Bazel (unconditional)

Same nodes; every dashed edge above is a hard `deps` entry:

```mermaid
flowchart LR
  Logging
  Profiler
  Memory --> Logging
  Memory --> Profiler
  Vectorization --> Logging
  Vectorization --> Memory
  Vectorization --> Profiler
  Core --> Logging
  Core --> Memory
  Parallel --> Profiler
  Graph --> Parallel
  Models
```

### CMake add_subdirectory order

Profiler is configured **before** Memory / Vectorization / Parallel so
`TARGET Profiler::Profiler` succeeds.

```mermaid
flowchart LR
  L[1 Logging] --> P[2 Profiler] --> M[3 Memory] --> V[4 Vectorization] --> C[5 Core] --> Par[6 Parallel] --> Mod[7 Models] --> G[8 Graph]
```

### `--project.NAME` scopes

What `XSIGMA_LIBRARY_PROJECT` actually `add_subdirectory`s (not the
link graph — only these modules exist in that configure):

```mermaid
flowchart TB
  subgraph logging["--project.logging"]
    L1[Logging]
  end
  subgraph profiler["--project.profiler"]
    P1[Profiler]
  end
  subgraph memory["--project.memory"]
    M1[Memory] --> Lmem[Logging]
  end
  subgraph vectorization["--project.vectorization"]
    V2[Vectorization] --> L2[Logging]
    V2 --> M2[Memory]
    V2 --> P2[Profiler]
  end
  subgraph parallel["--project.parallel"]
    Par2[Parallel] --> P3[Profiler]
  end
  subgraph core["--project.core"]
    C3[Core] --> L3[Logging]
    C3 --> M3[Memory]
    M3 --> P4[Profiler]
    V3[Vectorization]
    Par3[Parallel]
  end
  subgraph models["--project.models"]
    Mod1[Models]
  end
  subgraph graph_scope["--project.graph"]
    G1[Graph] --> Par4[Parallel]
    Par4 --> P5[Profiler]
  end
```

### Third-party (typical)

```mermaid
flowchart LR
  Logging --> fmt
  Logging --> loguru_glog_spdlog["loguru / glog / spdlog"]
  Profiler --> fmt
  Profiler --> kineto_or_itt["kineto xor ittapi"]
  Profiler --> gpu_rt["CUDA / HIP / Metal"]
  Memory --> fmt
  Memory --> cpuinfo
  Memory --> mimalloc
  Memory --> gpu_rt
  Vectorization --> sleef_svml_mkl["SLEEF / SVML / MKL"]
  Core --> fmt
  Core --> cpuinfo
  Core --> magic_enum
  Parallel --> tbb_omp["TBB / OpenMP / Threads"]
```

Vendored trees live under `ThirdParty/` — do not edit them. See
[readme/third-party-dependencies.md](readme/third-party-dependencies.md).

## Libraries

| Library | Role | Depends on (`Library/*`) | Compile-time gates |
|---|---|---|---|
| **Logging** | Log backends (native / loguru / glog / spdlog) | none | — |
| **Profiler** | Native XPlane + Kineto/ITT | none | `PROFILER_HAS_KINETO` / `PROFILER_HAS_ITT`; `PROFILER_HAS_METAL` / `CUDA` / `HIP` |
| **Memory** | Allocators, GPU pools | Logging, Profiler | `MEMORY_HAS_PROFILER` |
| **Vectorization** | SIMD / GPU packets | Logging, Memory, Profiler | `VECTORIZATION_HAS_PROFILER` |
| **Core** | Legacy computational core | Logging, Memory | (inherits Memory’s Profiler link when Memory has it) |
| **Parallel** | Thread pools / TBB / OpenMP | Profiler | `PARALLEL_HAS_PROFILER` |
| **Models** | SABR/ZABR + QA calibrator | none | — |
| **Graph** | Dependency DAG construction and execution | Parallel (required); Profiler through Parallel | inherits `PARALLEL_HAS_PROFILER` through Parallel |

### Memory gates

`MEMORY_HAS_PROFILER` is **computed**, not a cache option:

- Starts at **0**.
- Becomes **1** when `MEMORY_ENABLE_PROFILER` is ON (default **ON**) **and**
  `Profiler::Profiler` exists.
- Bazel always defines `MEMORY_HAS_PROFILER=1` (Memory always `deps` Profiler).

Logging is **required**: Memory always links `Logging::Logging`. There is no
`MEMORY_ENABLE_LOGGING` / `MEMORY_HAS_LOGGING` opt-out. `--project.memory`
configures Logging + Memory.

### Vectorization / Parallel gates

These are **not** user options. They follow whatever targets CMake has
already created:

| Macro | 1 when |
|---|---|
| `VECTORIZATION_HAS_PROFILER` | `Profiler::Profiler` exists |
| `PARALLEL_HAS_PROFILER` | `Profiler::Profiler` exists |

Logging and Memory are **required** for Vectorization (`FATAL_ERROR` if
`Logging::Logging` or `Memory::Memory` is missing). There is no
`VECTORIZATION_HAS_LOGGING` or `VECTORIZATION_HAS_MEMORY` gate.

## CMake configure order

Root `CMakeLists.txt` `_xsigma_lib_order`:

1. Logging
2. Profiler
3. Memory
4. Vectorization
5. Core
6. Parallel
7. Models
8. Graph

Profiler is early on purpose so `TARGET Profiler::Profiler` is true when
Memory, Vectorization, and Parallel run. Graph is configured after Parallel
because it requires the `Parallel::Parallel` target.

### `--project.NAME` / `XSIGMA_LIBRARY_PROJECT`

Only the listed modules are `add_subdirectory`’d (order above still
applies):

| `--project.` | Modules configured |
|---|---|
| `logging` | Logging |
| `profiler` | Profiler |
| `memory` | Logging, Memory |
| `vectorization` | Logging, Memory, Vectorization, Profiler |
| `parallel` | Parallel, Profiler |
| `core` | Parallel, Profiler, Logging, Memory, Vectorization, Core |
| `models` | Models |
| `graph` | Profiler, Parallel, Graph |
| *(empty)* | all eight |

`--project.memory` therefore has `MEMORY_HAS_PROFILER=0` (Profiler is not
configured) and always has Logging. `--project.profiler` does not build Memory.

## Third-party (by library)

Always-vendored under `ThirdParty/` — do not edit those trees.

| Library | Typical third-party / system deps |
|---|---|
| Logging | fmt; one of loguru / spdlog / glog (`LOGGING_BACKEND`, default **LOGURU**) |
| Profiler | fmt; kineto **or** ittapi; CUDA / HIP / Metal+Foundation when that GPU backend is on |
| Memory | fmt, cpuinfo; optional mimalloc, tbbmalloc, numa, CUDA / HIP / Metal |
| Vectorization | Logging/Memory/Profiler as above; optional SLEEF, SVML, MKL, Accelerate, LibTorch (tests), CUDA / HIP / Metal |
| Core | fmt, cpuinfo; optional magic_enum, Enzyme, compression/snappy, MKL |
| Parallel | Threads; optional TBB, OpenMP |
| Models | none |
| Graph | Threads and transitive dependencies through Parallel; no additional direct third-party dependency |

Test binaries additionally link Google Test (and Google Benchmark when
enabled).

## Bazel notes

- Package graph matches the mermaid diagram. CMake optional edges (Profiler on
  Memory/Vectorization/Parallel) are unconditional `deps` in Bazel. Logging and
  Memory are required for Vectorization in both build systems.
- Profiler `BUILD.bazel` must **not** take `//Library/Core` or
  `//Library/Memory` — that would cycle with Memory → Profiler.
- Metal / CUDA / HIP are `select()`s on `--define=memory_enable_*` /
  `vectorization_enable_*` (see `.bazelrc` `build:metal` and
  `Scripts/setup_bazel.py --gpu_backend`).

## Related docs

- [PROJECT_FLAGS.md](PROJECT_FLAGS.md) — CMake cache flags
- [Graph guide](graph/README.md) — DAG execution, caching contracts and pricing integration status
- [readme/third-party-dependencies.md](readme/third-party-dependencies.md) — vendored packages
- [profiler/profiler.md](profiler/profiler.md) — Profiler instrumentation and `HAS_PROFILER` call sites
- [BAZEL_USER_GUIDE.md](BAZEL_USER_GUIDE.md) — Bazel configs and known gaps
