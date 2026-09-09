# Graph design review: persistent incremental evaluation

Review date: 2026-09-09

Scope: `Library/Graph`, with emphasis on shared curves and other quantitative data.

## Required behavior

A node should be computed on its first request, and its result should then be
reused until a dependency changes. Multiple consumers of the same curve must
share that computation and its cached result.

This review applies the strict interpretation discussed in the design review:
recomputing a predecessor does not require recomputing its consumers if the
predecessor's result remains unchanged. For `quotes -> curve -> prices`, a changed
quote may require recalibrating the curve, but an identical resulting curve should
allow its prices to remain cached.

The guarantee applies to successfully cached results within an evaluation
session. Initial computation and retries after failure are necessary exceptions.
Evicting a cached result also makes later recomputation necessary, so eviction
would explicitly relax the guarantee.

## Assessment

The current design partly satisfies this requirement. A shared node executes once
within a graph run, and `run_incremental()` reuses results when callers provide a
consistent cache and version baseline. However, persistent reuse and precise
change detection are not guarantees of the main execution API.

Keep the immutable DAG, dependency analysis, and scheduler. Introduce a persistent
evaluation layer that owns cached results and the information establishing their
validity.

## Findings

### 1. [P1] Shared dependency discovery can break invalidation

Source: [keyed_graph_builder.h, discovery filtering](../../Library/Graph/keyed_graph_builder.h#L264).

`resolve_with_discovery()` filters out already-resolved keys before invoking the
resolution path that records dependency edges. A second trade requesting an
existing curve can therefore lose its dependency on that curve. Scheduling,
input delivery, pruning, and downstream invalidation then operate on an
incomplete graph.

An executable probe resolved two consumers that both declared the same shared
curve. The first consumer had one dependency; the second had zero.

**Required change:** record every dependency, including dependencies whose nodes
already exist. Memoization should prevent rebuilding a dependency while
preserving every consumer's edge. Discovery must describe the dependencies a
computation uses, including those already available; reporting only globally
missing keys cannot establish the complete dependency set. Repeated discovery
passes also need to preserve input ordering without accidentally adding the same
dependency port repeatedly.

### 2. [P1] Ordinary targeted requests do not retain shared results

Sources: [run_to contract](../../Library/Graph/graph_executor.h#L106),
[intermediate result release](../../Library/Graph/graph_executor.cpp#L446).

The keyed builder memoizes node construction, not evaluated curve objects.
`run_to()` releases intermediate result slots when their scheduled consumers
finish and returns only the requested outputs. A subsequent targeted request
creates fresh execution state.

In a probe, requesting trade A followed by trade B, with both trades depending on
one unchanged curve, computed the curve twice. The requirement calls for one
persistent cached result shared across both requests.

**Required change:** retain successful node results in an evaluation session that
outlives individual requests. Combine target selection and cache reuse in the
normal evaluation path. Release temporary execution references independently of
the persistent cache's ownership.

### 3. [P1] Dirtiness forces unnecessary downstream recomputation

Source: [transitive dirty propagation](../../Library/Graph/graph_executor.cpp#L270).

`run_incremental()` marks every descendant of a dirty node dirty before evaluating
intermediate results. A dirty node is then unconditionally recomputed; there is no
comparison that can preserve a downstream cached result when an intermediate
output stays unchanged.

A probe used an exact integer transformation to make different source inputs
produce the same intermediate result. After changing the source, the quote,
curve, and price execution counts were `2, 2, 2`. Under the strict requirement,
they should have been `2, 2, 1`.

**Required change:** distinguish a node that needs validation from a node whose
output changed. Validate predecessors first, then decide whether their current
output revisions require executing the consumer. Advance a node's output
revision only when its observable result changes.

### 4. [P1 — design risk] Results and cache validity have separate owners

Source: [incremental cache and baseline contract](../../Library/Graph/graph_executor.h#L134).

Callers maintain both `in_out_cache` and `baseline_versions`. Execution updates
the cache but does not advance the external baseline. Repeating a request with an
old baseline forces unnecessary work even though the cache already contains the
latest results. This is a caller-contract hazard, not a failure to implement the
documented baseline comparison.

Both structures also use positional node IDs. The header documents the risk of
IDs changing across rebuilds. Even stable IDs are insufficient when a node's
dependency list or computation recipe changes without an associated version
change. An earlier probe retained IDs and stamps while switching a price's curve
dependency: incremental evaluation returned `10`, while full evaluation returned
`20`, and the incremental status reported success.

**Required change:** manage each cached result together with its observed
dependency revisions, output revision, and computation identity. Use stable
semantic keys and explicit graph-definition or recipe identity to validate reuse
across graph definitions. Updating a result and its validity metadata should be
one coherent publication operation.

### 5. [P2] Market-data revisions are frozen into graph definitions

Source: [builder version stamping](../../Library/Graph/graph_builder.h#L51).

Version stamps are copied into the frozen graph. Stamping the builder after
`build()` does not update the existing graph. The supported version-stamping
workflow therefore requires another build to publish a changed quote revision.

**Required change:** keep source values and their revisions in the evaluation
state. Routine market updates should preserve the compiled topology. Structural
dependency or computation-recipe changes should remain explicit graph-definition
changes with appropriate cache invalidation.

## Proposed ownership model

| Component | Responsibility |
| --- | --- |
| Graph definition | Stable semantic keys, complete ordered dependencies, computation recipes, result types, and equality policies |
| Evaluation session | Cached immutable results, observed dependency revisions, output revisions, consistent market snapshots, and in-flight computations |
| Executor | Validate requested dependencies and schedule computations required by the session |

Each cached node should retain:

- Its immutable result and output revision.
- The dependency identities and output revisions used to compute that result.
- The relevant computation-recipe identity or revision.
- Validity, failure, and in-flight evaluation state for the requested context.

The existing scheduler can remain responsible for execution order. Cache validity
and result publication belong to the evaluation layer.

## Evaluation algorithm

For a requested node:

1. Ensure its dependencies are current for the requested snapshot.
2. Compare their output revisions and the computation recipe with the metadata
   recorded alongside the cached result.
3. Reuse the cached result when those inputs match and the entry is valid.
4. Otherwise, compute once for that dependency state and compare the new result
   with the previous result, when one exists.
5. Record the dependency revisions used by the successful computation, including
   when the resulting value is unchanged.
6. Advance the node's output revision only when its observable result changes.
   When the result is equivalent, retain the previous result and output revision.

An upstream change may mark descendants as needing validation. It must not
automatically require every descendant's work function to run. Evaluation should
validate the ancestors of requested outputs and leave unrelated work cached.

## Contract details

### Equality and revisions

`std::any` has no generic equality operation. Each result type needs an explicit
semantic equality or trustworthy output-revision policy. For a curve, this must
cover everything consumers can observe. Pointer identity and an arbitrary
floating-point tolerance are insufficient definitions of unchanged output.

If a result type cannot establish equivalence, conservative downstream
recomputation is possible, but it does not provide the strict unchanged-output
guarantee for that type.

### Complete inputs and consistent snapshots

Every changing input that affects a result must be represented in the validity
model. This includes valuation date, calibration settings, model parameters, and
random-stream identity when applicable. Callables that read changing external
state without declaring it as an input cannot support reliable cache reuse.

Each computation must consume a consistent input snapshot and publish its result
against the exact revisions it consumed. A computation started for an older
snapshot must not overwrite a newer snapshot's valid result.

### Shared execution and result ownership

Concurrent requests for the same node, recipe, and dependency state must share
one in-flight computation. Requests for different dependency states must remain
distinct.

Prefer immutable result handles so consumers can share curves and larger data
objects. The current executor copies contained `std::any` values while gathering
inputs and serving clean cached nodes. Persistent cache ownership should avoid
turning result reuse into repeated payload duplication.

Retain cached results for the evaluation session's defined lifetime. Releasing
temporary buffers is compatible with persistent caching; evicting the cached
result itself explicitly permits future recomputation.

## Acceptance cases

| Case | Required outcome |
| --- | --- |
| First request | Compute each required node once |
| Repeated unchanged request | Execute no cached node work again |
| Sequential trades sharing one curve | Reuse the same cached curve |
| Concurrent requests sharing one curve and input state | Share one in-flight curve computation |
| Unrelated quote changes | Preserve unaffected cached branches |
| Relevant quote changes and curve output changes | Recompute the curve and requested affected consumers |
| Relevant quote changes but curve output is identical | Recompute the curve and retain consumer results |
| Same-value source update | Preserve the source output revision under its equality policy |
| Dependency or recipe changes | Invalidate incompatible cached results, even with stable node IDs |
| Evaluation fails | Report failure without presenting an older result as valid for the new inputs |
| Older and newer snapshots overlap | Publish and reuse results only within compatible snapshot state |

## Validation performed

The earlier repository build and configured test run passed, including all 41
Graph tests. No repository implementation changes were made during the reviews.

Temporary executable probes compiled against the built Graph library produced:

| Probe | Observed result |
| --- | --- |
| Unchanged incremental request | Quote, curve, and price each remained at one execution |
| Changed source with equal intermediate output | Quote, curve, and price each reached two executions; the price should have remained at one |
| Two targeted trade requests sharing an unchanged curve | Curve executed twice |
| Repeated request with an unadvanced external baseline | All nodes executed again |
| Discovery of an already-resolved shared dependency | First consumer had one dependency; second had zero |
| Changed edge with unchanged positional IDs and stamps | Incremental result was 10; full result was 20 |

These probes are review evidence, not committed regression tests. The acceptance
cases above should become implementation tests when the evaluation contract is
introduced. The existing passing suite validates the current behavior; it does
not establish compliance with the stricter persistent-caching requirement.
