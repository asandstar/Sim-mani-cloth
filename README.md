# Sim-mani-cloth
cloth simulation+robotic manipulation
```shell
conda create -n taichi-cloth python python=3.10 -y
conda activate taichi-cloth
pip install taichi
python cloth_ggui.py
```
# Taichi Cloth Tearing Demo (Two Clamps)

This project reproduces and extends a Taichi cloth simulation demo and adds:
- table contact and friction,
- projection-based sphere collision,
- a two-clamp grasp tool,
- scripted manipulation (drop → settle → grasp → lift → pull),
- a controllable mid-seam tearing effect.

The tearing behavior is intentionally engineered for stability and visual clarity,
rather than being a fully physically-accurate fracture model.

## Motivation / Background

Cloth tearing is challenging in real-time simulation because it couples:
- stiff elastic forces,
- contact and friction,
- large deformations,
- topology changes (connectivity updates),
- numerical stability constraints.

The goal of this work is to build a robust demo that:
1) is easy to run and visually inspect,
2) produces a predictable "tear from the middle" behavior,
3) avoids solver explosions during grasp and manipulation.

## System Overview

### Cloth Representation
- Cloth is an `n x n` grid of mass points:
  - position field: `x[i, j]`
  - velocity field: `v[i, j]`
- Springs connect each point to its 8-neighborhood offsets stored in `spring_offsets`.
- Each spring has:
  - rest length `spring_L0[k]`,
  - offset `spring_off[k]`,
  - opposite spring index `spring_opp[k]` (bidirectional consistency),
  - alive flag `spring_alive[i, j, k] ∈ {0, 1}`.

### Forces and Integration
Per substep:
1. Gravity:
   - `v += gravity * dt`
2. Internal forces (for each alive spring):
   - elastic spring force based on stretch ratio
   - dashpot damping along spring direction
3. Air drag:
   - `v *= exp(-drag_damping * dt)`
4. Semi-implicit Euler:
   - `x += dt * v`

### Collisions
- Sphere collision (projection-based):
  - project penetrating points to the sphere surface,
  - remove inward normal velocity component.
- Table contact:
  - project points to `y = table_y + eps`,
  - zero normal downward velocity,
  - apply tangential friction as exponential decay on `v.x` and `v.z`.
- Additional boundary clamp for x/z within `table_bounds`.

## Two-Clamp Grasp Tool

Instead of solving rigid body contact forces, clamps are implemented as a kinematic tool:

- Two clamp positions: `clamp_pos[0]`, `clamp_pos[1]`
- Each clamp controls a neighborhood of cloth points within radius `clamp_R`.
- On "attach", clamp positions are snapped to chosen cloth vertices to avoid sudden impulses.
- During grasp, points in the neighborhood are constrained:
  - `x[ii, jj] = clamp_pos + clamp_dpos[...]`
  - `v[ii, jj] = clamp_vel`

This provides stable grasping and predictable motion control.

## Tearing Design (Mid-Seam Gate)

### Why a gated tearing rule?
A global strain-based break rule tends to produce uncontrolled cracks across the cloth.
To get a consistent "tear from the middle", tearing is restricted to a predefined seam.

### Rule
A spring breaks only if:
- tearing is enabled: `enable_tear == 1`,
- the spring crosses the mid seam (between `i = mid-1` and `i = mid`),
- the stretch ratio exceeds a threshold: `stretch > tear_ratio_dyn`.

Implementation detail:
- breaking is done by flipping the connectivity flag:
  - `spring_alive[i, j, k] = 0` and `spring_alive[j, ..., opp] = 0`
- additionally, cross-seam diagonal springs near the broken segment are also disabled to
  reduce residual mesh connections that visually prevent separation.

### Dynamic threshold
Two thresholds are used:
- `TEAR_RATIO_SAFE`: high threshold used during lift/motion to avoid accidental tearing,
- `TEAR_RATIO_TEAR`: low threshold used during the actual pulling stage.

The threshold is switched once per phase transition to prevent reset/phase glitches.

## Rendering Topology Update

Breaking springs alone does not automatically break triangles in rendering.
We maintain a triangle index buffer and invalidate triangles that rely on broken edges:

- `update_mesh_indices_by_tears()` checks local spring connectivity
- if an edge/diagonal needed by a triangle is broken, triangle indices are set to `DUMMY`

This produces a clear visible separation once the seam connectivity is gone.

## Scripted Stages (Drop–Settle–Manipulate)

A phase-based controller drives the demo:

- Phase 0: Drop  
  Cloth falls onto the table with gravity; clamps are off.

- Phase 1: Attach  
  Clamps snap to target cloth vertices to avoid impulsive grabbing.

- Phase 15: Close  
  Clamp gap transitions from open to close (mainly visual).

- Phase 2: Lift  
  Both clamps move upward to a target height.

- Phase 25: Lower & Notch  
  Lower to a tearing-friendly height; apply a pre-cut notch along the seam.

- Phase 3: Pull & Tear  
  Enable tearing and pull clamps apart at a controlled speed.

- Phase 4: Stop  
  Pause for inspection.

## Key Challenges and How They Were Addressed

1. **Explosions during grasp**
   - Cause: sudden constraint introduction creates large velocity spikes.
   - Fix: snap clamp positions to cloth vertices on attach, then constrain a local patch.

2. **Uncontrolled tearing everywhere**
   - Cause: global stretch-based breaking is too permissive under large deformation.
   - Fix: mid-seam gate restricts eligible springs to a single predefined seam.

3. **Connectivity breaks but mesh still looks connected**
   - Cause: triangles remain in the render index buffer.
   - Fix: invalidate triangles crossing broken edges using `DUMMY` indices.

4. **Reset / phase transitions causing tearing threshold glitches**
   - Fix: phase-entry configuration using `last_phase`, plus explicit reset of
     `enable_tear` and `tear_ratio_dyn`.

## Trade-offs

This demo prioritizes:
- stability,
- controllability,
- a clear and repeatable tearing pattern,

over full physical realism of fracture mechanics.
The tearing logic is a discrete connectivity rule (spring deletion + mesh invalidation),
not a continuous damage/fracture model.

## Milestones (Commit Tags)

- M0: Taichi demo reproduction
- M1: Table contact
- M2: Projection-based collision
- M3: Two-clamp tool (replacing two fingertips)
- M4: Attachment-based grasp
- M5: Drop–Settle–Manipulate + more realistic-looking tearing control

## Controls
- `P`: pause / resume
- `N`: step one frame (when paused)
- `R`: reset
