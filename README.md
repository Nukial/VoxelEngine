# VoxelEngine

> GPU voxel simulation and ray-marched rendering research prototype for Unity.

VoxelEngine is a source-available research project that explores how far Unity can be pushed when voxel data, simulation, and rendering are moved away from the traditional GameObject/MeshRenderer workflow and into GPU buffers, compute shaders, and ray marching.

This repository is **not** presented as a complete production-ready game engine. It is a technical prototype for studying GPU-driven voxel simulation, volume rendering, packed voxel data, adaptive quality, and the practical limits of using Unity as a host/editor around a custom high-performance rendering path.

## Demo video

A visual demo video will be added here after recording.

<!-- Replace this placeholder with a YouTube link or embedded media when ready. -->

```md
[Watch the demo](YOUR_VIDEO_LINK_HERE)
```

## Project status

**Status:** research prototype / experimental engine core  
**Target:** Unity GPU voxel simulation and rendering experiments  
**Recommended Unity version:** Unity `6000.3.8f1`  
**Render pipeline:** Universal Render Pipeline based project setup  
**License:** Source-available research license. See [`LICENSE`](LICENSE).

## Why this project exists

The original idea behind VoxelEngine was to test whether a voxel world could be simulated and rendered in Unity without following Unity's usual object-oriented rendering model.

Unity is already very strong when a project fits its normal rendering pipeline: meshes, renderers, batching, culling, physics, lighting, prefabs, and editor workflows. VoxelEngine intentionally explores a different direction:

- store the world as packed voxel data,
- simulate materials on the GPU,
- render the volume through a ray-marched shader,
- use brick maps to skip empty regions,
- adjust ray/shadow quality dynamically under GPU pressure,
- use Unity mainly as the editor, input layer, scene host, and build environment.

This makes the project valuable as a research prototype, but it also exposes why high-performance voxel volume rendering does not automatically become a good game architecture.

## Core technical ideas

### Packed voxel data

Each voxel is stored as a 32-bit unsigned integer:

- 8 bits: material ID,
- 16 bits: RGB565 color,
- 8 bits: auxiliary data, currently used for temperature and light.

This keeps the simulation compact and GPU-friendly.

### GPU buffers and double buffering

The world is stored in GPU `GraphicsBuffer` objects. Simulation uses a read buffer and a write buffer so each step can be computed from a stable previous state.

### Compute shader simulation

The simulation layer is implemented with compute shaders. It includes early experimental behavior for:

- powder materials such as sand/snow/coal,
- liquid materials such as water/lava,
- gas-like steam movement,
- lava/water reactions,
- heat propagation,
- light propagation,
- simple burn/char/melt/evaporate state changes.

### Ray-marched voxel rendering

The voxel volume is rendered through a custom ray-marching shader instead of generating a normal mesh for every visible surface. A brick map is used to detect empty areas and skip parts of the volume traversal.

### Adaptive quality

Runtime quality can be adjusted based on camera movement, frame time pressure, distance, and edge cases around the volume. This is intended to keep the visual demo stable under heavier GPU load.

## What this project is good for

VoxelEngine is useful for:

- studying GPU voxel data layouts,
- testing compute shader cellular automata,
- exploring voxel ray marching in Unity,
- understanding GPU/CPU synchronization problems,
- experimenting with brick-map acceleration,
- creating visually interesting voxel simulation demos,
- learning where Unity's default pipeline helps and where it becomes a constraint.

## What this project is not

VoxelEngine is currently **not**:

- a Minecraft-like production engine,
- a complete game framework,
- a streaming infinite voxel world,
- a ready networking solution,
- a physics/collision replacement for Unity Physics,
- a general-purpose terrain system,
- a polished tool for level designers,
- a commercial SDK.

## Key limitations and research problems

### 1. Unity pipeline mismatch

Unity is highly optimized for meshes, renderers, culling, batching, prefabs, physics, and scene objects. VoxelEngine bypasses many of those systems because the real world data lives in GPU buffers.

This creates a trade-off: the GPU path can be very powerful for simulation and visuals, but it does not naturally integrate with Unity gameplay systems.

### 2. CPU and GPU synchronization

Gameplay often needs CPU-side answers: what voxel was clicked, what material is at a position, whether something collides, whether an entity can move somewhere, and so on.

When the authoritative voxel state lives on the GPU, reading it back to the CPU can introduce latency and overhead. This is one of the most important practical limits of this architecture.

### 3. Collision and physics are not solved

Unity Physics does not automatically understand a custom voxel volume stored in a `GraphicsBuffer`. Real gameplay would need a dedicated collision representation, hybrid CPU cache, generated colliders, or a separate physics approximation.

### 4. World streaming is not complete

The current direction is suitable for bounded voxel volumes and controlled demos. A real game-scale world would need chunk streaming, memory paging, region updates, serialization, and careful GPU memory management.

### 5. Rendering is impressive but not enough for a game

A simulation can look visually rich without forming a meaningful game loop. To become a game, the system still needs rules, progression, interaction design, content tools, save/load, AI, UI, and performance budgets for real gameplay.

### 6. Editing many voxels requires a better path

Single voxel edits and simple sphere edits are enough for experimentation, but large-scale editing should eventually move toward batched GPU-side update commands instead of many small CPU-to-GPU writes.

### 7. Debugging GPU simulation is difficult

Compute shader logic can be harder to inspect than CPU gameplay code. Bugs may appear as visual artifacts, unstable simulation, race-condition-like behavior, or platform-specific rendering problems.

## Suggested future direction

The most realistic long-term direction is a **hybrid architecture**:

- use GPU voxel simulation for special regions such as sand, liquid, fire, destruction, smoke, or magical effects,
- use chunk meshes or other conventional representations for stable terrain,
- keep gameplay-critical state accessible to CPU systems,
- use Unity for editor workflow, camera, input, UI, audio, build pipeline, and high-level scene management,
- keep the pure GPU volume path as a research/demo layer rather than forcing the whole game to depend on it.

## References and acknowledgements

VoxelEngine acknowledges [`JorisAR/GDVoxelPlayground`](https://github.com/JorisAR/GDVoxelPlayground) as an important technical reference and inspiration source.

GDVoxelPlayground is a Godot-based voxel playground that implements cellular automata voxel simulation rendered using ray marching. Its documented feature set includes a ray marcher optimized using a brick map and cellular automata with multiple materials.

VoxelEngine explores related ideas in Unity/C#/HLSL, including GPU-style voxel data, ray-marched rendering, brick-map acceleration, and material simulation. VoxelEngine is a separate research prototype and is not affiliated with, endorsed by, or maintained by JorisAR.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and MIT license notice information related to GDVoxelPlayground.

## Repository layout

Important areas of the project include:

```text
Assets/VoxelEngine/Scripts/      C# runtime systems
Assets/VoxelEngine/Shaders/      Compute shaders and ray-marching shader
Assets/VoxelEngine/Editor/       Unity editor helpers
Assets/Scenes/                   Demo scene assets
Packages/manifest.json           Unity package dependencies
ProjectSettings/                 Unity project settings
```

## Getting started

1. Clone the repository.
2. Open it with Unity `6000.3.8f1` or a compatible Unity 6 version.
3. Allow Unity to restore packages from `Packages/manifest.json`.
4. Open the demo scene under `Assets/Scenes/`.
5. Enter Play Mode and inspect the `VoxelWorld` object settings.

If the project fails to open or render correctly, first check:

- Unity version compatibility,
- URP configuration,
- compute shader support on the target GPU,
- graphics API selection,
- shader compilation errors,
- package restore errors.

## Hardware expectations

This project is GPU-oriented. Performance depends heavily on:

- GPU compute capability,
- VRAM bandwidth,
- world size,
- voxel scale,
- ray step count,
- shadow step count,
- simulation tick rate,
- light propagation passes,
- render distance.

Lowering ray steps, shadow steps, render distance, simulation tick rate, and light propagation passes can significantly improve performance.

## Contributing

This repository is primarily a personal research project. Contributions, issues, benchmark reports, and technical discussion are welcome if they stay aligned with the research goal.

Please do not submit changes that turn the project into a general-purpose game framework unless they are clearly separated from the research core.

## License

This project is released under a custom **Source-Available Research License**.

You may read the code, study it, fork it for personal non-commercial research, and reference the ideas with credit.

You may not use this project as the base of a commercial product, sell it, repackage it, redistribute it as an engine/plugin/SDK, or remove attribution without explicit written permission from the copyright holder.

See [`LICENSE`](LICENSE) for the full terms.

Third-party references and notices are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Disclaimer

VoxelEngine is provided as-is. It is experimental software and may contain bugs, performance problems, platform-specific issues, incomplete systems, or unstable behavior. Use it for study and experimentation, not as a guaranteed production foundation.