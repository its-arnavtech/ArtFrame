# ArtFrame

`ArtFrame` is a modular real-time webcam artwork. Its original two-hand AR strip remains intact, including the risograph, cyanotype, and stippling style renderers. A persistent ModernGL fluid simulation transports two-hand velocity and dye through a lower-resolution GPU field. Independent display-resolution stages turn those fields into ink, fluid glass, or chromatic liquid and can then apply a physical-print Risograph treatment.

Both tracked hands can occlude the liquid with a stable, feathered landmark silhouette. The mask and unmodified camera frame are uploaded to persistent GPU textures, so real hand pixels are composited above the liquid without framebuffer readback.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run.py
```

ArtFrame maintains a name-aware catalog of cameras exposed by the platform's native OpenCV backend. The catalog refreshes in the background, so cameras inserted or removed while the app is running appear without restarting. NVIDIA Broadcast is filtered by name before a capture is opened. The configured Logitech C920e is preferred by stable device name, with camera index `0` as a fallback preference if that name is absent. The preferred device receives a short head start; if its driver stalls or it does not deliver a frame, other permitted cameras are validated concurrently and the first working camera becomes active. The default capture request is native FHD (`1920x1080`, MJPEG, 30 FPS); the capture thread downsamples it once to an HD (`1280x720`) artwork/display frame. The actual negotiated camera mode is printed at startup because camera drivers may choose a nearby supported mode.

Camera discovery, capture, switch validation, and MediaPipe perception run outside the render thread. Slow device drivers, camera delivery, or hand inference therefore cannot block animation, input, or GPU presentation. Both capture and tracking use latest-frame semantics: stale queued frames are dropped instead of adding latency. Interaction state is updated only for a completed tracking result, while the liquid source stabilizer predicts a short distance between results to keep 60 FPS motion continuous.

### Camera troubleshooting

- ArtFrame prints every detected camera plus the active device, OpenCV backend, negotiated resolution, and FPS.
- Press `C` to cycle through available permitted cameras. A replacement becomes active only after it delivers a valid frame; a failed switch leaves the current camera running.
- If no camera is present, ArtFrame remains open and automatically connects when one is inserted.
- NVIDIA Broadcast is excluded before capture. The Windows build enumerates and captures through Media Foundation rather than opening unknown numeric indices.
- If no physical camera appears, check Windows **Privacy & security -> Camera** and close other camera applications. ArtFrame will continue polling without requiring a restart.
- Tracking uses the MediaPipe Tasks `HandLandmarker` API and the bundled `assets/models/hand_landmarker.task` model. The one-time TensorFlow Lite delegate and feedback-manager lines are upstream runtime diagnostics, not deprecation warnings.
- A protobuf deprecation warning indicates an old environment. Reinstall `requirements.txt`; the pinned MediaPipe Tasks runtime does not require protobuf.

The app retains OpenCV webcam capture, MediaPipe tracking, and the existing CPU strip/style composition. The composed background, raw camera foreground, and one-channel hand mask cross the CPU/GPU boundary as normalized 8-bit (`f1`) texture uploads; integer textures are reserved for integer samplers. The simulation, artistic material, occlusion composition, and final presentation remain GPU-resident with no framebuffer readback. If GPU initialization fails, the OpenCV rendering path remains available and applies the same hand mask with CPU alpha composition.

## Controls

- `1`: risograph style
- `2`: cyanotype style
- `3`: stippling style
- `space`: next style
- `S`: toggle the legacy two-hand strip layer (off by default)
- `W`: toggle the fluid-motion layer (on by default)
- `C`: cycle through cameras detected at startup or inserted while running
- `v`: cycle liquid debug visualization
- `m`: next liquid material (`fluid_glass` -> `pinch_fluid` -> `chromatic` -> `ink`)
- `p`: next liquid palette
- `k`: next Riso palette
- `r`: next Riso quality profile
- `t`: toggle delayed GPU timer-query instrumentation
- `q` or `ESC`: quit

## Architecture

- `app/camera`: name-aware hot-plug catalog, nonblocking validated camera switching, and threaded webcam capture
- `app/tracking`: asynchronous MediaPipe detection, fingertip extraction, anchors, and smoothing
- `app/interaction`: tracker-neutral `HandControl` and `InteractionState`
- `app/geometry`: strip construction and perspective warping
- `app/styles`: risograph, cyanotype, stippling, and style registry
- `app/graphics/backends`: GLFW display and concrete ModernGL backend
- `app/graphics/liquid`: configuration, stable sources, solver resources, pass graph/executor, stress inputs, and orchestration
- `app/graphics/liquid/materials`: solver-independent material interface, registry, palettes, and display renderer
- `app/graphics/print`: solver-independent print-treatment contract, Risograph configuration, palettes, quality profiles, deterministic screening, renderer, and persistent resources
- `app/graphics/layer_compositor.py`: final camera/art/hand layer composition
- `app/graphics/particles`: independent particle architecture
- `app/graphics/shaders`: numerical, material, layered-composite, and presentation GLSL passes
- `app/compositing`: strip overlay and tracker-neutral hand-mask generation
- `app/ui`: controls and optional diagnostics

The dependency direction remains:

```text
PERCEPTION -> INTERACTION STATE -> GPU SIMULATION -> MATERIAL -> PRINT TREATMENT -> LAYER COMPOSITION -> DISPLAY
```

Graphics consumes `InteractionState`, never detections or landmarks. Hand landmark conversion stops at `app/compositing/hand_occlusion.py`; the graphics API receives only a camera image and mask texture. Persistent textures, framebuffers, programs, fullscreen geometry, and optional query rings are owned below `app/graphics`. `main.py` remains orchestration only.

## Rendering pipeline

```text
mirrored camera -> tracking -> InteractionState
                -> landmark hull -> expand -> feather -> temporal mask

composed camera background ----+
solver fields -> liquid material -> Riso print treatment -+
composed camera background ------------------------------+-> GPU layered composite -> presentation
raw camera + hand mask ----------------------------------+
```

The layered composite order is camera background, liquid material, optional print treatment, then real hand pixels. Camera and mask textures use the same top-left image coordinates and the same OpenGL Y conversion. The fluid fields use normalized bottom-left UVs; hand source and material interaction uniforms explicitly invert normalized Y once. Simulation resolution is independent of camera and display resolution. Fluid glass is the default material and the Risograph print treatment is disabled by default; both architectures remain independently selectable/configurable.

## Fluid simulation

The GPU graph executes:

```text
velocity injection -> velocity advection -> curl -> vorticity confinement
-> closed-wall boundary -> divergence -> pressure (Jacobi N times)
-> velocity projection -> closed-wall boundary
-> dye injection -> dye advection
```

Velocity, dye, and pressure use ping-pong framebuffers. Divergence and curl have dedicated single-channel half-float targets. The solver ends after field generation. A separate material pass samples those lower-resolution fields into a display-resolution half-float target. The print and final-composition stages consume those results without owning or changing the solver. The default 1280x720 HD display with `simulation_scale=0.375` runs numerical passes at 480x270 with 18 pressure iterations. Camera resolution, display resolution, tracking resolution, and fluid resolution remain independent, so higher-quality input does not multiply solver cost.

Advection backtraces normalized coordinates and uses GPU linear filtering for bilinear sampling. Textures do not repeat, and neighbor reads clamp to half a texel inside the domain. A dedicated velocity pass zeros the outer 1.5 texels before divergence and after projection, producing a contained closed/no-slip wall. Pressure uses clamped neighbors for a simple zero-normal-gradient wall condition. The boundary pass also clears non-finite velocity and enforces a documented magnitude ceiling to protect half-float fields from isolated spikes.

Vorticity confinement derives force from the curl field and its magnitude gradient; it does not add procedural noise. Its conservative default restores small eddies lost through semi-Lagrangian dissipation.

`LiquidSourceStabilizer` is separate from tracking smoothing. It limits normalized source velocity while preserving direction, applies timestep-aware exponential response to position, velocity, pinch, openness, and influence, briefly predicts source position between asynchronous tracking results, and holds a disappearing source before smoothly fading its simulation and material influence. Raw perception state remains untouched.

The frame delta is scaled and clamped before simulation. Velocity and dye retention values are exponentiated relative to 60 Hz, so decay is not directly frame-rate dependent. Optional dye diffusion defaults to zero.

## Hand occlusion

`HandMaskGenerator` accepts neutral `HandDetection` values rather than importing MediaPipe. For each sufficiently confident hand it normalizes the 21 landmark positions, builds and fills a convex hull, dilates it with an elliptical kernel, feathers it with a Gaussian blur, and applies timestep-aware exponential temporal smoothing. Both hands are accumulated into one mask. By default the mask is generated at half display resolution only when tracking completes, then enlarged once and reused until the next result.

Expansion, feather radius, confidence threshold, temporal response, and the enabled flag live in `AppConfig.hand_occlusion`. Expansion and feather are fractions of the shorter frame dimension, keeping their visual scale independent of camera resolution. This first implementation intentionally favors stability and speed over semantic segmentation; it cannot recover untracked wrist/forearm pixels or gaps beyond the landmark hull.

## Liquid materials and palettes

`LiquidMaterial` describes a fragment shader plus shared uniforms. It receives dye, velocity, curl/vorticity, pressure, display/simulation metadata, elapsed time, palette colors, and `InteractionState`; it never owns or advances the solver and knows nothing about MediaPipe.

Available materials:

- `ink`: layered pigment density, translucent edges, flow-aligned paper grain, local gradients, pressure, velocity, and curl detail
- `fluid_glass`: flow-driven screen-space transmission, density-gradient lens normals, restrained spectral dispersion, Fresnel reflection, internal scene reflection, softbox highlights, curvature shadows, and edge caustics. It intentionally avoids independent animated noise and mirrored chrome so stationary fluid remains optically stable and transparent.
- `pinch_fluid`: filled refractive liquid volumes centered on each thumb/index midpoint. Finger motion stretches the volume into a short trail and drives its internal refraction, dye, pressure, and curl detail; loss of tracking fades the complete volume instead of removing it in one frame.
- `chromatic`: restrained field-dependent channel separation with velocity and vorticity color variation

Available palettes:

- `neutral_chrome` (default for fluid glass)
- `cyan_blue`
- `magenta_orange`
- `blue_violet`
- `monochrome_ink`

Custom `LiquidPalette` instances can be registered without changing the solver or shaders. Material appearance currently maps hand velocity to energy/detail, average pinch to intensity, and distance between hands to material scale. Lightweight procedural grain is static in material space and modulated by flow; it is not random animated noise.

## Risograph print treatment

The GPU print stage is architecturally separate from both simulation and material rendering. Its first pass derives primary ink density, secondary ink density, edges, and breakup from actual dye density, material coverage, velocity, and vorticity. Its second pass renders two stable angled screens, subtle registration offsets, subtractive-style ink overlap, paper color/fibers, deterministic grain, optional posterization, and a small motion-aware persistent-history blend.

The effect is anchored to display pixels and uses no frame-randomized noise, preventing stationary dots and paper grain from crawling. Low dye produces sparse coverage, medium dye exposes the halftone, and high dye produces dense overlapping inks. Hand velocity adds structured detail and local registration movement; pinch increases concentration, openness broadens coverage, and hand distance increases spread.

Riso palettes are independent from liquid palettes and define primary ink, secondary ink, paper, and accent:

- `cyan_blue`
- `magenta_orange`
- `blue_violet`
- `monochrome_ink`

Quality profiles do not change display or fluid resolution:

- `draft`: coarser screen detail, reduced paper/registration detail, no temporal history
- `standard`: balanced screen, paper, registration, and light history
- `high`: finer screen/detail response, full registration behavior, and stronger stable history

`AppConfig.print_treatment` controls enablement, treatment selection, palette, quality, dot scale/strength, threshold, screen angle, density response, registration, paper/grain, edge breakup, and posterization.

## Configuration

GPU settings live in `AppConfig`; solver settings live in its `liquid` field:

- native camera capture width, height, and requested FPS
- camera catalog refresh interval and excluded device-name tokens
- independent HD output/display and low-resolution tracking sizes
- target display FPS and vsync
- simulation/visualization enabled flags
- simulation scale
- timestep scale and maximum timestep
- velocity injection strength and mapping scale
- source radius and source stabilization times
- maximum source and fluid velocities
- velocity and dye decay
- dye injection, velocity coupling, and optional diffusion
- pressure iteration count
- vorticity strength
- per-hand colors
- initial debug view
- optional GPU timing and query lag

Artistic settings live in `AppConfig.liquid_art`: initial material, palette, intensity, procedural texture strength, glass refraction, dispersion, roughness, and edge brightness. Hand-mask settings live independently in `AppConfig.hand_occlusion`.

Benchmark helpers expose common simulation scales plus 10, 20, 30, and 40 pressure iterations. Runtime reconfiguration creates new size-dependent targets first, swaps them in, releases old targets, and reuses compiled programs.

The clean artwork view is the default (`debug_hud=False`). The HUD and CPU finger overlay can be enabled independently for diagnostics. A 60 FPS frame pacer prevents an unbounded render loop from starving the asynchronous tracker on systems where vsync is unavailable or ineffective.

## Performance architecture

The default quality/performance split is:

```text
camera       1920x1080 @ 30 FPS (native detail)
display      1280x720  @ up to 60 FPS (HD artwork)
tracking      480x270  (asynchronous latest frame)
hand mask     640x360  (updated with tracking)
fluid         480x270  (persistent GPU fields, 18 pressure iterations)
```

Unchanged camera frames, foreground frames, and hand masks are not uploaded again. CPU overlays avoid frame copies when disabled. These optimizations preserve the existing OpenCV fallback and strip/style architecture; they only change scheduling, resolution allocation, and redundant work.

## Diagnostics

The `v` key cycles:

- normal composite
- dye
- velocity magnitude
- velocity direction
- pressure
- divergence
- vorticity
- source locations
- ink material
- fluid glass material
- chromatic material
- hand mask
- selected material output
- Riso density
- Riso halftone
- Riso registration
- paper texture
- final Riso output

GPU timing is disabled by default. When enabled, independent non-overlapping query rings measure camera upload, liquid simulation, liquid material, Riso treatment, final composition, and presentation. Results are read only when a ring slot is reused several frames later. GPU frame time is their rolling-average sum and is displayed separately from CPU frame time and display FPS.

## Implementation phases

1. **Completed:** original strip, tracking, styles, warping, and composition.
2. **Completed:** tracker-neutral interaction and graphics architecture.
3. **Completed:** ModernGL/GLFW backend and direct GPU presentation.
4. **Completed:** persistent liquid resource and pass architecture.
5. **Completed:** GPU advection, pressure projection, dye transport, and visualization.
6. **Completed:** delayed GPU timing, deterministic stress inputs, source stabilization, vorticity, closed walls, debug fields, and safe target recreation.
7. **Completed:** two-hand feathered occlusion, GPU foreground composition, simulation/material separation, ink, fluid glass, chromatic liquid, and palettes.
8. **Completed:** strict physical-camera selection, modular GPU print architecture, Risograph screening, physical ink overlap, paper treatment, stable registration, quality profiles, debug layers, and split timings.
9. **Completed:** native FHD camera negotiation, HD presentation, asynchronous latest-frame tracking, half-resolution mask processing, cached GPU uploads, 60 FPS pacing, and short source prediction for smoother motion.
10. **Completed:** physically inspired fluid-glass transmission, restrained dispersion, Fresnel reflection, curvature/caustic edges, smooth surface unions, solver-only optical motion, and cleaner low-diffusion flow tuning.
11. **Completed:** filled pinch-fluid material mode, convex pinch-volume refraction, motion-driven internal detail/trails, and influence-based source fade-out across interaction, simulation, and material layers.
12. **Completed:** name-aware camera discovery, NVIDIA Broadcast pre-capture filtering, background hot-plug polling, nonblocking validated switching, no-camera startup, and camera-coordinate tracking reset.
13. Next: cyanotype/cyanotone as a separate print treatment. Segmentation, stippling, particles, reaction diffusion, and new solvers remain deferred.

## Tests and benchmarks

```bash
python -m pytest
python -m compileall -q app tests run.py
python -m app.graphics.gpu_smoke
python -m app.graphics.gpu_smoke --visible --camera-index 0 --frames 600
python -m app.graphics.gpu_smoke --gpu-timing --stress --cycle-debug --cycle-materials --cycle-palettes --cycle-riso --frames 600
python -m app.graphics.gpu_benchmark --frames 240
python -m app.graphics.gpu_benchmark --frames 240 --pressure-sweep
```

Normal pytest never creates a GPU context. Stress mode cycles rapid two-hand motion, crossings, dropout, stationary/rapid combinations, extreme and tiny velocities, boundary sources, alternating directions, pinch changes, and changing hand distance. By default the GPU smoke test also uploads a deterministic, soft two-hand foreground mask; `--no-occlusion` disables that fixture. GPU smoke tests allocate every resource, compile every solver, material, Riso, and composite shader, and execute the complete graph. None reads rendered pixels back to the CPU.

On hybrid-GPU Windows laptops, GLFW/WGL uses the adapter selected by Windows and the display driver. ArtFrame reports the actual OpenGL vendor, renderer, and version. It contains no RTX-specific code and does not alter system GPU configuration.
