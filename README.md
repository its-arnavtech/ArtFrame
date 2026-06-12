# ArtFrame

`ArtFrame` is a modular Python scaffold for a real-time webcam AR effect. The intended final experience is a virtual graphic strip that appears between both hands and follows their movement, with artistic rendering styles such as risograph, cyanotype, and stippling.

This first phase focuses on clean architecture, interfaces, placeholder implementations, and basic tests. The current runtime also tracks individual fingertip points so the strip can expand, skew, and taper from finger geometry. The advanced visual effects and production-grade hand foreground compositing are intentionally left for later phases.

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

The app opens the default webcam and displays the live frame. If both hands are detected, a placeholder style canvas is warped onto a strip controlled by the left and right fingertip spreads. Debug dots show the tracked thumb, index, middle, ring, and pinky tips for each hand.

## Controls

- `1`: risograph style
- `2`: cyanotype style
- `3`: stippling style
- `space`: next style
- `q` or `ESC`: quit

## Architecture

- `app/camera`: webcam capture
- `app/tracking`: MediaPipe hand detection, fingertip extraction, anchor extraction, and smoothing
- `app/geometry`: finger-driven strip quad construction and perspective warping
- `app/styles`: style interface, placeholder renderers, and style registry
- `app/compositing`: overlay compositing
- `app/ui`: keyboard controls and HUD drawing
- `app/utils`: small utility helpers

The main loop in `app/main.py` delegates work to the modules above so later effects can be upgraded without turning the runtime into one large script.

## Implementation Phases

1. Scaffold architecture, data types, placeholder styles, and tests.
2. Improve fingertip/anchor robustness and temporal behavior.
3. Replace placeholder visual styles with richer risograph, cyanotype, and stippling effects.
4. Add better masking, occlusion, and hand foreground compositing.
5. Tune performance and polish the live AR experience.

## Tests

```bash
pytest
```
