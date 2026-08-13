from __future__ import annotations

import argparse

from app.graphics.gpu_smoke import run_smoke_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ArtFrame liquid GPU profiles")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--pressure-sweep", action="store_true")
    args = parser.parse_args()

    profiles = [("320x180", 1.0 / 3.0, 20), ("480x270", 0.5, 20), ("640x360", 2.0 / 3.0, 20), ("960x540", 1.0, 20)]
    if args.pressure_sweep:
        profiles = [(f"480x270 p{iterations}", 0.5, iterations) for iterations in (10, 20, 30, 40)]

    print("profile, submission_fps, gpu_frame_ms, gpu_simulation_ms, gpu_material_ms, gpu_riso_ms, gpu_composition_ms")
    for name, scale, iterations in profiles:
        info, fps = run_smoke_test(
            args.frames,
            (args.width, args.height),
            visible=False,
            simulation_scale=scale,
            pressure_iterations=iterations,
            gpu_timing=True,
            stress=True,
        )
        print(
            f"{name}, {fps:.1f}, {info['gpu_frame_ms']}, "
            f"{info['gpu_simulation_ms']}, {info['gpu_material_ms']}, "
            f"{info['gpu_riso_ms']}, {info['gpu_composition_ms']}"
        )


if __name__ == "__main__":
    main()
