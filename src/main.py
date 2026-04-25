from __future__ import annotations

import argparse
import json
from pathlib import Path

from texture_contrast import GLCMConfig, contrast_features, generate_demo_textures, load_grayscale


def _load_variant14_config(repo_root: Path) -> GLCMConfig:
    cfg_path = repo_root / "config" / "variant14.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return GLCMConfig(
        levels=int(data["levels"]),
        distance=int(data["distance"]),
        angles_deg=tuple(int(x) for x in data["angles_deg"]),
        symmetric=bool(data.get("symmetric", True)),
        normalized=bool(data.get("normalized", True)),
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Lab 8: Texture analysis (GLCM contrast).")
    parser.add_argument("--image", type=str, help="Path to input image.")
    parser.add_argument("--levels", type=int, default=None, help="Gray levels for quantization (2..256).")
    parser.add_argument("--distance", type=int, default=None, help="Pixel distance for GLCM.")
    parser.add_argument("--angles", type=str, default=None, help='Comma-separated angles in degrees, e.g. "0,45,90,135".')
    parser.add_argument("--variant", type=int, default=14, help="Variant number (default: 14).")
    parser.add_argument("--demo", action="store_true", help="Generate demo textures and compute features for them.")
    parser.add_argument("--json-out", type=str, default=None, help="Write results to JSON file.")
    args = parser.parse_args()

    if args.variant != 14:
        raise SystemExit("Only variant 14 is preconfigured in this repo. Use --levels/--distance/--angles to override.")

    cfg = _load_variant14_config(repo_root)
    if args.levels is not None:
        cfg = GLCMConfig(**{**cfg.__dict__, "levels": args.levels})
    if args.distance is not None:
        cfg = GLCMConfig(**{**cfg.__dict__, "distance": args.distance})
    if args.angles is not None:
        angles = tuple(int(x.strip()) for x in args.angles.split(",") if x.strip())
        cfg = GLCMConfig(**{**cfg.__dict__, "angles_deg": angles})

    results_all: dict[str, dict[str, float]] = {}

    if args.demo:
        paths = generate_demo_textures(repo_root / "data" / "generated")
        for p in paths:
            img = load_grayscale(p)
            results_all[p.name] = contrast_features(img, cfg)
    else:
        if not args.image:
            raise SystemExit("Provide --image PATH or use --demo.")
        img = load_grayscale(args.image)
        results_all[Path(args.image).name] = contrast_features(img, cfg)

    print(json.dumps({"config": cfg.__dict__, "results": results_all}, ensure_ascii=False, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"config": cfg.__dict__, "results": results_all}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
