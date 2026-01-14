import argparse
import json

from vision.florence_vision import FlorenceConfig, FlorenceVision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--od", action="append", default=[])
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None, help="e.g. cuda, cpu")
    parser.add_argument("--dtype", default=None, help="fp16|bf16|fp32")
    args = parser.parse_args()

    cfg = FlorenceConfig()
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    if args.model_id:
        cfg.model_id = args.model_id
    if args.device:
        cfg.device = args.device
    if args.dtype:
        dt = args.dtype.lower().strip()
        if dt == "fp16":
            import torch

            cfg.dtype = torch.float16
        elif dt == "bf16":
            import torch

            cfg.dtype = torch.bfloat16
        elif dt == "fp32":
            import torch

            cfg.dtype = torch.float32
        else:
            raise ValueError("--dtype must be one of: fp16, bf16, fp32")

    vision = FlorenceVision(cfg)
    items = vision.analyze_screen(
        screenshot_path=args.image,
        od_queries=args.od or None,
        enable_ocr=not args.no_ocr,
    )
    print(json.dumps(items, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
