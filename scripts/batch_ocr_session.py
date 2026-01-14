import argparse
import json
import sys
import time
from pathlib import Path

import requests


def _ocr_cache_path(repo_root: Path, rel: str, cache_version: int = 2) -> Path:
    import hashlib

    key = f"v{cache_version}:{rel}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    cache_dir = repo_root / "data" / "captures" / "_cache" / "ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}.json"


def _vlm_ocr_cache_path(repo_root: Path, rel: str, model: str, cache_version: int = 1) -> Path:
    import hashlib

    key = f"v{cache_version}:{model}:{rel}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    cache_dir = repo_root / "data" / "captures" / "_cache" / "vlm_ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}.json"


def _local_vlm_ocr_cache_path(repo_root: Path, rel: str, model: str, cache_version: int = 1) -> Path:
    import hashlib

    key = f"v{cache_version}:{model}:{rel}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    cache_dir = repo_root / "data" / "captures" / "_cache" / "local_vlm_ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="e.g. session_20260113_110034")
    parser.add_argument("--backend", default="http://127.0.0.1:8000", help="FastAPI base url")
    parser.add_argument("--engine", default="florence", choices=["florence", "vlm", "local_vlm"], help="OCR engine")
    parser.add_argument("--model", default="", help="VLM model tag (required when engine=vlm)")
    parser.add_argument("--model-path", dest="model_path", default="", help="Local VLM model repo_id or local dir (required when engine=local_vlm)")
    parser.add_argument("--base-url", dest="base_url", default="", help="Ollama base url (default http://127.0.0.1:11434)")
    parser.add_argument("--timeout-s", dest="timeout_s", type=int, default=1800, help="VLM read timeout seconds")
    parser.add_argument("--keep-alive", dest="keep_alive", default="10m", help="Ollama keep_alive (e.g. 10m)")
    parser.add_argument("--num-ctx", dest="num_ctx", type=int, default=8192, help="Ollama num_ctx")
    parser.add_argument("--num-predict", dest="num_predict", type=int, default=2048, help="Ollama num_predict")
    parser.add_argument("--num-gpu", dest="num_gpu", type=int, default=99, help="Ollama num_gpu (0=auto)")
    parser.add_argument("--force", action="store_true", help="force re-run OCR even if cache exists")
    parser.add_argument(
        "--out",
        default="",
        help="output jsonl path. default: data/captures/<session>/ocr_index.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0, help="0=all")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    session_dir = repo_root / "data" / "captures" / args.session
    if not session_dir.exists() or not session_dir.is_dir():
        print(f"session not found: {session_dir}", file=sys.stderr)
        return 2

    engine = (args.engine or "florence").strip().lower()
    model = (args.model or "").strip()
    model_path = (args.model_path or "").strip()
    base_url = (args.base_url or "").strip() or "http://127.0.0.1:11434"
    if engine == "vlm" and not model:
        print("--model is required when --engine=vlm", file=sys.stderr)
        return 2
    if engine == "local_vlm" and not model_path:
        print("--model-path is required when --engine=local_vlm", file=sys.stderr)
        return 2

    timeout_s = int(args.timeout_s) if args.timeout_s else 1800
    if timeout_s < 30:
        timeout_s = 30

    keep_alive = str(args.keep_alive or "10m")
    num_ctx = int(args.num_ctx) if args.num_ctx else 8192
    if num_ctx < 512:
        num_ctx = 512

    num_predict = int(args.num_predict) if args.num_predict else 2048
    if num_predict < 64:
        num_predict = 64

    num_gpu = int(args.num_gpu) if args.num_gpu is not None else 99
    if num_gpu < 0:
        num_gpu = 0

    if args.out:
        out_path = Path(args.out)
    else:
        if engine == "vlm":
            out_path = session_dir / "vlm_ocr_index.jsonl"
        elif engine == "local_vlm":
            out_path = session_dir / "local_vlm_ocr_index.jsonl"
        else:
            out_path = session_dir / "ocr_index.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(session_dir.glob("*.png"))
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    total = len(images)
    if total == 0:
        print("no images")
        return 0

    if engine == "vlm":
        url = args.backend.rstrip("/") + "/api/v1/vlm/ocr"
    elif engine == "local_vlm":
        url = args.backend.rstrip("/") + "/api/v1/local_vlm/ocr"
    else:
        url = args.backend.rstrip("/") + "/api/v1/vision/ocr"

    ok = 0
    cached_ok = 0
    skipped = 0
    failed = 0

    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for i, img_path in enumerate(images, start=1):
            rel = f"{args.session}/{img_path.name}".replace("\\", "/")

            if engine == "vlm":
                cache_path = _vlm_ocr_cache_path(repo_root, rel, model=model, cache_version=1)
            elif engine == "local_vlm":
                cache_path = _local_vlm_ocr_cache_path(repo_root, rel, model=model_path, cache_version=1)
            else:
                cache_path = _ocr_cache_path(repo_root, rel, cache_version=2)

            if (not args.force) and cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    items = cached.get("items") or []
                    if not isinstance(items, list):
                        items = []

                    rec = {"image": rel, "items": items}
                    if engine == "vlm":
                        rec["model"] = model
                    if engine == "local_vlm":
                        rec["model"] = model_path
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cached_ok += 1
                except Exception:
                    skipped += 1

                if i % 50 == 0 or i == total:
                    dt = time.time() - t0
                    print(
                        f"[{i}/{total}] cached_ok={cached_ok} skipped={skipped} ok={ok} failed={failed} elapsed={dt:.1f}s"
                    )
                continue

            try:
                if i == 1 or i % 10 == 0 or i == total:
                    dt = time.time() - t0
                    print(f"[{i}/{total}] request image={img_path.name} elapsed={dt:.1f}s")
                params = {"path": rel, "force": 1 if args.force else 0}
                if engine == "vlm":
                    params["model"] = model
                    params["base_url"] = base_url
                    params["timeout_s"] = timeout_s
                    params["keep_alive"] = keep_alive
                    params["num_ctx"] = num_ctx
                    params["num_predict"] = num_predict
                    params["num_gpu"] = num_gpu
                if engine == "local_vlm":
                    params["model"] = model_path

                req_timeout = timeout_s + 60 if engine in ("vlm", "local_vlm") else 600
                resp = requests.get(url, params=params, timeout=req_timeout)
                if resp.status_code != 200:
                    failed += 1
                    print(f"[{i}/{total}] HTTP {resp.status_code}: {resp.text[:200]}")
                    continue

                data = resp.json()
                items = data.get("items") or []

                rec = {"image": rel, "items": items}
                if engine == "vlm":
                    rec["model"] = model
                if engine == "local_vlm":
                    rec["model"] = model_path
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1

                if i % 10 == 0 or i == total:
                    dt = time.time() - t0
                    print(
                        f"[{i}/{total}] cached_ok={cached_ok} skipped={skipped} ok={ok} failed={failed} elapsed={dt:.1f}s"
                    )

            except Exception as e:
                failed += 1
                print(f"[{i}/{total}] error: {e}")

    dt = time.time() - t0
    print(
        f"done: out={out_path} cached_ok={cached_ok} skipped={skipped} ok={ok} failed={failed} elapsed={dt:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
