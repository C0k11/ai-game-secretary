import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labels.jsonl")
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "florence_lora.jsonl"

    with labels_path.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            image = obj.get("image")
            ann = obj.get("annotations") or []
            if not image:
                continue

            rec = {
                "image": image,
                "annotations": ann,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(str(out_jsonl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
