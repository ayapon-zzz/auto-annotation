"""SAM3 のテキストプロンプト ("person") で人物マスクを少量バッチ生成する。

GT bbox を使わず画像中の全人物を検出+セグメントするため、
CrowdHuman の GT に含まれない背景の人物 (観客など) も対象になる。

運用: 少量 (--batch, 既定50枚) ずつ生成 → FiftyOne で人手修正 → 次のバッチ。
出力は COCO RLE の JSONL (逐次追記・再開可能)。--finalize-only で COCO JSON 化。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

CATEGORIES = [{"id": 1, "name": "person"}]
DEFAULT_WEIGHTS = (
    "~/.cache/huggingface/hub/models--facebook--sam3/snapshots/"
    "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
)


def run_inference(
    images_dir: Path,
    jsonl_path: Path,
    weights: str,
    device: str,
    imgsz: int,
    batch: int,
    conf_floor: float,
) -> None:
    from ultralytics.models.sam import SAM3SemanticPredictor

    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    done: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                done.add(json.loads(line)["file_name"])
    remaining = [p for p in images if p.name not in done]
    print(f"対象 {len(images)} 枚 / 処理済み {len(done)} / 残り {len(remaining)}")
    if not remaining:
        return
    targets = remaining[:batch]

    predictor = SAM3SemanticPredictor(
        overrides=dict(model=str(Path(weights).expanduser()), device=device,
                       imgsz=imgsz, verbose=False, save=False)
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as out:
        for i, img_path in enumerate(targets, start=1):
            results = predictor(source=str(img_path), text=["person"])
            r = results[0]
            with Image.open(img_path) as im:
                width, height = im.size
            dets = []
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy().astype(np.uint8)
                confs = r.boxes.conf.cpu().numpy()
                for m, c in zip(masks, confs):
                    if c < conf_floor:
                        continue
                    if m.shape != (height, width):
                        m = np.array(Image.fromarray(m).resize((width, height), Image.NEAREST))
                    rle = mask_util.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("ascii")
                    dets.append({"rle": rle, "score": round(float(c), 4)})
            out.write(json.dumps({
                "file_name": img_path.name,
                "width": width, "height": height,
                "detections": dets,
            }) + "\n")
            out.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(targets)} 枚 (全体 {len(done) + i}/{len(images)})")
                if device == "mps":
                    import torch
                    torch.mps.empty_cache()
    print(f"バッチ完了: {len(targets)} 枚 (通算 {len(done) + len(targets)}/{len(images)})")


def finalize(jsonl_path: Path, output_path: Path) -> None:
    """JSONL を segmentation 付き COCO JSON に変換する。bbox はマスクから導出。"""
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1
    with open(jsonl_path) as f:
        for image_id, line in enumerate(f, start=1):
            rec = json.loads(line)
            coco["images"].append({
                "id": image_id, "file_name": rec["file_name"],
                "width": rec["width"], "height": rec["height"],
            })
            for det in rec["detections"]:
                rle = det["rle"]
                x, y, w, h = mask_util.toBbox(rle).tolist()
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": float(mask_util.area(rle)),
                    "segmentation": rle,
                    "iscrowd": 0,
                    "score": det["score"],
                    "status": "auto",
                })
                ann_id += 1
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"完了: {len(coco['images'])} 枚 / マスク {ann_id - 1} 件 -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/raw/crowdhuman_full/images"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/sam3_masks.json"))
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=1008, help="M2/MPS では 1008 が上限 (1288 は OOM)")
    parser.add_argument("--batch", type=int, default=50, help="このバッチで処理する枚数")
    parser.add_argument("--conf-floor", type=float, default=0.25)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    jsonl_path = args.output.with_suffix(".jsonl")
    if args.finalize_only:
        finalize(jsonl_path, args.output)
        return
    run_inference(args.images, jsonl_path, args.weights, args.device,
                  args.imgsz, args.batch, args.conf_floor)


if __name__ == "__main__":
    main()
