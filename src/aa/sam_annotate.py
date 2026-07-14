"""GT bbox をプロンプトに SAM2 でインスタンスマスクを量産する。

大規模実行対応 (annotate.py と同じ設計):
  - 1 画像 1 行の JSONL に COCO RLE 形式で逐次追記 (中断しても再実行で続きから)
  - --max-images で分割実行し、外側ループでプロセスを再起動 (MPS メモリリーク対策)
  - 箱プロンプトは 24 個ずつに分割して MPS の OOM を回避
  - 全画像処理後 --finalize-only で segmentation 付き COCO JSON に変換
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from ultralytics import SAM

CATEGORIES = [{"id": 1, "name": "person"}]
# MPS メモリに収まる箱プロンプト数の上限: sam2.1_t は 64、sam2.1_b / SAM3 は 24
DEFAULT_BOX_CHUNK = 24


def load_gt_boxes(gt_path: Path) -> tuple[dict, dict]:
    """GT から画像情報と person bbox (iscrowd 除く) を読み込む。"""
    with open(gt_path) as f:
        gt = json.load(f)
    images = {im["file_name"]: im for im in gt["images"]}
    boxes: dict[str, list] = {}
    id_to_name = {im["id"]: im["file_name"] for im in gt["images"]}
    for a in gt["annotations"]:
        if a["category_id"] == 1 and not a.get("iscrowd"):
            boxes.setdefault(id_to_name[a["image_id"]], []).append(a["bbox"])
    return images, boxes


def run_inference(
    images_dir: Path,
    jsonl_path: Path,
    boxes_by_name: dict[str, list],
    weights: str,
    device: str,
    max_images: int | None,
    box_chunk: int = DEFAULT_BOX_CHUNK,
) -> None:
    done: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                done.add(json.loads(line)["file_name"])
    targets = sorted(n for n in boxes_by_name if n not in done)
    print(f"対象 {len(boxes_by_name)} 枚 / 処理済み {len(done)} / 残り {len(targets)}")
    if not targets:
        return
    if max_images:
        targets = targets[:max_images]

    model = SAM(weights)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as out:
        for i, name in enumerate(targets, start=1):
            path = images_dir / name
            xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes_by_name[name]]
            rles = []
            for j in range(0, len(xyxy), box_chunk):
                results = model(path, bboxes=xyxy[j:j + box_chunk], device=device, verbose=False)
                masks = results[0].masks.data.cpu().numpy().astype(np.uint8)
                for m in masks:
                    rle = mask_util.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("ascii")
                    rles.append(rle)
            out.write(json.dumps({"file_name": name, "boxes": boxes_by_name[name], "rles": rles}) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(targets)} 枚 (全体 {len(done) + i}/{len(boxes_by_name)})")
                if device == "mps":
                    import torch
                    torch.mps.empty_cache()


def finalize(jsonl_path: Path, images: dict, output_path: Path) -> None:
    """JSONL を segmentation 付き COCO JSON に変換する。"""
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1
    with open(jsonl_path) as f:
        for image_id, line in enumerate(f, start=1):
            rec = json.loads(line)
            info = images[rec["file_name"]]
            coco["images"].append({
                "id": image_id, "file_name": rec["file_name"],
                "width": info["width"], "height": info["height"],
            })
            for bbox, rle in zip(rec["boxes"], rec["rles"]):
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": float(mask_util.area(rle)),
                    "segmentation": rle,
                    "iscrowd": 0,
                    "status": "auto",
                })
                ann_id += 1
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"完了: {len(coco['images'])} 枚 / マスク {ann_id - 1} 件 -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=Path("data/annotations/crowdhuman_full_gt.json"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/crowdhuman_full/images"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/crowdhuman_full_masks.json"))
    parser.add_argument("--weights", default="sam2.1_b.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--box-chunk", type=int, default=DEFAULT_BOX_CHUNK)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    images, boxes = load_gt_boxes(args.gt)
    jsonl_path = args.output.with_suffix(".jsonl")
    if args.finalize_only:
        finalize(jsonl_path, images, args.output)
        return
    run_inference(args.images, jsonl_path, boxes, args.weights, args.device,
                  args.max_images, args.box_chunk)


if __name__ == "__main__":
    main()
