"""検出モデル (YOLO11) で画像を推論し、信頼度付き COCO JSON を出力する。

大規模データ対応:
  - 推論結果は 1 画像 1 行の JSONL に逐次追記する(中断しても再実行で続きから)
  - 全画像の推論後、しきい値分類を適用して COCO JSON に変換する
"""

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO

CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "head"},
]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def classify_status(score: float, accept: float, review: float) -> str:
    if score >= accept:
        return "auto"
    if score >= review:
        return "needs_review"
    return "discard"


def run_inference(
    input_dir: Path,
    jsonl_path: Path,
    weights: str,
    conf_floor: float,
    imgsz: int,
    device: str | None = None,
    max_images: int | None = None,
) -> None:
    """未処理の画像だけ推論し、結果を JSONL に追記する。"""
    images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"画像が見つかりません: {input_dir}")

    done: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                done.add(json.loads(line)["file_name"])
    remaining = [p for p in images if p.name not in done]
    print(f"対象 {len(images)} 枚 / 処理済み {len(done)} / 残り {len(remaining)}")
    if not remaining:
        return
    if max_images:
        # 長時間実行でのメモリリーク対策: 一定枚数でプロセスを終了させ、外側のループで再起動する
        remaining = remaining[:max_images]

    model = YOLO(weights)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as out:
        for i, img_path in enumerate(remaining, start=1):
            results = model.predict(
                img_path, conf=conf_floor, classes=[0], max_det=1000,
                imgsz=imgsz, device=device, verbose=False,
            )
            r = results[0]
            h, w = r.orig_shape
            detections = []
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": round(float(box.conf[0]), 4),
                })
            out.write(json.dumps({
                "file_name": img_path.name,
                "width": w, "height": h,
                "detections": detections,
            }) + "\n")
            out.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(remaining)} 枚推論済み (全体 {len(done) + i}/{len(images)})")
                if device == "mps":
                    import torch
                    torch.mps.empty_cache()


def finalize(jsonl_path: Path, output_path: Path, accept: float, review: float) -> None:
    """JSONL をしきい値分類付きの COCO JSON に変換する。"""
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    counts = {"auto": 0, "needs_review": 0, "discard": 0}
    ann_id = 1
    with open(jsonl_path) as f:
        for image_id, line in enumerate(f, start=1):
            rec = json.loads(line)
            coco["images"].append({
                "id": image_id,
                "file_name": rec["file_name"],
                "width": rec["width"],
                "height": rec["height"],
            })
            for det in rec["detections"]:
                status = classify_status(det["score"], accept, review)
                counts[status] += 1
                if status == "discard":
                    continue
                x, y, bw, bh = det["bbox"]
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": det["bbox"],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "score": det["score"],
                    "status": status,
                })
                ann_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f)

    total = sum(counts.values())
    print(f"完了: {len(coco['images'])} 枚 / 検出 {total} 件 -> {output_path}")
    print(f"  自動採用: {counts['auto']} / 要レビュー: {counts['needs_review']} / 破棄: {counts['discard']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/crowdhuman/images"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/crowdhuman_auto.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--weights", help="設定ファイルのモデルを上書き")
    parser.add_argument("--finalize-only", action="store_true", help="推論せず JSONL から COCO を再生成")
    parser.add_argument("--max-images", type=int, help="この実行で処理する最大枚数 (メモリリーク対策の分割実行用)")
    parser.add_argument("--no-finalize", action="store_true", help="推論のみ行い COCO 変換をスキップ")
    args = parser.parse_args()

    cfg = load_config(args.config)
    jsonl_path = args.output.with_suffix(".jsonl")
    if not args.finalize_only:
        run_inference(
            input_dir=args.input,
            jsonl_path=jsonl_path,
            weights=args.weights or cfg["model"]["weights"],
            conf_floor=cfg["model"]["conf_floor"],
            imgsz=cfg["model"].get("imgsz", 1280),
            device=cfg["model"].get("device"),
            max_images=args.max_images,
        )
    if args.no_finalize:
        return
    finalize(
        jsonl_path=jsonl_path,
        output_path=args.output,
        accept=cfg["thresholds"]["accept"],
        review=cfg["thresholds"]["review"],
    )


if __name__ == "__main__":
    main()
