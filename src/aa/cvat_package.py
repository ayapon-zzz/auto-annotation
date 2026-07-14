"""CVAT 手動アップロード方式 (認証情報レス) のパッケージ作成と取り込み。

pack:   マスク JSONL からバッチを切り出し、CVAT にドラッグ&ドロップできる
        「画像 zip + COCO アノテーション json」を data/cvat/<batch>/ に作る
import: CVAT の Export (COCO 1.0) でダウンロードした zip/json を取り込み、
        status=reviewed の COCO JSON として保存する

CVAT 側の操作:
  1. Tasks -> Create new task -> 名前入力、images.zip をドロップ -> Submit
  2. タスクの Actions -> Upload annotations -> COCO 1.0 -> annotations.json
  3. 修正作業 (ブラシ/消しゴム、AIツールで背景人物の追加)
  4. Actions -> Export task dataset -> COCO 1.0 -> ダウンロード
  5. このスクリプトの import にそのファイルを渡す
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

CATEGORIES = [{"id": 1, "name": "person"}]


def pack(jsonl_path: Path, images_dir: Path, done_dir: Path, batch_size: int) -> None:
    packed: set[str] = set()
    for prev in sorted(done_dir.glob("batch*/annotations.json")) if done_dir.exists() else []:
        with open(prev) as f:
            packed.update(im["file_name"] for im in json.load(f)["images"])

    records = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["file_name"] not in packed:
                records.append(rec)
            if len(records) >= batch_size:
                break
    if not records:
        raise SystemExit("パッケージ対象がありません (全て梱包済みか、生成が未完)")

    batch_no = 1 + (sum(1 for _ in done_dir.glob("batch*")) if done_dir.exists() else 0)
    batch_dir = done_dir / f"batch{batch_no}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1
    from pycocotools import mask as mask_util
    with zipfile.ZipFile(batch_dir / "images.zip", "w") as zf:
        for image_id, rec in enumerate(records, start=1):
            zf.write(images_dir / rec["file_name"], rec["file_name"])
            if "width" not in rec:  # bbox プロンプト版 JSONL は画像サイズを持たない
                from PIL import Image
                with Image.open(images_dir / rec["file_name"]) as im:
                    rec["width"], rec["height"] = im.size
            coco["images"].append({
                "id": image_id, "file_name": rec["file_name"],
                "width": rec["width"], "height": rec["height"],
            })
            for det in rec["detections"] if "detections" in rec else [
                {"rle": r, "score": None} for r in rec.get("rles", [])
            ]:
                rle = det["rle"]
                x, y, w, h = mask_util.toBbox(rle).tolist()
                coco["annotations"].append({
                    "id": ann_id, "image_id": image_id, "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": float(mask_util.area(rle)),
                    "segmentation": rle, "iscrowd": 0,
                })
                ann_id += 1
    with open(batch_dir / "annotations.json", "w") as f:
        json.dump(coco, f)
    print(f"バッチ {batch_no} を作成: {len(records)} 枚 / マスク {ann_id - 1} 件")
    print(f"  アップロード用: {batch_dir}/images.zip と {batch_dir}/annotations.json")
    print("  CVAT: タスク作成 -> images.zip / Upload annotations -> COCO 1.0 -> annotations.json")


def import_export(export_path: Path, output_path: Path) -> None:
    """CVAT の Export (COCO 1.0) を reviewed 付き COCO JSON として保存する。"""
    if export_path.suffix == ".zip":
        with zipfile.ZipFile(export_path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".json"))
            with zf.open(name) as f:
                coco = json.load(f)
    else:
        with open(export_path) as f:
            coco = json.load(f)

    for a in coco["annotations"]:
        a["status"] = "reviewed"
    person_ids = [c["id"] for c in coco.get("categories", []) if c["name"] == "person"]
    n_person = sum(1 for a in coco["annotations"] if a["category_id"] in person_ids)
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"取り込み完了: 画像 {len(coco['images'])} 枚 / person {n_person} 件 -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pack", "import"])
    parser.add_argument("--jsonl", type=Path, default=Path("data/annotations/crowdhuman_sam3_masks.jsonl"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/crowdhuman_full/images"))
    parser.add_argument("--cvat-dir", type=Path, default=Path("data/cvat"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--export", type=Path, help="import 時: CVAT からダウンロードした zip/json")
    parser.add_argument("--output", type=Path, default=Path("data/annotations/crowdhuman_sam3_reviewed.json"))
    args = parser.parse_args()

    if args.mode == "pack":
        pack(args.jsonl, args.images, args.cvat_dir, args.batch_size)
    else:
        if not args.export:
            raise SystemExit("--export に CVAT からダウンロードしたファイルを指定してください")
        import_export(args.export, args.output)


if __name__ == "__main__":
    main()
