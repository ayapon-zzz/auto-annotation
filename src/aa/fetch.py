"""HuggingFace から人混み画像データセットを取得し、COCO 形式の GT に変換する。

CrowdHuman は本家 (sshao0516/CrowdHuman) の zip が巨大なため、画像は
個別ファイルでホストされている jamarks/CrowdHuman-train から選択取得し、
アノテーション (odgt) のみ本家から取得して ID で突き合わせる。
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

ANNOTATION_REPO = "sshao0516/CrowdHuman"
IMAGE_REPO = "jamarks/CrowdHuman-train"

CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "head"},
]


def load_odgt(path: Path) -> dict[str, dict]:
    """odgt (JSON Lines) を画像 ID → レコードの dict にする。"""
    records = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["ID"]] = rec
    return records


def odgt_to_coco_annotations(rec: dict, image_id: int, start_ann_id: int) -> list[dict]:
    """1画像分の odgt gtboxes を COCO annotation のリストに変換する。

    vbox (可視領域) を person、hbox (頭部) を head として出力する。
    fbox (全身推定) は遮蔽部分を含み COCO 学習済み検出器の出力と定義がずれるため、
    vbox を優先し、無い場合のみ fbox にフォールバックする。
    tag が "mask" のもの (ignore 領域) は iscrowd=1 の person として残す。
    """
    anns = []
    ann_id = start_ann_id
    for gt in rec.get("gtboxes", []):
        is_ignore = gt.get("tag") != "person" or gt.get("extra", {}).get("ignore") == 1
        fbox = gt.get("vbox") or gt.get("fbox")
        if fbox:
            anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(v) for v in fbox],
                "area": float(fbox[2] * fbox[3]),
                "iscrowd": 1 if is_ignore else 0,
                "status": "gt",
            })
            ann_id += 1
        hbox = gt.get("hbox")
        if hbox and not is_ignore and gt.get("head_attr", {}).get("ignore") != 1:
            anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 2,
                "bbox": [float(v) for v in hbox],
                "area": float(hbox[2] * hbox[3]),
                "iscrowd": 0,
                "status": "gt",
            })
            ann_id += 1
    return anns


def fetch_crowdhuman(limit: int, out_dir: Path, ann_dir: Path) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("odgt アノテーションを取得中...")
    odgt_path = hf_hub_download(
        ANNOTATION_REPO, "annotation_train.odgt", repo_type="dataset"
    )
    records = load_odgt(Path(odgt_path))
    print(f"odgt レコード数: {len(records)}")

    print("画像リポジトリのファイル一覧を取得中...")
    repo_files = list_repo_files(IMAGE_REPO, repo_type="dataset")
    jpg_files = {Path(p).stem: p for p in repo_files if p.endswith(".jpg")}
    print(f"画像ファイル数: {len(jpg_files)}")

    # odgt と画像の両方が存在する ID を limit 件選ぶ(決定的になるようソート)
    common_ids = sorted(set(records) & set(jpg_files))[:limit]
    if not common_ids:
        raise SystemExit("odgt と画像リポジトリで一致する ID がありません")

    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1
    for image_id, img_key in enumerate(common_ids, start=1):
        local_name = f"{img_key}.jpg"
        dest = images_dir / local_name
        if not dest.exists():
            cached = hf_hub_download(
                IMAGE_REPO, jpg_files[img_key], repo_type="dataset"
            )
            dest.symlink_to(cached)
        with Image.open(dest) as im:
            width, height = im.size
        coco["images"].append({
            "id": image_id,
            "file_name": local_name,
            "width": width,
            "height": height,
        })
        anns = odgt_to_coco_annotations(records[img_key], image_id, ann_id)
        coco["annotations"].extend(anns)
        ann_id += len(anns)
        if image_id % 20 == 0:
            print(f"  {image_id}/{len(common_ids)} 枚取得済み")

    gt_path = ann_dir / "crowdhuman_gt.json"
    with open(gt_path, "w") as f:
        json.dump(coco, f)

    n_person = sum(1 for a in coco["annotations"] if a["category_id"] == 1)
    n_head = sum(1 for a in coco["annotations"] if a["category_id"] == 2)
    print(f"完了: 画像 {len(coco['images'])} 枚 -> {images_dir}")
    print(f"GT: person {n_person} / head {n_head} 件 -> {gt_path}")


ZIPS = {
    "train": ["CrowdHuman_train01.zip", "CrowdHuman_train02.zip", "CrowdHuman_train03.zip"],
    "val": ["CrowdHuman_val.zip"],
    "test": ["CrowdHuman_test.zip"],  # GT なし。自動アノテーション対象としてのみ使用
}
ODGT = {"train": "annotation_train.odgt", "val": "annotation_val.odgt"}


def fetch_crowdhuman_full(splits: list[str], out_dir: Path, ann_dir: Path) -> None:
    """zip を1つずつダウンロード→展開→削除してディスクピークを抑えつつ全量取得する。"""
    import zipfile

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    zips_dir = out_dir / "zips"

    for split in splits:
        for zname in ZIPS[split]:
            marker = out_dir / f".{zname}.done"
            if marker.exists():
                print(f"{zname}: 展開済み、スキップ")
                continue
            print(f"{zname}: ダウンロード中...")
            zpath = Path(hf_hub_download(
                ANNOTATION_REPO, zname, repo_type="dataset", local_dir=zips_dir
            ))
            print(f"{zname}: 展開中...")
            n = 0
            with zipfile.ZipFile(zpath) as zf:
                for info in zf.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".jpg"):
                        continue
                    dest = images_dir / Path(info.filename).name
                    if not dest.exists():
                        with zf.open(info) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
                    n += 1
            zpath.unlink()  # ディスク節約のため zip は即削除
            marker.touch()
            print(f"{zname}: {n} 枚展開、zip 削除済み")

    # GT 構築 (train/val のみ。展開済み画像と odgt の共通分)
    records: dict[str, dict] = {}
    for split in splits:
        if split not in ODGT:
            continue
        odgt_path = hf_hub_download(ANNOTATION_REPO, ODGT[split], repo_type="dataset")
        records.update(load_odgt(Path(odgt_path)))

    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1
    image_id = 0
    matched = sorted(k for k in records if (images_dir / f"{k}.jpg").exists())
    print(f"GT 変換中: {len(matched)} 枚 (画像サイズ読み取りに数分かかります)")
    for img_key in matched:
        image_id += 1
        with Image.open(images_dir / f"{img_key}.jpg") as im:
            width, height = im.size
        coco["images"].append({
            "id": image_id, "file_name": f"{img_key}.jpg",
            "width": width, "height": height,
        })
        anns = odgt_to_coco_annotations(records[img_key], image_id, ann_id)
        coco["annotations"].extend(anns)
        ann_id += len(anns)
        if image_id % 2000 == 0:
            print(f"  GT 変換 {image_id}/{len(matched)}")

    gt_path = ann_dir / "crowdhuman_full_gt.json"
    with open(gt_path, "w") as f:
        json.dump(coco, f)

    total_images = len(list(images_dir.glob("*.jpg")))
    n_person = sum(1 for a in coco["annotations"] if a["category_id"] == 1)
    n_head = sum(1 for a in coco["annotations"] if a["category_id"] == 2)
    print(f"完了: 画像 {total_images} 枚 -> {images_dir}")
    print(f"GT: {len(coco['images'])} 枚分 / person {n_person} / head {n_head} -> {gt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="crowdhuman", choices=["crowdhuman"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--full", action="store_true", help="zip 経由で全量取得")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    if args.full:
        fetch_crowdhuman_full(
            splits=args.splits,
            out_dir=args.data_dir / "raw" / "crowdhuman_full",
            ann_dir=args.data_dir / "annotations",
        )
    else:
        fetch_crowdhuman(
            limit=args.limit,
            out_dir=args.data_dir / "raw" / args.dataset,
            ann_dir=args.data_dir / "annotations",
        )


if __name__ == "__main__":
    main()
