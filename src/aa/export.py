"""レビュー済みアノテーションを COCO / YOLO 形式のデータセットとして出力する。"""

import argparse
import json
import random
import shutil
from datetime import date
from pathlib import Path

VALID_STATUS = {"reviewed", "gt"}


def load_coco(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def split_images(images: list[dict], train_ratio: float, seed: int = 0) -> dict[str, list[dict]]:
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return {"train": shuffled[:n_train], "val": shuffled[n_train:]}


def export_coco(coco: dict, splits: dict, images_dir: Path, out_dir: Path) -> None:
    anns_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    for split, images in splits.items():
        img_out = out_dir / "images" / split
        img_out.mkdir(parents=True, exist_ok=True)
        subset = {
            "images": images,
            "annotations": [a for im in images for a in anns_by_image.get(im["id"], [])],
            "categories": coco["categories"],
        }
        ann_out = out_dir / "annotations"
        ann_out.mkdir(parents=True, exist_ok=True)
        with open(ann_out / f"instances_{split}.json", "w") as f:
            json.dump(subset, f)
        for im in images:
            src = images_dir / im["file_name"]
            shutil.copy(src, img_out / im["file_name"])


def export_yolo(coco: dict, splits: dict, images_dir: Path, out_dir: Path) -> None:
    anns_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    for split, images in splits.items():
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for im in images:
            shutil.copy(images_dir / im["file_name"], img_out / im["file_name"])
            w, h = im["width"], im["height"]
            lines = []
            for a in anns_by_image.get(im["id"], []):
                x, y, bw, bh = a["bbox"]
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                cls = a["category_id"] - 1  # person=0, head=1
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
            stem = Path(im["file_name"]).stem
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    names = {c["id"] - 1: c["name"] for c in coco["categories"]}
    yaml_lines = [f"path: {out_dir.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {k}: {v}" for k, v in sorted(names.items())]
    (out_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")


def write_readme(out_dir: Path, n_images: dict, n_anns: dict, include_auto: bool) -> None:
    (out_dir / "README.md").write_text(f"""# CrowdHuman subset — auto-annotated

- 由来: CrowdHuman (https://www.crowdhuman.org/) — **研究用途限定ライセンス**
- 画像取得元: HuggingFace `jamarks/CrowdHuman-train` / GT: `sshao0516/CrowdHuman`
- アノテーション: YOLO11 による自動検出 + 人手レビュー ({'auto 含む' if include_auto else 'reviewed のみ'})
- 生成日: {date.today().isoformat()}
- 画像数: train {n_images['train']} / val {n_images['val']}
- アノテーション数: train {n_anns['train']} / val {n_anns['val']}
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/crowdhuman_reviewed.json"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/crowdhuman/images"))
    parser.add_argument("--format", choices=["coco", "yolo", "both"], default="both")
    parser.add_argument("--include-auto", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--name", default=None, help="出力ディレクトリ名")
    args = parser.parse_args()

    coco = load_coco(args.annotations)
    allowed = VALID_STATUS | ({"auto", "needs_review"} if args.include_auto else set())
    kept = [a for a in coco["annotations"] if a.get("status", "gt") in allowed]
    image_ids = {a["image_id"] for a in kept}
    images = [im for im in coco["images"] if im["id"] in image_ids]
    coco = {"images": images, "annotations": kept, "categories": coco["categories"]}

    if not images:
        raise SystemExit("出力対象がありません (reviewed が無い場合は --include-auto を検討)")

    name = args.name or f"crowdhuman_{date.today().isoformat()}"
    splits = split_images(images, args.train_ratio)

    anns_by_split = {}
    for split, ims in splits.items():
        ids = {im["id"] for im in ims}
        anns_by_split[split] = sum(1 for a in kept if a["image_id"] in ids)

    for fmt in (["coco", "yolo"] if args.format == "both" else [args.format]):
        out_dir = Path("data/exports") / name / fmt
        if fmt == "coco":
            export_coco(coco, splits, args.images, out_dir)
        else:
            export_yolo(coco, splits, args.images, out_dir)
        write_readme(out_dir, {k: len(v) for k, v in splits.items()}, anns_by_split, args.include_auto)
        print(f"{fmt.upper()} 形式を出力: {out_dir}")

    print(f"画像: train {len(splits['train'])} / val {len(splits['val'])}")
    print(f"アノテーション: train {anns_by_split['train']} / val {anns_by_split['val']}")


if __name__ == "__main__":
    main()
