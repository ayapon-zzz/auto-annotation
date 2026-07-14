"""アノテーション結果を画像に描画して確認用 PNG を出力する。

信頼度による色分け: 緑 = 自動採用 (>=0.5) / 橙 = 要レビュー (0.25-0.5)。
--density で人数帯ごとのサンプルを選ぶ。
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

STATUS_COLORS = {"auto": (0, 200, 80), "needs_review": (255, 140, 0)}


def draw(images_dir: Path, out_dir: Path, coco: dict, image_ids: list[int]) -> list[Path]:
    images = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for iid in image_ids:
        im_info = images[iid]
        img = Image.open(images_dir / im_info["file_name"]).convert("RGB")
        d = ImageDraw.Draw(img)
        anns = anns_by_image.get(iid, [])
        for a in anns:
            x, y, w, h = a["bbox"]
            color = STATUS_COLORS.get(a.get("status", "auto"), (0, 200, 80))
            d.rectangle([x, y, x + w, y + h], outline=color, width=3)
            d.text((x + 2, y + 2), f"{a.get('score', 0):.2f}", fill=color)
        n_auto = sum(1 for a in anns if a.get("status") == "auto")
        n_rev = len(anns) - n_auto
        dest = out_dir / f"{Path(im_info['file_name']).stem}_n{len(anns)}.png"
        img.save(dest)
        paths.append(dest)
        print(f"{dest.name}: 検出 {len(anns)} (採用 {n_auto} / 要レビュー {n_rev})")
    return paths


def pick_by_density(coco: dict, per_bin: int = 2) -> list[int]:
    """低密度・中密度・高密度から per_bin 枚ずつ選ぶ。"""
    counts = {im["id"]: 0 for im in coco["images"]}
    for a in coco["annotations"]:
        counts[a["image_id"]] += 1
    ranked = sorted(counts, key=counts.get)
    n = len(ranked)
    picks = []
    for lo, hi in [(0.1, 0.15), (0.5, 0.55), (0.97, 1.0)]:  # 低 / 中 / 高密度帯
        band = ranked[int(n * lo):int(n * hi)]
        picks.extend(band[:per_bin])
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/crowdhuman_full_auto.json"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/crowdhuman_full/images"))
    parser.add_argument("--out", type=Path, default=Path("data/viz"))
    parser.add_argument("--per-bin", type=int, default=2)
    args = parser.parse_args()

    with open(args.annotations) as f:
        coco = json.load(f)
    ids = pick_by_density(coco, args.per_bin)
    draw(args.images, args.out, coco, ids)


if __name__ == "__main__":
    main()
