"""CrowdHuman の GT bbox をプロンプトに SAM2 マスクを生成し、オーバーレイ画像を出力する。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import SAM


def overlay_masks(img: Image.Image, masks: np.ndarray) -> Image.Image:
    rng = np.random.default_rng(0)
    base = np.array(img.convert("RGB"), dtype=np.float32)
    for m in masks:
        if m.shape != base.shape[:2]:
            m = np.array(Image.fromarray(m).resize(img.size, Image.NEAREST))
        color = rng.integers(60, 255, 3)
        base[m] = base[m] * 0.45 + color * 0.55
    return Image.fromarray(base.astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=Path("data/annotations/crowdhuman_full_gt.json"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/crowdhuman_full/images"))
    parser.add_argument("--out", type=Path, default=Path("data/viz/sam"))
    parser.add_argument("--num", type=int, default=3, help="密度帯 (中央値/75%/最大) から選ぶ枚数")
    parser.add_argument("--weights", default="sam2.1_b.pt")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    images = {im["id"]: im for im in gt["images"]}
    boxes_by_image: dict[int, list] = {}
    for a in gt["annotations"]:
        if a["category_id"] == 1 and not a.get("iscrowd"):
            boxes_by_image.setdefault(a["image_id"], []).append(a["bbox"])

    ranked = sorted(boxes_by_image, key=lambda i: len(boxes_by_image[i]))
    n = len(ranked)
    picks = [ranked[n // 2], ranked[int(n * 0.75)], ranked[-1]][: args.num]

    model = SAM(args.weights)
    args.out.mkdir(parents=True, exist_ok=True)
    for iid in picks:
        info = images[iid]
        path = args.images / info["file_name"]
        xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes_by_image[iid]]
        # 箱プロンプトを一括で渡すと MPS がメモリ不足になるため分割する
        masks_list = []
        for i in range(0, len(xyxy), 24):
            results = model(path, bboxes=xyxy[i:i + 24], device=args.device, verbose=False)
            masks_list.append(results[0].masks.data.cpu().numpy().astype(bool))
        masks = np.concatenate(masks_list)
        with Image.open(path) as img:
            out_img = overlay_masks(img, masks)
        dest = args.out / f"{Path(info['file_name']).stem}_sam_n{len(xyxy)}.png"
        out_img.save(dest)
        print(f"{dest}: {len(xyxy)} 人分のマスク生成")


if __name__ == "__main__":
    main()
