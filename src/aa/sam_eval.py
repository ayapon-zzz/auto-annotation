"""SAM2 のマスク精度を定量評価する。

COCO val2017 の person について、GT bbox をプロンプトに SAM2 でマスクを生成し、
COCO のマスク GT との IoU を測る。「正しい箱を与えたときの SAM2 の上限性能」が分かる。
"""

import argparse

import fiftyone.zoo as foz
import numpy as np
from PIL import Image
from ultralytics import SAM


def gt_full_mask(det, w: int, h: int) -> np.ndarray | None:
    """fiftyone の Detection (相対bbox + パッチマスク) を全画面 bool マスクにする。"""
    if det.mask is None:
        return None
    x, y, bw, bh = det.bounding_box  # 相対座標
    x0, y0 = round(x * w), round(y * h)
    mask = det.mask
    full = np.zeros((h, w), dtype=bool)
    mh, mw = mask.shape
    full[y0:min(y0 + mh, h), x0:min(x0 + mw, w)] = mask[: h - y0, : w - x0]
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--weights", default="sam2.1_b.pt")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    load = lambda: foz.load_zoo_dataset(
        "coco-2017", split="validation",
        label_types=["segmentations"], classes=["person"],
        max_samples=args.samples, shuffle=True, seed=42,
    )
    dataset = load()
    if len(dataset) == 0:  # 過去の異常終了で空のデータセットが残っている場合は作り直す
        import fiftyone as fo
        fo.delete_dataset(dataset.name)
        dataset = load()
    print(f"評価対象: {len(dataset)} 枚")
    model = SAM(args.weights)

    ious: list[tuple[float, float]] = []  # (IoU, bbox面積比)
    for si, sample in enumerate(dataset, start=1):
        with Image.open(sample.filepath) as im:
            w, h = im.size
        persons = [d for d in sample.ground_truth.detections
                   if d.label == "person" and d.mask is not None
                   and not getattr(d, "iscrowd", 0)]
        if not persons:
            continue
        boxes = []
        for d in persons:
            x, y, bw, bh = d.bounding_box
            boxes.append([x * w, y * h, (x + bw) * w, (y + bh) * h])
        results = model(sample.filepath, bboxes=boxes, device=args.device, verbose=False)
        pred_masks = results[0].masks.data.cpu().numpy().astype(bool)
        for d, pm in zip(persons, pred_masks):
            gm = gt_full_mask(d, w, h)
            if gm is None or gm.sum() == 0:
                continue
            if pm.shape != gm.shape:  # SAM 出力が推論解像度の場合に合わせる
                pm = np.array(Image.fromarray(pm).resize((w, h), Image.NEAREST))
            inter = (pm & gm).sum()
            union = (pm | gm).sum()
            area_ratio = d.bounding_box[2] * d.bounding_box[3]
            ious.append((inter / union if union else 0.0, area_ratio))
        if si % 10 == 0:
            print(f"  {si}/{args.samples} 枚 (インスタンス {len(ious)})")

    arr = np.array([i for i, _ in ious])
    areas = np.array([a for _, a in ious])
    small, large = arr[areas < 0.02], arr[areas >= 0.02]
    print(f"\nSAM2 ({args.weights}) person マスク精度 — GT bbox プロンプト, {len(arr)} インスタンス")
    print(f"  平均 IoU   : {arr.mean():.3f} / 中央値 {np.median(arr):.3f}")
    print(f"  IoU < 0.7  : {(arr < 0.7).mean():.1%} (人手修正が必要になりうる割合)")
    print(f"  小さい人物 (bbox面積<2%): 平均 {small.mean():.3f} ({len(small)}件)")
    print(f"  大きい人物            : 平均 {large.mean():.3f} ({len(large)}件)")


if __name__ == "__main__":
    main()
