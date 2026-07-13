"""自動アノテーション結果を GT (COCO JSON) と比較し AP / カウント MAE を算出する。"""

import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def count_mae(gt: dict, detections: list[dict], category_id: int, min_score: float) -> tuple[float, float]:
    """画像ごとの人数カウントの MAE と平均 GT 人数を返す。iscrowd(ignore 領域)は除外。

    detections は GT の image_id に対応付け済みの検出リスト。
    """
    gt_counts = defaultdict(int)
    for a in gt["annotations"]:
        if a["category_id"] == category_id and not a.get("iscrowd"):
            gt_counts[a["image_id"]] += 1
    pred_counts = defaultdict(int)
    for a in detections:
        if a["category_id"] == category_id and a.get("score", 1.0) >= min_score:
            pred_counts[a["image_id"]] += 1
    image_ids = [im["id"] for im in gt["images"]]
    errors = [abs(gt_counts[i] - pred_counts[i]) for i in image_ids]
    mean_gt = sum(gt_counts[i] for i in image_ids) / len(image_ids)
    return sum(errors) / len(errors), mean_gt


def evaluate(gt_path: Path, pred_path: Path, min_score: float) -> None:
    with open(gt_path) as f:
        gt = json.load(f)
    with open(pred_path) as f:
        pred = json.load(f)

    # GT と予測で画像集合が異なる場合があるため、file_name で image_id を対応付ける
    gt_id_by_name = {im["file_name"]: im["id"] for im in gt["images"]}
    pred_name_by_id = {im["id"]: im["file_name"] for im in pred["images"]}

    # pycocotools は予測をフラットな検出リストで受け取る
    detections = []
    for a in pred["annotations"]:
        gt_image_id = gt_id_by_name.get(pred_name_by_id[a["image_id"]])
        if gt_image_id is None:
            continue  # GT が無い画像 (test split など) は評価対象外
        detections.append({
            "image_id": gt_image_id,
            "category_id": a["category_id"],
            "bbox": a["bbox"],
            "score": a.get("score", 1.0),
        })
    n_skipped = len({i for i in pred_name_by_id.values()} - set(gt_id_by_name))
    if n_skipped:
        print(f"注: GT の無い {n_skipped} 枚は評価から除外")

    coco_gt = COCO()
    coco_gt.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.catIds = [1]  # person のみ
        ev.params.maxDets = [1, 100, 1000]  # 人混み対応で上限拡大
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    ap = ev.stats[0]      # AP@[0.5:0.95]
    ap50 = ev.stats[1]    # AP@0.5
    ar = ev.stats[8]      # AR@1000
    mae, mean_gt = count_mae(gt, detections, category_id=1, min_score=min_score)

    print(f"person 検出精度 (GT: {gt_path.name}, 予測: {pred_path.name})")
    print(f"  AP@0.5       : {ap50:.3f}")
    print(f"  AP@[.5:.95]  : {ap:.3f}")
    print(f"  AR@1000      : {ar:.3f}")
    print(f"  カウント MAE : {mae:.2f} (score >= {min_score}, 平均GT人数 {mean_gt:.1f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=Path("data/annotations/crowdhuman_gt.json"))
    parser.add_argument("--pred", type=Path, default=Path("data/annotations/crowdhuman_auto.json"))
    parser.add_argument("--min-score", type=float, default=0.5, help="カウント時の信頼度しきい値")
    args = parser.parse_args()
    evaluate(args.gt, args.pred, args.min_score)


if __name__ == "__main__":
    main()
