"""FiftyOne に画像・予測・GT をロードしてレビューセッションを起動する。

レビュー操作:
  - 予測 bbox を確認し、不要な検出には UI で "reject" タグを付ける
  - 画像単位で確認が済んだら画像に "done" タグを付ける
  - 保存は自動 (FiftyOne DB)。終了後 `--apply` で JSON に書き戻す
"""

import argparse
import json
from pathlib import Path

import fiftyone as fo

DATASET_NAME = "aa-review"


def load_coco(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_dataset(images_dir: Path, pred_path: Path, gt_path: Path | None) -> fo.Dataset:
    if fo.dataset_exists(DATASET_NAME):
        fo.delete_dataset(DATASET_NAME)
    dataset = fo.Dataset(DATASET_NAME)
    dataset.persistent = True

    pred = load_coco(pred_path)
    images = {im["id"]: im for im in pred["images"]}
    pred_by_image: dict[int, list] = {i: [] for i in images}
    for a in pred["annotations"]:
        pred_by_image[a["image_id"]].append(a)

    gt_by_image: dict[str, list] = {}
    if gt_path and gt_path.exists():
        gt = load_coco(gt_path)
        gt_images = {im["id"]: im["file_name"] for im in gt["images"]}
        for a in gt["annotations"]:
            if a["category_id"] == 1 and not a.get("iscrowd"):
                gt_by_image.setdefault(gt_images[a["image_id"]], []).append(a)

    samples = []
    for im in images.values():
        filepath = (images_dir / im["file_name"]).resolve()
        w, h = im["width"], im["height"]

        def to_detection(a: dict, with_score: bool) -> fo.Detection:
            x, y, bw, bh = a["bbox"]
            det = fo.Detection(
                label="person",
                bounding_box=[x / w, y / h, bw / w, bh / h],
            )
            if with_score:
                det.confidence = a.get("score")
                det["ann_id"] = a["id"]
                det["status"] = a.get("status", "auto")
            return det

        preds = pred_by_image[im["id"]]
        sample = fo.Sample(filepath=str(filepath))
        sample["predictions"] = fo.Detections(
            detections=[to_detection(a, True) for a in preds]
        )
        if gt_by_image:
            sample["ground_truth"] = fo.Detections(
                detections=[
                    to_detection(a, False) for a in gt_by_image.get(im["file_name"], [])
                ]
            )
        # 要レビュー検出が多い画像を先に見られるようソートキーを付与
        sample["n_needs_review"] = sum(1 for a in preds if a.get("status") == "needs_review")
        samples.append(sample)

    dataset.add_samples(samples)
    return dataset


def apply_review(pred_path: Path, output_path: Path) -> None:
    """FiftyOne 上のタグを読み取り、レビュー結果を COCO JSON に書き戻す。"""
    dataset = fo.load_dataset(DATASET_NAME)
    rejected_ids = set()
    reviewed_files = set()
    for sample in dataset:
        fname = Path(sample.filepath).name
        if "done" in (sample.tags or []):
            reviewed_files.add(fname)
        for det in sample["predictions"].detections:
            if "reject" in (det.tags or []):
                rejected_ids.add(det["ann_id"])

    pred = load_coco(pred_path)
    file_by_image = {im["id"]: im["file_name"] for im in pred["images"]}
    for a in pred["annotations"]:
        if a["id"] in rejected_ids:
            a["status"] = "rejected"
        elif file_by_image[a["image_id"]] in reviewed_files:
            a["status"] = "reviewed"

    with open(output_path, "w") as f:
        json.dump(pred, f)

    n_rej = sum(1 for a in pred["annotations"] if a["status"] == "rejected")
    n_rev = sum(1 for a in pred["annotations"] if a["status"] == "reviewed")
    print(f"書き戻し完了 -> {output_path}")
    print(f"  レビュー済み画像: {len(reviewed_files)} / 承認: {n_rev} / 却下: {n_rej}")
    if n_rev + n_rej:
        print(f"  自動アノテーション採用率: {n_rev / (n_rev + n_rej):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/raw/crowdhuman/images"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/crowdhuman_auto.json"))
    parser.add_argument("--gt", type=Path, default=Path("data/annotations/crowdhuman_gt.json"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/crowdhuman_reviewed.json"))
    parser.add_argument("--apply", action="store_true", help="レビュー結果を JSON に書き戻して終了")
    args = parser.parse_args()

    if args.apply:
        apply_review(args.annotations, args.output)
        return

    dataset = build_dataset(args.images, args.annotations, args.gt)
    view = dataset.sort_by("n_needs_review", reverse=True)
    session = fo.launch_app(view)
    print(f"レビューUI: {session.url}")
    print("要レビュー検出が多い画像から順に表示しています。")
    print('検出の却下は "reject" タグ、画像の確認完了は "done" タグを付けてください。')
    session.wait()


if __name__ == "__main__":
    main()
