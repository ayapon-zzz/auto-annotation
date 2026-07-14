"""FiftyOne 経由で CVAT にアノテーションバッチを送り、修正結果を取り込む。

前提: ローカル CVAT (http://localhost:8080) が起動していること。
認証は環境変数 FIFTYONE_CVAT_USERNAME / FIFTYONE_CVAT_PASSWORD、
または --username / --password で渡す。

push: マスク付き COCO JSON から画像バッチを CVAT タスクとして作成
      (SAM3 の生成マスクが編集可能な状態でプリロードされる)
pull: CVAT での修正・追加を取り込み、reviewed 付き COCO JSON に保存
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import fiftyone as fo

from aa.review import build_dataset

DATASET_NAME = "aa-cvat"
KEYCHAIN_SERVICE = "app.cvat.ai"


def keychain_credentials() -> tuple[str | None, str | None]:
    """macOS キーチェーンから CVAT の認証情報を取得する (平文ファイルを避ける)。"""
    try:
        meta = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE],
            capture_output=True, text=True, check=True,
        )
        username = None
        for line in (meta.stdout + meta.stderr).splitlines():
            if '"acct"' in line and '="' in line:
                username = line.rsplit('="', 1)[-1].rstrip('"')
        password = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return username, password
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def push(images_dir: Path, pred_path: Path, batch: int, anno_key: str,
         url: str, username: str | None, password: str | None) -> None:
    dataset = build_dataset(images_dir, pred_path, gt_path=None)
    # 未送信の先頭 batch 枚を対象にする
    view = dataset.sort_by("filepath").limit(batch)
    kwargs = {}
    if username:
        kwargs.update(username=username, password=password)
    view.annotate(
        anno_key,
        backend="cvat",
        url=url,
        label_field="predictions",
        label_type="instances",  # マスク付き検出として送る
        classes=["person"],
        segment_size=25,
        task_name=f"aa-{anno_key}",
        **kwargs,
    )
    print(f"CVAT にタスク作成: {len(view)} 枚 (anno_key={anno_key})")
    print(f"ブラウザで {url} を開いて修正してください。")


def pull(pred_path: Path, output_path: Path, anno_key: str,
         username: str | None, password: str | None) -> None:
    dataset = fo.load_dataset(DATASET_NAME)
    kwargs = {}
    if username:
        kwargs.update(username=username, password=password)
    dataset.load_annotations(anno_key, **kwargs)

    # 修正済みマスクを COCO JSON に書き出す (RLE 化)
    import numpy as np
    from pycocotools import mask as mask_util

    coco = {"images": [], "annotations": [],
            "categories": [{"id": 1, "name": "person"}]}
    ann_id = 1
    view = dataset.match(fo.ViewField("predictions") != None)  # noqa: E711
    for image_id, sample in enumerate(view, start=1):
        meta = sample.metadata or fo.ImageMetadata.build_for(sample.filepath)
        w, h = meta.width, meta.height
        coco["images"].append({
            "id": image_id, "file_name": Path(sample.filepath).name,
            "width": w, "height": h,
        })
        for det in sample["predictions"].detections:
            if det.mask is None:
                continue
            x, y, bw, bh = det.bounding_box
            full = np.zeros((h, w), dtype=np.uint8)
            x0, y0 = round(x * w), round(y * h)
            mh, mw = det.mask.shape
            full[y0:min(y0 + mh, h), x0:min(x0 + mw, w)] = \
                det.mask[: h - y0, : w - x0]
            rle = mask_util.encode(np.asfortranarray(full))
            rle["counts"] = rle["counts"].decode("ascii")
            bx, by, bw_, bh_ = mask_util.toBbox(rle).tolist()
            coco["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": 1,
                "bbox": [bx, by, bw_, bh_],
                "area": float(mask_util.area(rle)),
                "segmentation": rle, "iscrowd": 0,
                "status": "reviewed",
            })
            ann_id += 1
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"取り込み完了: {len(coco['images'])} 枚 / {ann_id - 1} マスク -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["push", "pull"])
    parser.add_argument("--images", type=Path, default=Path("data/raw/crowdhuman_full/images"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/crowdhuman_sam3_masks.json"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/crowdhuman_sam3_reviewed.json"))
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--anno-key", default="batch1")
    parser.add_argument("--url", default=os.environ.get("FIFTYONE_CVAT_URL", "https://app.cvat.ai"))
    parser.add_argument("--username", default=os.environ.get("FIFTYONE_CVAT_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("FIFTYONE_CVAT_PASSWORD"))
    args = parser.parse_args()

    if not args.username or not args.password:
        ku, kp = keychain_credentials()
        args.username = args.username or ku
        args.password = args.password or kp
        if not args.password:
            raise SystemExit(
                "認証情報がありません。キーチェーンに登録してください:\n"
                f"  security add-generic-password -s {KEYCHAIN_SERVICE} -a <ユーザー名> -w"
            )

    if args.mode == "push":
        push(args.images, args.annotations, args.batch, args.anno_key,
             args.url, args.username, args.password)
    else:
        pull(args.annotations, args.output, args.anno_key,
             args.username, args.password)


if __name__ == "__main__":
    main()
