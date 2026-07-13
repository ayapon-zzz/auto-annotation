---
name: export-dataset
description: レビュー済みアノテーションを COCO / YOLO 形式のデータセットとして data/exports/ に出力する。train/val 分割と README(データセットカード)生成、HF への push にも対応。
---

# export-dataset

最終アノテーションを学習にそのまま使える形式でエクスポートする。

## 手順

1. 引数: `[dataset_slug] [--format coco|yolo|both] [--include-auto] [--split 0.9] [--push <hf_repo>]`。
   デフォルトは `both`、`reviewed` のみ、train:val = 9:1。
2. `src/aa/export.py` が存在すれば実行、なければ実装してから実行:
   `uv run python -m aa.export --annotations data/annotations/<slug>_reviewed.json --format both`
3. 出力構成(`data/exports/<slug>_<日付>/`):
   - COCO: `annotations/instances_{train,val}.json` + `images/{train,val}/`
   - YOLO: `images/`, `labels/`(正規化 xywh txt)+ `dataset.yaml`
   - `README.md`: 由来データセット・ライセンス・アノテーション方法・統計を記載
4. 検証: エクスポート後に `pycocotools` で JSON をロードして整合性確認、
   YOLO 形式は画像とラベルファイルの対応が取れているか確認する。
5. `--push` 指定時のみ `huggingface_hub` でアップロードする。指定がなければ push しない。
6. 完了後に報告: 出力先パス、画像数・アノテーション数(train/val 別)、形式。

## 注意

- 元データセットのライセンス(CrowdHuman は研究用途限定)を README に必ず継承する
- `rejected` の annotation は除外、`--include-auto` 指定時のみ `auto` を含める
- HF への push は公開行為なので、リポジトリ名と公開/非公開をユーザーに確認してから行う
