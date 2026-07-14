---
name: fetch-dataset
description: HuggingFace から人混み画像データセットを取得し data/raw/ に展開、GT があれば COCO JSON に変換する。引数はデータセットID(省略時 sshao0516/CrowdHuman)と枚数上限。
---

# fetch-dataset

HuggingFace のデータセットをローカルに取得し、パイプラインの入力形式に整える。

## 手順

1. 引数を解釈する: `<dataset_id> [--limit N] [--split train|val]`。
   省略時は `sshao0516/CrowdHuman`、`--limit 100`、`--split val`。
2. `src/aa/fetch.py` が存在すれば実行:
   `uv run python -m aa.fetch --dataset <dataset_id> --limit <N> --split <split>`
3. 存在しない場合(Phase 1 未完)は、`datasets.load_dataset` または
   `huggingface_hub.snapshot_download` を使うスクリプトをまず実装してから実行する。
   - 画像は `data/raw/<dataset名のスラッグ>/images/` に保存
   - GT アノテーションがあれば COCO JSON に変換して
     `data/annotations/<スラッグ>_gt.json` に保存
   - CrowdHuman の GT は odgt 形式(1行1画像の JSON Lines)。`fbox`(full box)を
     person bbox として、`hbox`(head box)を head として変換する
4. 完了後に報告: 取得枚数、保存先、GT の有無とアノテーション総数。

## 注意

- 全量ダウンロードは数十GBになりうる。`--limit` 未指定でユーザーの明示がなければ 100 枚に留める
- gated dataset でアクセス拒否された場合は `huggingface-cli login` が必要な旨をユーザーに伝える
  (`! huggingface-cli login` で実行してもらう)
- 既に取得済みのデータがあれば再ダウンロードせずスキップする
- コードの変更が必要になった場合は main に直接コミットせず、必ずブランチを切って変更し PR を作成する(CLAUDE.md の開発規約参照)
