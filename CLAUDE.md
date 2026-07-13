# AA — Auto Annotation Tool(自動アノテーションツール)

## プロジェクト概要

HuggingFace 上の人混み(crowd)画像を対象に、事前学習済みモデルで自動アノテーションを行い、
人手レビューを経て高品質なデータセット(COCO / YOLO 形式)を出力するパイプライン。

最初のターゲット:**人混み画像における人物検出(bbox)+ 頭部点アノテーション(crowd counting 用)**

## アーキテクチャ

```
AA/
├── CLAUDE.md            # このファイル
├── ROADMAP.md           # フェーズ別ロードマップ(必読)
├── pyproject.toml       # 依存管理(uv を使用)
├── configs/             # パイプライン設定(YAML)
│   └── default.yaml
├── src/aa/
│   ├── fetch.py         # HuggingFace からのデータ取得
│   ├── annotate.py      # 自動アノテーション(検出モデル推論)
│   ├── review.py        # FiftyOne によるレビューセッション起動
│   ├── export.py        # COCO / YOLO 形式へのエクスポート
│   └── metrics.py       # GT があるデータでの精度評価
├── data/
│   ├── raw/             # 取得した元画像(git 管理外)
│   ├── annotations/     # 自動アノテーション結果(JSON)
│   └── exports/         # 最終出力データセット
└── .claude/skills/      # ワークフロー用スキル(下記)
```

## 技術スタック

- **Python 3.11+ / uv** — 環境・依存管理。`uv run` でスクリプト実行
- **huggingface_hub / datasets** — データ取得
- **ultralytics(YOLO11)** — 人物 bbox 検出のベースモデル。人混み特化には
  CrowdHuman でファインチューニングしたモデルに差し替え可能
- **FiftyOne** — アノテーション結果の可視化・人手レビュー・修正
- **pycocotools** — COCO 形式の入出力と評価(AP / MAE)

## データセット(HuggingFace、確認済み)

| ID | 用途 |
|----|------|
| `sshao0516/CrowdHuman` | 人物 bbox の GT あり。精度評価・ファインチューニング用 |
| `KTAEHWA/shanghaitech-crowd-counting` | 頭部点 GT あり。crowd counting 評価用 |
| `e8035669/crowdhuman_coco` | CrowdHuman の COCO 形式版 |

## アノテーション形式の規約

- 中間形式は **COCO JSON** に統一(`data/annotations/*.json`)
- 各 annotation に `score`(モデル信頼度)と `status`
  (`auto` / `reviewed` / `rejected`)フィールドを付与
- 人物 bbox はカテゴリ `person`(id=1)、頭部点はカテゴリ `head`(id=2、
  bbox は点中心の小さな正方形で表現)
- 信頼度しきい値:`score >= 0.5` を自動採用、`0.25–0.5` はレビュー対象、
  `< 0.25` は破棄(configs/default.yaml で変更可)

## ワークフロー(スキル)

| スキル | 役割 |
|--------|------|
| `/fetch-dataset` | HF からデータセット取得 → `data/raw/` に展開 |
| `/auto-annotate` | 検出モデルで推論 → COCO JSON 出力 |
| `/review-annotations` | FiftyOne でレビュー UI 起動、低信頼度サンプル優先表示 |
| `/export-dataset` | reviewed 済みアノテーションを COCO / YOLO 形式で出力 |

## 開発規約

- `data/` 配下は git 管理外(`.gitignore` 済み)。再現はスクリプト+設定で担保
- 新しい処理は必ず `configs/*.yaml` で設定可能にし、ハードコードしない
- GT のあるデータセットで変更前後の AP / MAE を比較してから採用する
- コミット前に `uv run pytest` を通す
