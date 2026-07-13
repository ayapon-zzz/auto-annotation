---
name: auto-annotate
description: data/raw/ の画像に対して検出モデル(YOLO11)で自動アノテーションを実行し、COCO JSON を出力。GT があれば AP/MAE も算出する。人混み画像の人物 bbox・頭部検出用。
---

# auto-annotate

事前学習済み検出モデルで画像を推論し、信頼度付き COCO JSON アノテーションを生成する。

## 手順

1. 引数を解釈する: `[dataset_slug] [--model yolo11x.pt] [--sahi] [--conf-floor 0.15]`。
   dataset_slug 省略時は `data/raw/` 配下で最新のディレクトリを対象とする。
2. `src/aa/annotate.py` が存在すれば実行:
   `uv run python -m aa.annotate --input data/raw/<slug>/images --output data/annotations/<slug>_auto.json`
   未実装なら CLAUDE.md の規約に従って実装してから実行する。
3. 推論の要点:
   - person クラスのみ抽出(COCO class 0)
   - 検出信頼度は低め(0.15)で拾い、JSON に `score` を保存。しきい値分類は後段で行う
   - `--sahi` 指定時はスライス推論(タイル 640px、オーバーラップ 20%)で小さい人物を拾う
   - 各 annotation に `status: "auto"` を付与
4. `data/annotations/<slug>_gt.json` が存在する場合は `src/aa/metrics.py` で評価:
   - AP@0.5(bbox 検出精度)
   - カウント MAE(1画像あたりの人数誤差)— crowd counting 用途の主要指標
5. 完了後に報告: 処理枚数、検出総数、信頼度帯ごとの内訳(採用 >=0.5 / 要レビュー 0.25–0.5 / 破棄 <0.25)、GT があれば AP と MAE。

## 注意

- しきい値は `configs/default.yaml` を優先し、引数指定があればそちらを使う
- 初回はモデル重みのダウンロードが走る(yolo11x で約110MB)
- 評価スコアが前回より悪化した場合は、その旨を明示して原因候補を挙げる
