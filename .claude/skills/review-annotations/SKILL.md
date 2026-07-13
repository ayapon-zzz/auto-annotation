---
name: review-annotations
description: 自動アノテーション結果を FiftyOne UI にロードして人手レビューを行う。低信頼度・GT不一致サンプルを優先表示し、レビュー結果を COCO JSON に書き戻す。
---

# review-annotations

FiftyOne でアノテーションをレビューし、承認/修正/却下の結果を JSON に反映する。

## 手順

1. 引数: `[dataset_slug]`。省略時は `data/annotations/` の最新 `_auto.json` を対象。
2. `src/aa/review.py` が存在すれば実行、なければ実装してから実行:
   `uv run python -m aa.review --annotations data/annotations/<slug>_auto.json`
3. レビューセッションの構成:
   - 画像+予測 bbox(信頼度表示)+ GT(あれば別色)をロード
   - デフォルトのソート: 信頼度 0.25–0.5 の検出を含む画像を先頭に
   - GT がある場合は false positive / false negative が多い画像を優先
4. ユーザーがブラウザ UI でレビューする間、セッションはバックグラウンドで維持する
   (`run_in_background` で起動し、URL を伝える)
5. レビュー完了の合図を受けたら、FiftyOne のタグ/修正を読み取り:
   - 承認 → `status: "reviewed"`
   - 却下タグ → `status: "rejected"`
   - bbox 修正 → 座標を更新して `status: "reviewed"`
   結果を `data/annotations/<slug>_reviewed.json` に保存する。
6. 完了後に報告: レビュー枚数、承認/修正/却下の内訳、自動アノテーション採用率。

## 注意

- FiftyOne 初回起動は DB 初期化で時間がかかる
- レビュー中にユーザーを待つ場合、ポーリングせずユーザーの合図を待つ
- 採用率(修正不要だった検出の割合)は Phase 6 のモデル改善効果を測る KPI なので必ず記録する
