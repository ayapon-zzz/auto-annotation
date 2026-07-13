# auto-annotation

人混み画像向けの自動アノテーションパイプライン。HuggingFace 上のデータセットを取得し、
事前学習済み検出モデル(YOLO11)で人物 bbox を自動付与、人手レビューを経て
COCO / YOLO 形式のデータセットを出力する。

## パイプライン

```
fetch(HF取得) → annotate(YOLO11推論) → metrics(AP/MAE評価) → review(FiftyOne) → export(COCO/YOLO)
```

```bash
uv sync
uv run python -m aa.fetch --limit 100           # CrowdHuman サブセット取得 + GT 変換
uv run python -m aa.annotate                    # 自動アノテーション (信頼度3段階分類)
uv run python -m aa.metrics                     # GT との精度評価
uv run python -m aa.review                      # FiftyOne レビュー UI
uv run python -m aa.export                      # データセット出力
```

全量実行(24,388枚)の手順・実績・運用ノウハウは [ROADMAP.md](ROADMAP.md) を参照。
プロジェクト構成と規約は [CLAUDE.md](CLAUDE.md) を参照。

## 実績 (CrowdHuman 24,388枚)

- 検出 546,967 件(自動採用 292,342 / 要レビュー 145,315)
- AP@0.5 = 0.586 / カウント MAE = 11.62(GT 19,370枚で評価、yolo11m・imgsz 1280)

## データセットのライセンスに関する注意

本リポジトリはコードのみを含み、画像・アノテーションデータは含まない。
使用している [CrowdHuman](https://www.crowdhuman.org/) データセットは
**非商用・学術研究用途限定**のライセンスで提供されている。本ツールで生成した
アノテーションを配布する場合は、元データセットのライセンス条件に従うこと。

## ライセンス

コード部分は MIT License。
