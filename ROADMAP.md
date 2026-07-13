# ロードマップ — 自動アノテーションツール

目標:HuggingFace の人混み画像に対する自動アノテーション → 人手レビュー → データセット出力
までを一気通貫で回せるツールを作る。まず「人物 bbox 検出」で動くものを作り、
次に「頭部点(crowd counting)」へ拡張、最後に自己改善ループ(アクティブラーニング)を組む。

---

## Phase 0: 環境構築(0.5日)

- [x] `uv init` でプロジェクト初期化、Python 3.11+
- [x] 依存追加: `ultralytics`, `datasets`, `huggingface_hub`, `fiftyone`,
      `pycocotools`, `pyyaml`, `pillow`, `pytest`
- [x] ディレクトリ構成作成(CLAUDE.md 参照)、`.gitignore`(`data/`, `*.pt`, `fiftyone/`)
- [x] `configs/default.yaml` 作成(データセットID、モデル名、信頼度しきい値、出力先)
- [x] 動作確認: YOLO11 のサンプル推論が 1 枚の画像で通ること

**完了条件**: `uv run python -c "from ultralytics import YOLO"` が通り、サンプル画像で person 検出できる

## Phase 1: データ取得(0.5日)

- [x] `src/aa/fetch.py`: HF データセットをダウンロードし `data/raw/<dataset>/` に画像展開
  - `sshao0516/CrowdHuman` の val split(GT あり → 後の評価に使う)
  - まずは **100枚程度のサブセット** で開発を回す(`--limit` オプション)
- [x] GT がある場合は COCO JSON に変換して `data/annotations/<dataset>_gt.json` に保存
- [x] `/fetch-dataset` スキルから呼べることを確認

**完了条件**: `uv run python -m aa.fetch --dataset sshao0516/CrowdHuman --limit 100` で画像とGTが揃う

## Phase 2: 自動アノテーション MVP — 人物 bbox(1–2日)

- [x] `src/aa/annotate.py`: YOLO11x(または yolo11m)で person クラスのみ推論
  - 出力: COCO JSON(`score`, `status: "auto"` 付き)
  - 人混み対策: `conf=0.15` の低しきい値で拾い、後段でしきい値分類
  - 大画像対応: SAHI(スライス推論)をオプションで有効化 — 人混みの小さい人物の取りこぼし対策
- [x] `src/aa/metrics.py`: CrowdHuman GT に対して AP@0.5 / 人数カウント MAE を算出
- [x] ベースライン記録(CrowdHuman train サブセット100枚、GT は vbox 基準):

  | 設定 | AP@0.5 | AP@[.5:.95] | AR@1000 | カウントMAE |
  |------|--------|-------------|---------|-------------|
  | yolo11x, imgsz=640, GT=fbox | 0.216 | 0.076 | 0.164 | 10.06 |
  | yolo11x, imgsz=640, GT=vbox | 0.491 | 0.367 | 0.415 | 10.06 |
  | **yolo11x, imgsz=1280, GT=vbox** | **0.696** | **0.496** | **0.569** | **6.46** |
  | yolo11m, imgsz=1280, GT=vbox | 0.647 | 0.451 | 0.519 | 7.35 |

  速度 (M2, MPS): yolo11x 5.0s/枚, yolo11m 1.15s/枚。全量 24,388 枚の実行は
  m (7.7h) を採用 — x (33h) との精度差 AP -0.05 はレビューと Phase 6 の
  ファインチューニングで回収する方針。CPU 実行は MPS 比 4 倍遅いので必ず device=mps を指定。

  学び: ① GT は fbox でなく vbox(可視領域)を使う — COCO 学習済み検出器の出力定義と一致
  ② 人混みでは推論解像度が支配的(640→1280 で AP +0.2, MAE -3.6)
- [x] 全量実行 (2026-07-13): CrowdHuman 24,388 枚 (train+val+test) を yolo11m/MPS で約8時間で完走。
  検出 546,967 件 (自動採用 292,342 / 要レビュー 145,315 / 破棄 109,310)。
  GT 19,370 枚での評価: AP@0.5 0.586 / カウント MAE 11.62 (平均 GT 人数 22.7 — 100枚ベンチより高密度)。
  運用ノウハウ: 長時間実行は MPS のメモリリークで OS に kill されるため、
  2,000 枚ごとのプロセス再起動 + `torch.mps.empty_cache()` + JSONL 逐次保存による再開で対処。
- [ ] 改善イテレーション(効果を測って採用):
  1. NMS → Soft-NMS / agnostic NMS(重なりの多い人混みでの抑制ミス対策)
  2. スライス推論のタイルサイズ調整
  3. CrowdHuman ファインチューン済み公開重みへの差し替え

**完了条件**: 100枚に対し自動アノテーションが走り、AP@0.5 と MAE がレポートされる

## Phase 3: レビュー・修正フロー(1–2日)

- [ ] `src/aa/review.py`: FiftyOne に画像+予測+(あれば)GT をロードし UI 起動
  - 低信頼度(0.25–0.5)・GT との不一致が大きいサンプルを優先ソート
  - レビューで承認/修正/削除 → `status` を `reviewed` / `rejected` に更新
- [ ] レビュー結果を COCO JSON に書き戻す(`data/annotations/<dataset>_reviewed.json`)
- [ ] レビュー効率の指標を記録: 1枚あたりレビュー時間、自動アノテーション採用率

**完了条件**: FiftyOne 上で 20 枚レビューし、修正が JSON に反映される

## Phase 4: エクスポートと配布(0.5日)

- [ ] `src/aa/export.py`: `reviewed` のみ(または `auto` 含む)を COCO / YOLO txt 形式で出力
- [ ] train/val 分割オプション、データセットカード(README)自動生成
- [ ] オプション: `huggingface_hub` で自分の HF リポジトリへ push

**完了条件**: エクスポートした YOLO 形式データで `yolo train` がエラーなく開始できる

## Phase 5: 頭部点アノテーション(crowd counting 対応)(2–3日)

- [ ] 頭部検出モデルの導入(CrowdHuman の head bbox でファインチューニング、
      または公開の頭部検出重みを利用)
- [ ] bbox → 点(頭部中心)変換、`KTAEHWA/shanghaitech-crowd-counting` の GT で MAE / RMSE 評価
- [ ] 密度が非常に高い画像(数百人〜)向け: P2PNet 系の点予測モデルを検討
- [ ] FiftyOne レビューを点アノテーションに対応させる(キーポイント表示)

**完了条件**: ShanghaiTech で カウント MAE がベースライン(素の person 検出)より改善

## Phase 6: アクティブラーニング・自己改善ループ(3日〜)

- [ ] reviewed データでモデルをファインチューニング → 再アノテーション → 精度向上を確認
- [ ] 不確実性サンプリング: モデルが迷う画像を優先的にレビュー対象へ
- [ ] パイプライン全体を 1 コマンド化: fetch → annotate → (review) → export → finetune
- [ ] 人混み以外のドメイン(車両、動物など)への横展開はここで設計

**完了条件**: 「アノテーション→学習→再アノテーション」1周でレビュー必要枚数が減ることを実証

---

## マイルストーン早見表

| 時期 | 到達点 |
|------|--------|
| Week 1 | Phase 0–2: CrowdHuman 100枚で自動 bbox アノテーション+精度レポート |
| Week 2 | Phase 3–4: レビューUI+COCO/YOLOエクスポートで最初のデータセット完成 |
| Week 3 | Phase 5: 頭部点アノテーションで crowd counting 用データ生成 |
| Week 4+ | Phase 6: アクティブラーニングで半自動化ループ確立 |

## リスクと対策

- **人混みの重なり・遮蔽で検出漏れ** → SAHI スライス推論 + CrowdHuman ファインチューン重み
- **数百人規模の超高密度画像** → bbox 検出は破綻するため Phase 5 で点予測モデルへ切替
- **データセットのライセンス** → CrowdHuman は研究用途限定。エクスポート時に必ずライセンス表記を継承
- **FiftyOne の学習コスト** → まず標準UIのタグ付け機能だけで運用開始し、必要になったらプラグイン化
