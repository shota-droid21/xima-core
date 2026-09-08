# xima-core API Reference

このドキュメントは xima-core の公開 HTTP API を網羅します。主要な用途はワークスペース／実験の管理、ラベル操作、画像取得、ジョブ実行（スクリプト呼び出し）、およびヘルスチェックです。

ベース URL: `http://{host}:{port}`（`scripts/run-local.sh` の既定は `127.0.0.1:27800`。任意の docker-compose 構成では `127.0.0.1:27801`）

認証: 既定の `local` モードでは、`/health` と `/local/session` を除くすべての経路に
`X-Xima-Local-Key` ヘッダが必要です（`README.md` の「認証モード」を参照）。
以下の例は簡潔さのためヘッダを省略しています。`XIMA_AGENT_AUTH_MODE=open` ではヘッダ不要です。

共通のパス規約:

- ワークスペース名および実験名はサニタイズされ、パストラバーサルを防止します。
- ラベルの正本はワークスペース配下の `label_input/labels.json` です。

## ヘルスチェック

- GET `/health`
  - 説明: サーバが稼働しているかの確認。バージョン情報を含む簡易レスポンスを返します。
  - 例:

```bash
curl http://127.0.0.1:27801/health
```

## ワークスペース管理

- GET `/workspaces`
  - 説明: 既存ワークスペース一覧を返します。
  - クエリ:
    - `order=sidebar` を指定すると、サイドバー用に保存された順序（後述の `/workspaces/sidebar-order`）を優先して返します。未登録の workspace は末尾に ID 昇順で補完されます。

  - レスポンス例:
    - `{ "workspaces": [{ "id": "ws_test", "display_name": "作業用", "workspace_root": "..." }, ...] }`

- POST `/workspaces`
  - 説明: 新しいワークスペースを作成します。成功時は作成フラグとルートパスを返します。
  - ボディ: JSON `{ "display_name": "...", "id": "..." }`
    - `id` は省略可能（省略時はサーバが短い ID を生成）。
    - `id` を指定する場合は `[a-z0-9]{8}`（base36 8 文字）を推奨します。

- GET `/workspaces/{workspace}`
  - 説明: 指定ワークスペースのメタ情報（exists, paths）を返します。

- PATCH `/workspaces/{workspace}`
  - 説明: ワークスペースの `display_name` を変更します。
  - ボディ: JSON `{ "display_name": "..." }`

- GET `/workspaces/{workspace}/backup`
  - 説明: 指定 workspace を `tar.zst` 形式でダウンロードします（互換API）。
  - 形式: `workspace.json` を先頭に含み、`source_images/` と `experiments/` を同梱します。

- POST `/workspaces/{workspace}/backup-jobs`
  - 説明: workspace backup 作成ジョブを開始します。
  - 補足:
    - 一時データは `workspaces/.tmp/backups` 配下に保存されます。
    - ジョブメタは `workspaces/.xima/jobs/workspace_backup` 配下に永続保存されます。
    - ジョブ開始時に、**24時間より古い backup 成果物**は自動削除されます。
  - レスポンス例: `{ "job_id": "...", "status": "queued", "workspace": "..." }`

- GET `/workspaces/{workspace}/backup-jobs`
  - 説明: workspace backup ジョブ一覧を返します（新しい順）。

- GET `/workspaces/{workspace}/backup-jobs/{job_id}`
  - 説明: 指定ジョブの詳細（`status`, `progress`, `filename`, `size_bytes` など）を返します。

- GET `/workspaces/{workspace}/backup-jobs/{job_id}/download`
  - 説明: 指定ジョブで作成された `tar.zst` をダウンロードします（`status=done` のみ）。
  - 補足:
    - 成果物が cleanup 済みの場合は `410 Gone`（`backup artifact expired; run backup again`）を返します。

- POST `/workspaces/restore-jobs`
  - 説明: `tar.zst` バックアップをアップロードし、workspace restore ジョブを開始します。
  - リクエスト: `Content-Type: application/octet-stream`
  - 復元前チェック:
    - `workspace.json` から `workspace_id` を先読み
    - 同じ workspace id が既に存在する場合は `409` でブロック
    - `.trash` 内で予約済み id と衝突する場合も `409` でブロック
  - 補足:
    - 一時データは `workspaces/.tmp/restore` 配下に保存されます。
    - ジョブメタは `workspaces/.xima/jobs/workspace_restore` 配下に永続保存されます。

- GET `/workspaces/restore-jobs/{job_id}`
  - 説明: restore ジョブの詳細（`status`, `progress`, `error` など）を返します。

- POST `/workspaces/restore`
  - 説明: `tar.zst` バックアップから workspace を同期復元します（互換API）。
  - 補足: restore時の一時データは `workspaces/.tmp/restore` 配下を利用します。
  - リクエスト: `Content-Type: application/octet-stream` でアーカイブ本体を送信
  - 復元前チェック:
    - `workspace.json` から `workspace_id` を先読み
    - 同じ workspace id が既に存在する場合は `409` でブロック
    - `.trash` 内で予約済み id と衝突する場合も `409` でブロック
  - 備考:
    - FastAPI 起動時に `workspaces/.tmp` は cleanup されます。
    - 再起動時、`queued`/`running` の backup/restore ジョブは `error` に更新されます（ジョブメタは保持）。

- GET `/workspaces/sidebar-order`
  - 説明: サイドバーの並び順（workspace順 / workspace内experiment順）を取得します。
  - レスポンス例:
    - `{ "version": 1, "workspaces": ["2u4sk928", ...], "experiments": {"2u4sk928": ["9zd5s05u", ...]}, "updated_at": "..." }`

- PUT `/workspaces/sidebar-order`
  - 説明: サイドバーの並び順を保存します（**last write wins**）。
  - ボディ: JSON
    - `{"workspaces": ["...", ...]}`（workspace順のみ更新）
    - `{"experiments": {"<workspace>": ["<experiment>", ...]}}`（workspace内experiment順のみ更新）
    - 両方同時指定も可能
  - 補足:
    - 不正な ID や未知の ID は保存時に無視されます。

例:

```bash
curl -s http://127.0.0.1:27801/workspaces | jq .

WS_ID=$(curl -sS -X POST http://127.0.0.1:27801/workspaces \
  -H 'Content-Type: application/json' \
  -d '{"display_name":"作業用"}' | jq -r .workspace)

curl -s http://127.0.0.1:27801/workspaces/${WS_ID} | jq .

curl -X PATCH http://127.0.0.1:27801/workspaces/${WS_ID} \
  -H 'Content-Type: application/json' \
  -d '{"display_name":"作業用"}' | jq .

# workspace backup (tar.zst)
curl -L http://127.0.0.1:27801/workspaces/${WS_ID}/backup -o ${WS_ID}.tar.zst

# workspace backup (job start -> poll -> download)
JOB_ID=$(curl -sS -X POST http://127.0.0.1:27801/workspaces/${WS_ID}/backup-jobs | jq -r .job_id)
curl -sS http://127.0.0.1:27801/workspaces/${WS_ID}/backup-jobs/${JOB_ID} | jq .
curl -L http://127.0.0.1:27801/workspaces/${WS_ID}/backup-jobs/${JOB_ID}/download -o ${WS_ID}.tar.zst

# workspace restore
RESTORE_JOB_ID=$(curl -sS -X POST http://127.0.0.1:27801/workspaces/restore-jobs \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @${WS_ID}.tar.zst | jq -r .job_id)
curl -sS http://127.0.0.1:27801/workspaces/restore-jobs/${RESTORE_JOB_ID} | jq .

# workspace restore (compatibility: synchronous)
curl -X POST http://127.0.0.1:27801/workspaces/restore \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @${WS_ID}.tar.zst | jq .
```

## 実験（Experiments）

- GET `/workspaces/{workspace}/experiments`
  - 説明: 指定ワークスペース内の実験一覧を返します。
  - クエリ:
    - `order=sidebar` を指定すると、サイドバー用に保存された順序（`/workspaces/sidebar-order` の `experiments[workspace]`）を優先して返します。未登録の experiment は末尾に ID 昇順で補完されます。

- POST `/workspaces/{workspace}/experiments`
  - 説明: 実験を作成します。必要なディレクトリ（`label_input`, `dataset`, `config`）と初期 `labels.json`（テンプレートまたは既存 `current.json` のマイグレーション）を準備します。
  - ボディ: JSON `{ "display_name": "...", "id": "..." }`
    - `id` は省略可能（省略時はサーバが短い ID を生成）。
    - `id` を指定する場合は `[a-z0-9]{8}`（base36 8 文字）を推奨します。

## ID バリデーション（Validate）

短 ID の形式と重複を事前チェックできます。

- GET `/validate/workspace-id?id=<id>`
  - レスポンス: `{ "id": "...", "valid": true|false, "exists": true|false }`

- GET `/validate/experiment-id?workspace=<workspace>&id=<id>`
  - レスポンス: `{ "workspace": "...", "workspace_exists": true|false, "id": "...", "valid": true|false, "exists": true|false }`

- PATCH `/workspaces/{workspace}/experiments/{experiment}`
  - 説明: 実験の `display_name` を変更します。
  - ボディ: JSON `{ "display_name": "..." }`

例:

```bash
curl -s http://127.0.0.1:27801/workspaces/${WS_ID}/experiments | jq .

EXP_ID=$(curl -sS -X POST http://127.0.0.1:27801/workspaces/${WS_ID}/experiments \
  -H 'Content-Type: application/json' \
  -d '{"display_name":"実験A"}' | jq -r .experiment)

curl -X PATCH http://127.0.0.1:27801/workspaces/${WS_ID}/experiments/${EXP_ID} \
  -H 'Content-Type: application/json' \
  -d '{"display_name":"実験A"}' | jq .
```

## ラベル操作（label-input）

- GET `/workspaces/{ws}/experiments/{exp}/label-input`
  - 説明: 現在の `labels.json` を取得します。

- PUT `/workspaces/{ws}/experiments/{exp}/label-input`
  - 説明: `labels.json` を**全置換（full-file overwrite）**します。PUT を受ける前に既存の `labels.json` を自動で履歴へスナップショットします。初回書き込み時は新規ファイルのスナップショットが作成されます。
  - ボディ: 新しい `labels.json`（完全な JSON）。

- GET `/workspaces/{ws}/experiments/{exp}/label-schema`
  - 説明: ラベルスキーマ JSON (`label_input/label_schema.json`) を取得します。実験作成時または初回取得時に `DEFAULT_LABEL_SCHEMA` を元にファイルを生成します（デフォルトは固定 `split` ヘッドのみ。その他のヘッドはユーザが `label_schema.json` を編集して追加・削除できます）。
  - 例:

```bash
curl -s http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/label-schema | jq .
```

- PUT `/workspaces/{ws}/experiments/{exp}/label-schema`
  - 説明: `label_schema.json` を全置換保存します。**保存ごとに自動でバージョン（history）を作成**します。

- GET `/workspaces/{ws}/experiments/{exp}/label-schema/history`
  - 説明: schema バージョン一覧を返します（`id`, `filename`, `size_bytes`, `modified_at`）。

- GET `/workspaces/{ws}/experiments/{exp}/label-schema/history/{backup_id}`
  - 説明: 指定 schema バージョンの JSON を取得します。

- POST `/workspaces/{ws}/experiments/{exp}/label-schema/history/{backup_id}/restore`
  - 説明: 指定 schema バージョンを `label_schema.json` に復元します。復元後の内容は **新しいバージョンとして自動保存**されます。

### 履歴（history）

- GET `/workspaces/{ws}/experiments/{exp}/label-input/history`
  - 説明: 履歴バックアップのメタ一覧を返します（`id`, `filename`, `size_bytes`, `modified_at`）。

- POST `/workspaces/{ws}/experiments/{exp}/label-input/history`
  - 説明: **バックアップ専用（snapshot-only）**。現在の `labels.json` を履歴へコピーします（上書きはしません）。ボディ不要。

- GET `/workspaces/{ws}/experiments/{exp}/label-input/history/{backup_id}`
  - 説明: 指定バックアップの JSON を取得します。

- POST `/workspaces/{ws}/experiments/{exp}/label-input/history/{backup_id}/restore`
  - 説明: 指定バックアップを `labels.json` に復元します。復元前に現在の `labels.json` を履歴へスナップショットします。

バックアップ ID 形式: `YYYYMMDD_HHMMSS`（同秒に複数作成された場合は `-1`, `-2` の接尾辞が付与されます）。

例:

```bash
# 取得
curl -s http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/label-input | jq .

# 上書き（full-file）
curl -X PUT -H "Content-Type: application/json" --data-binary @labels.json \
  http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/label-input

# バックアップのみ（snapshot-only）
curl -X POST http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/label-input/history

# バックアップ一覧
curl -s http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/label-input/history | jq .

# 復元
curl -X POST http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/label-input/history/labels_20251220_023103/restore
```

## 埋め込みの状態（embeddings）

`cache/embeddings/<model_slug>/` に**どの CLIP モデルの埋め込みが、何件あるか**を返す
**読み取り専用** API。候補付与（`predict_labels`）とクラスタは学習に使ったのと同じモデルの
埋め込みを必要とするため、UI がジョブを投げる前に前提を確認するために使う。

- GET `/workspaces/{ws}/experiments/{exp}/embeddings`
  - クエリなし。experiment が無くても **200**（`target_count: 0`, `models: []`）。
  - レスポンス: `{ target_count, models: [{ clip_model_name, slug, count, dim, usable, updated_at }] }`
    - `target_count`: `labels.json` の item 数。**画像ファイルの実在は見ない**ため、
      `count < target_count` は未作成とは限らない（欠損画像・読めない画像は埋め込み対象から外れる）。進捗の目安。
    - `usable`: `index.json` と `embeddings.npy` が揃っているか。**`predict_labels` と `clusters` が動く条件と同じ**。
    - `count` / `dim`: `index.json` の値。行列ファイルは読まない。
    - `updated_at`: `index.json` の更新時刻（ISO8601 / UTC）。
  - **一覧に出ないもの**: キャッシュの版が違う / `index.json` が壊れている / モデル名が空。
    いずれも `embed_images` で作り直す対象なので「無い」と扱う。

```bash
curl -s "http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/embeddings" | jq .
```

## 類似画像クラスタリング（clusters）

CLIP 埋め込み（`embed_images` ジョブが生成する `cache/embeddings`）を使って、似た画像を
クラスタにまとめて返す **読み取り専用** API。UI はこの結果を使ってクラスタ単位で
「まとめてラベル付与」する（付与自体は上記 `label-input` の PUT に乗るため、正本フロー・
履歴はそのまま）。

- GET `/workspaces/{ws}/experiments/{exp}/clusters`
  - 前提: 先に `embed_images` ジョブで埋め込みを作成しておく。未作成時は **409** を返す。
  - クエリ:
    - `k`（省略可）: クラスタ数。省略時は枚数から自動決定（`auto_k`）。
    - `scope`: `all`（既定）/ `unlabeled` / `labeled_unassigned`。
      - `unlabeled`: 指定 head で未ラベルの画像だけを対象にする。削除マークの付いた画像は除く。
      - `labeled_unassigned`: **`split` 以外のいずれかの head に値があり、`split` が `train` でも `val` でもない**画像だけを対象にする。削除マークの付いた画像は除く。ラベルは付いているがデータセットに入らない画像（`apply_label` が捨てる画像）を集める。
    - `head`（省略可）: `unlabeled` の判定に使う head id。省略時は schema の分類 head（split 以外）を自動選択。`labeled_unassigned` は head を見ないため、省略時は自動選択せず、応答の `head` は `null` になる。
    - `width`（既定 256）: 返すサムネイルパスの幅。
  - レスポンス: `{ k, total, scope, head, clip_model_name, clusters: [{ cluster_id, size, representative, members: [{ file_id, path, thumb_path }] }] }`
    - クラスタはサイズ降順、メンバは中心に近い順（代表が先頭）。

```bash
# 事前に埋め込みを作成（jobs 経由 or スクリプト）
curl -s "http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/clusters?k=3" | jq .
# 未ラベルだけをクラスタリング
curl -s "http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/clusters?scope=unlabeled" | jq .
# ラベルは付いているが split が train/val でない（＝データセットに入らない）画像だけ
curl -s "http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/clusters?scope=labeled_unassigned" | jq .
```

## 画像取得

画像取得は拡張性のため 2 つの経路を提供します。

- GET `/workspaces/{ws}/experiments/{exp}/images/by-item/{id}`
  - 説明: `labels.json` 内のアイテムの `id` フィールドをキーにファイルを返します（現行の推奨方法）。
  - クエリ: `?w=256` で WebP サムネイルを返します。**オンザフライ生成は廃止**し、`make_label_list`（下記）などで事前生成されたキャッシュを返します。サムネイルは `experiments/<exp>/cache/thumbs/w256/<shard>/<key>.webp` に置いてください。存在しない場合は 404 を返します。`w` 未指定時は原寸（変換なし）。

- GET `/workspaces/{ws}/experiments/{exp}/images/by-file/{file_id}`
  - 説明: 旧仕様互換のため、アイテム内の `file_id` フィールドをキーにファイルを返します。
  - クエリ: `?w=256` で上記と同様に WebP サムネイルを返します（事前生成が必須）。

共通挙動:

- ファイルは `labels.json` の該当アイテムに記載されたパス（`rel_path` 等）を正規化して `source_dir` 配下から返却します。
- `source_dir` を逸脱するパスは拒否されます。
- サムネイルは `experiments/<exp>/cache/thumbs/w{width}/` を参照します。存在しない場合は 404。生成は `agent/pipeline/make_label_list.py --thumb-width 256 ...` を利用してください。
- レスポンス本体は FastAPI からは返さず、`X-Accel-Redirect` により nginx の internal location で静的配信します（高速化目的）。

例:

```bash
# by-item (labels.json の "id" を使用)
curl -O http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/images/by-item/<ID>

# by-file (labels.json の "file_id" を使用 - 旧仕様互換)
curl -O http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/images/by-file/<FILE_ID>
```

## ジョブ（Jobs）

エージェントはローカルのスクリプト（`agent/pipeline` 内）をサブプロセスで実行することで、`apply_label` や `augment_gray` 等の処理を実行します。実行中のスクリプトは `AGENT_JOB_FILE` 環境変数を通じて進捗を書き込みできます。

- POST `/workspaces/{ws}/experiments/{exp}/jobs`
  - 説明: ジョブを作成して即時実行（シングルジョブ排他）。レコードを `agent/state/jobs/<id>.json` とログ `agent/state/jobs/<id>.log` に保存します。
  - ボディ: JSON `{ "type": "apply_label"|"augment_gray"|"make_label_list"|"train_epoch"|"infer_heads"|"train"|"infer_scores", "args": {...} }`。
  - 既定の補完:
    - `apply_label`: `labels`, `root`, `dataset_root` を自動補完
    - `augment_gray`: `dataset_root` を自動補完
    - `make_label_list`: `input_dir`, `html_dir`, `output`, `thumb_width` を自動補完（デフォルトでサムネイル w256 を事前生成）
    - `train_epoch` / `train`: `index`（`experiments/<exp>/dataset/index.json`）を自動補完（ただし `args.raw_args` に `--index` が含まれる場合は補完しません）
    - `infer_heads` / `infer_scores`: `run_dir`（最新 `experiments/<exp>/models/run_*`）, `index`, `output` を自動補完（`args.raw_args` に同等オプションが含まれる場合は補完しません）
  - `train_epoch` / `infer_heads` の `args.device` では `cpu|cuda|metal(mps)` を指定可能です。`auto` 相当は `cuda -> mps -> cpu` の順で自動選択されます。

- GET `/workspaces/{ws}/experiments/{exp}/jobs/{job_id}`
  - 説明: ジョブのメタ（状態、開始・終了時刻、exit_code、progress 等）を返します。

- GET `/workspaces/{ws}/experiments/{exp}/jobs`?limit={n}&offset={n}
  - 説明: 指定スコープ（ws/exp）のジョブ履歴を返します。`agent/state/jobs/*.json` を読み込み、`created_at` 降順で返します。
  - クエリ:
    - `limit` (デフォルト 50, 最大 200)
    - `offset` (デフォルト 0)
  - レスポンス: `{ "jobs": [JobRecord, ...] }`

- DELETE `/workspaces/{ws}/experiments/{exp}/jobs`
  - 説明: 指定したジョブ履歴を物理削除します（`agent/state/jobs/<id>.json` と `agent/state/jobs/<id>.log`）。実行中のジョブは削除できず `skipped_running` に入ります。
  - ボディ: JSON `{ "job_ids": ["<JOB_ID>", ...] }`
  - レスポンス: `{ "deleted": ["..."], "skipped_running": ["..."], "not_found": ["..."] }`

- GET `/workspaces/{ws}/experiments/{exp}/jobs/{job_id}/logs`?tail={n}
  - 説明: ジョブのログファイルを返します。`tail` クエリで末尾行数を指定（デフォルト 5000）。

例:

```bash
# ジョブ作成
curl -X POST -H "Content-Type: application/json" -d '{"type":"apply_label","args":{}}' \
  http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/jobs

# labels.json を再生成 + サムネイル(w256)を事前生成
curl -X POST -H "Content-Type: application/json" -d '{"type":"make_label_list","args":{}}' \
  http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/jobs

# 例: サムネイル生成をスキップ
curl -X POST -H "Content-Type: application/json" -d '{"type":"make_label_list","args":{"skip_thumbs":true}}' \
  http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/jobs

# ジョブステータス取得
curl -s http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/jobs/<JOB_ID> | jq .

# ジョブ一覧（履歴）取得
curl -s 'http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/jobs?limit=50&offset=0' | jq .

# ジョブ履歴の削除
curl -s -X DELETE http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"job_ids":["<JOB_ID>"]}' | jq .

# ジョブログの末尾 200 行を取得
curl http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/jobs/<JOB_ID>/logs?tail=200
```

## スキーマと拡張点

- サポートされるジョブ種別: `apply_label`, `augment_gray`, `make_label_list`, `train_epoch`, `infer_heads`（旧互換: `train`, `infer_scores`）。
- `JobCreateRequest` は `type` と任意の `args` を受け取り、`jobs` マネージャが実行コマンドを組み立てます。

補足:

- `train_epoch` は `args.raw_args`（例: `"--heads shape,color --epochs 40 ..."`）で任意の CLI オプションを渡せます。

## 運用上の注意

- `labels.json` は大きくなりがちです。頻繁なバックアップや PUT による転送コストに注意してください。
- ファイルパスの正規化ルール: `rel_path`, `path`, `file_path`, `dataset_path` のいずれかをサポートし、すべて `source` 配下の相対パスへ正規化されます。

---

HTTP API の全経路は本ファイルにまとめてあります。導入と運用は [README.md](README.md) を参照してください。
