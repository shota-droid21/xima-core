# xima-agent (FastAPI MVP)

Local agent for CPU/GPU hosts (NVIDIA CUDA and Apple Silicon Metal/MPS). No DB; state and config live under `state/`. Listens on `127.0.0.1:27800` (direct) or `127.0.0.1:27801` via docker-compose nginx.

## Workspace layout (recommended)

```
agent/
  workspaces/
    <workspace>/
      source_images/          # canonical images (正本)
      experiments/
        <experiment>/
          label_input/
            labels.json
            history/
          dataset/
          config/
```

## Setup

1. `cd agent`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `uvicorn app.main:app --host 127.0.0.1 --port 27800`
5. Health check: `curl http://127.0.0.1:27800/health`

SSH tunnel (split setup):

```bash
ssh -N -L 27800:127.0.0.1:27800 user@gpu-host
# Then open http://127.0.0.1:27800/docs
```

## Config

Configuration is environment-variable based.

- `XIMA_WORKSPACES_ROOT`: absolute path to the workspaces root (default: `<repo>/workspaces` in the container filesystem)
- `XIMA_AGENT_PORT`: listen port (default: 27800)

Directory names are fixed (to avoid UI/agent mismatch):

- `source_images`
- `experiments`

## Experiments API

## Quickstart (recommended order)

1. Create workspace (scoped)

```bash
WS_ID=$(curl -sS -X POST http://127.0.0.1:27801/workspaces \
  -H 'Content-Type: application/json' \
  -d '{"display_name":"作業用ワークスペース"}' | jq -r .workspace)
echo "WS_ID=$WS_ID"
```

2. Create experiment (scoped)

```bash
EXP_ID=$(curl -sS -X POST http://127.0.0.1:27801/workspaces/${WS_ID}/experiments \
  -H 'Content-Type: application/json' \
  -d '{"display_name":"実験1"}' | jq -r .experiment)
echo "EXP_ID=$EXP_ID"
```

3. Optional: copy legacy assets (copy-only)

```bash
python scripts/bootstrap_workspace.py --workspace "${WS_ID}" --experiment default
```

Notes:

- `create` endpoints are responsible for making directories (`mkdir -p`) and never delete/move existing data.
- `select` endpoints only switch to an existing workspace/experiment and will error if the target does not exist.

- `GET /workspaces` : list available workspaces (returns `{ id, display_name, workspace_root }[]`).
- `POST /workspaces` : create a workspace (JSON body `{ display_name, id? }`, id is recommended as `[a-z0-9]{8}`).
- `PATCH /workspaces/{workspace}` : rename workspace display_name.
- `GET /workspaces/{workspace}/experiments` : list experiments (returns `{ id, display_name, path }[]`).
- `POST /workspaces/{workspace}/experiments` : create an experiment (JSON body `{ display_name, id? }`, id is recommended as `[a-z0-9]{8}`).
- `PATCH /workspaces/{workspace}/experiments/{experiment}` : rename experiment display_name.

## label_input (current)

- Scoped path used: `workspaces/<ws>/experiments/<exp>/label_input/labels.json`
- `GET /workspaces/{ws}/experiments/{exp}/label-input`
- `PUT /workspaces/{ws}/experiments/{exp}/label-input`
- Schema file lives at `workspaces/<ws>/experiments/<exp>/label_input/label_schema.json`.
- `GET /workspaces/{ws}/experiments/{exp}/label-schema` (see API.md for payload).
- `GET /workspaces/{ws}/experiments/{exp}/label-input/history`
- `POST /workspaces/{ws}/experiments/{exp}/label-input/history`
- `GET /workspaces/{ws}/experiments/{exp}/label-input/history/{backup_id}`
- `POST /workspaces/{ws}/experiments/{exp}/label-input/history/{backup_id}/restore`
- Normalization on save:

## Backup & Restore (label_input)

The agent automatically writes a timestamped backup into `label_input/history/` whenever you `PUT` the labels file.
The backup is a snapshot of the **current `labels.json` before it gets overwritten** (so you can roll back even if you accidentally `PUT` a partial/incorrect payload).

Important:

- `PUT` is a **full-file overwrite** API for `labels.json` (typically thousands of items). It is not meant to be used for backup-only.
- For backup-only (no overwrite), use `POST .../label-input/history`.

- Backup ID format: `YYYYMMDD_HHMMSS` (if a collision occurs within the same second a numeric suffix is appended, e.g. `20251220_123456-1`).
- Backups are stored as: `workspaces/<ws>/experiments/<exp>/label_input/labels_<backup_id>.json`.

Endpoints:

- `GET /workspaces/{ws}/experiments/{exp}/label-input/history` — list available backups (id, filename, size, modified_at).
- `POST /workspaces/{ws}/experiments/{exp}/label-input/history` — snapshot the current `labels.json` into history (no overwrite).
- `GET /workspaces/{ws}/experiments/{exp}/label-input/history/{backup_id}` — fetch a specific backup's JSON.
- `POST /workspaces/{ws}/experiments/{exp}/label-input/history/{backup_id}/restore` — restore the specified backup to the canonical `labels.json`. The agent will first snapshot the current `labels.json` into history before restoring.

Examples:

List backups:

```bash
curl -s http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/label-input/history | jq .
```

Snapshot current labels.json (backup-only):

```bash
curl -X POST http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/label-input/history
```

詳細な API 仕様は別ファイルにまとめています: [agent/API.md](agent/API.md)

Download a backup:

```bash
curl -s \
  http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/label-input/history/20251220_123456 \
  -o labels_20251220_123456.json
```

Restore a backup (this will create a history snapshot of the current labels.json first):

```bash
curl -X POST \
  http://127.0.0.1:27801/workspaces/ws_test/experiments/exp1/label-input/history/20251220_123456/restore
```

Notes:

- The canonical labels file is `workspaces/<ws>/experiments/<exp>/label_input/labels.json`.
- A default label schema is materialized on experiment creation (or on first schema fetch) into `label_input/label_schema.json` using `DEFAULT_LABEL_SCHEMA` (fixed `split` head only; add/remove other heads manually in `label_schema.json`).
- `GET /workspaces/{ws}/experiments/{exp}/label-schema` returns the schema JSON; callers should cache alongside labels.
- If an older `current.json` exists, the agent will migrate it to `labels.json` automatically on first access.
- Restores are best-effort and will create a backup of the pre-restore state in history; consider exporting important backups externally if you need a long-term archive.

  - Path fields (`rel_path`, `path`, `file_path`, `dataset_path`) are rewritten to **source_dir-relative POSIX paths**.
  - On failure (outside `source_dir`), 400 is returned.

Requirements inside `items[]` (or top-level list):

- Must include an image identifier using the canonical `id` field in `labels.json`.
- Must include a path field (candidates: `rel_path`, `path`, `file_path`, `dataset_path`) which will be stored as `rel_path` under `source_dir`.

## Images

Endpoints:

- `GET /workspaces/{ws}/experiments/{exp}/images/by-item/{id}` — 推奨: `labels.json` の `id` を指定して取得します。
- `GET /workspaces/{ws}/experiments/{exp}/images/by-file/{file_id}` — 互換: 旧仕様の `file_id` を指定して取得します。

Resolution: normalized `rel_path` under `source_dir`. Paths escaping `source_dir` are rejected.

Thumbnails:

- オンザフライ生成は廃止。`?w=256` などのリクエストは事前生成済みの `experiments/<exp>/cache/thumbs/w{width}/...` を返します。
- 生成には `agent/pipeline/make_label_list.py -i <images> -o <exp>/label_input/labels.json --thumb-width 256` を使用してください（labels.json の作成と同時にサムネイルを作成）。
- キャッシュが無い場合は 404 を返します。
- 画像の配信は `X-Accel-Redirect` により nginx が静的配信します（FastAPI はパス解決/認可/ヘッダのみ）。

## Jobs

- `POST /workspaces/{ws}/experiments/{exp}/jobs` : `{ "type": "apply_label"|"make_label_list"|"purge_deleted_images"|"augment_gray"|"train_epoch"|"train"|"infer_heads"|"infer_scores", "args": { ... } }` → `{ "job_id", "status" }`
- `GET /workspaces/{ws}/experiments/{exp}/jobs?limit=50&offset=0` : job history (created_at desc)
- `GET /workspaces/{ws}/experiments/{exp}/jobs/{job_id}` : job status
- `GET /workspaces/{ws}/experiments/{exp}/jobs/{job_id}/logs?tail=5000` : tail logs
- Scripts are resolved per job:
  - `agent/pipeline/<script>.py` (canonical)
  - transitional fallback: `02_train/<script>.py`
- Jobs are queued and executed asynchronously by worker(s). Running parallelism depends on worker concurrency.

make_label_list:

- `type=make_label_list` は `source_images/` を走査して `label_input/labels.json` を生成/更新し、同時に `experiments/<exp>/cache/thumbs/w256/` のサムネイルも事前生成します。
- 必要に応じて `args` で `thumb_width`, `skip_thumbs`, `remove_labels` を指定できます。

purge_deleted_images:

- `type=purge_deleted_images` は `label_input/labels.json` の `delete=true` item を対象に、`source_images/` から元画像を削除します。
- `args.dry_run=true` を指定すると削除せず件数確認のみ実行できます。
- 実削除後は `make_label_list` を実行して `labels.json` / サムネイルキャッシュを同期し、続けて `apply_label` を実行して dataset を再生成してください。

## Bootstrap (copy-only, no deletes)

Goal: keep existing `01_processing/downloads` and `02_train` intact; copy into agent workspace.

Script (recommended):

```bash
cd agent
python scripts/bootstrap_workspace.py --workspace ws_test --experiment default
```

What it does:

- Copies `<legacy-root>/01_processing/downloads/*` to `workspaces/<ws>/<source_dir>/` (if present).
- Leaves originals untouched.

Manual (rsync) example if you prefer:

```bash
rsync -av ../01_processing/downloads/ ./workspaces/ws_test/source_images/
```

## CORS

MVP uses `allow_origins=["*"]`. Restrict to SaaS domains for production.

## Notes

- Path traversal remains blocked; only `workspaces_root` 以下が許可されます。
- Image processing (thumbnails etc.) is intentionally not implemented.
