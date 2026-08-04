# xima-core

**xima-core** is the local-first core of the xima ecosystem.  
It provides labeling, dataset preparation, and training pipelines behind a local
HTTP API, and runs on hardware you control.

This repository contains the **core engine** (API + ML pipelines), released under
Apache-2.0. The graphical UI is a separate, proprietary bundle that is fetched at
setup time — it is **not part of this repository**. Everything here works without
it via the HTTP API and CLI.

There is no account, no API key, and no telemetry. In the default configuration
the API binds to `127.0.0.1`, and every input and output is a file under
`workspaces/`.

The one outbound request the engine makes is a **one-time download of the CLIP
weights** on the first training or inference run (cached under `~/.cache/clip`).
If the machine has no network access, fetch the weights separately and place them
in that cache.

---

## What xima-core is

- A **local ML engine** for image labeling and training
- Runs on your own machine (CPU / GPU)
- Exposes a local HTTP API (FastAPI)
- Uses the **filesystem as the source of truth**
- Can be used via **API / CLI only** (no UI required)

xima-core is designed for:

- ML engineers
- individual developers
- hobbyists running experiments locally

---

## What xima-core is NOT

- A hosted or managed training platform
- A no-code ML tool
- A dataset marketplace or model registry

The scope is deliberately narrow: label images, build a dataset, train a linear
head on CLIP embeddings, run inference. Anything past that is out of scope.

---

## Architecture (High Level)

```
┌──────────────┐
│   Your UI    │  (optional)
└──────┬───────┘
       │ HTTP (localhost)
┌──────▼───────┐
│  xima-core  │  FastAPI + pipeline scripts
│              │
│  - workspace │
│  - experiment│
│  - labels    │
│  - dataset   │
│  - models    │
│  - cache     │
└──────┬───────┘
       │
       ▼
 Local filesystem (source of truth)
```

---

## Directory Structure (Conceptual)

```
workspaces/
└─ <workspace_name>/            # user-defined, user-owned asset
   ├─ source_images/            # original images (canonical; only purge job deletes flagged files)
   │  └─ ... (user-defined subdirs)
   └─ experiments/
      └─ <experiment_name>/     # user-defined
         ├─ label_input/
         │  ├─ labels.json
         │  ├─ label_schema.json
         │  └─ history/
         ├─ dataset/            # generated (train/val)
         ├─ models/             # generated (.pt, etc.)
         └─ cache/              # generated (thumbnails, etc.)
```

### Workspace as User-Owned Asset

- `workspaces/` is the canonical source of truth
- The engine is replaceable; your data is not tied to any runtime
- Copying `workspaces/` enables full recovery on a new machine
- In the optional Docker setup, `workspaces/` is mounted as a volume by design

**Note on `source_images`:**

- Users may add or delete files
- Moving already-labeled files is prohibited
- Re-run `make_label_list` after additions or deletions
- If you run `purge_deleted_images`, follow with `make_label_list` (then `apply_label`) to sync labels/dataset

---

## Getting Started (Quick Start)

### Requirements

**Single machine (default): Docker is not required.**

- Python 3.12+ and a virtualenv (`core/.venv`)
- Jobs (training / inference / embedding) run **in-process** inside the API,
  so no broker (redis) and no separate worker process are needed.

Optional — only for distributed execution (GPU worker / multiple machines):

- Docker + Docker Compose
- (Optional) NVIDIA GPU + Docker GPU support
- Set `XIMA_JOB_BACKEND=celery` and `XIMA_CELERY_BROKER_URL`

> Note: stopping the API also stops any running job. On the next start those jobs
> are recorded as `error` (see Decision 036).

---

### Install and run (recommended)

```bash
git clone https://github.com/shota-droid21/xima-core.git
cd xima-core
./scripts/setup.sh
./scripts/run-local.sh
```

Then open <http://127.0.0.1:27800> in your browser.

`setup.sh` creates `core/.venv`, installs dependencies, fetches the prebuilt UI, and
initialises `workspaces/` and `state/`. It is **idempotent** — re-running it skips
anything that has not changed, and it never writes into `workspaces/`.

| Option | Effect |
| --- | --- |
| `--no-ui` | API / CLI only. No UI is fetched or built. |
| `--no-ml` | Skip torch / CLIP. Training and inference are unavailable. |
| `--help` | Show usage. |

Relevant environment variables:

| Variable | Purpose |
| --- | --- |
| `XIMA_PYTHON` | Use a specific `python3` |
| `XIMA_APP_DIST_URL` | Fetch the UI bundle from an explicit URL |
| `XIMA_WORKSPACES_ROOT` | Where workspaces live (default `core/workspaces`) |
| `XIMA_PORT` | API port (default `27800`) |

> The UI is distributed as a prebuilt bundle and is **not** part of this
> Apache-2.0 repository. Use `--no-ui` if you only need the API / CLI.

---

### Docker setup (optional — distributed execution only)

**You do not need this for normal use.** It exists for running jobs on a separate
GPU worker or across machines (`XIMA_JOB_BACKEND=celery`). Interactive launcher:

```bash
./scripts/agent-launcher.sh
```

Operation menu:

- `start`
- `stop`
- `restart`
- `rebuild`
- `log`
- `health`

Runtime settings are persisted in `xima-core/.launcher-config.env`.

- If the config file does not exist, launcher creates it with defaults.
- `start` / `stop` / `restart` / `log` / `health` read the saved config.
- `rebuild` does not read the config first; it asks runtime options again, then overwrites `xima-core/.launcher-config.env` with the new values.
- `stop` uses `docker compose stop` (containers are kept; not removed).
- `start` uses saved config and starts with `up -d --no-recreate` (keeps existing containers).
- `restart` does not re-prompt runtime options; it performs stop then start with saved config.
- `rebuild` re-prompts runtime options and starts with Docker `--build`.
- `start` / `restart` / `rebuild` run health checks (`:27800/health`, `:27801/health`) after boot.

Manual start command:

```bash
docker compose up -d --build
```

### Try It with Demo Data (optional)

To try labeling and training without preparing your own images, use the
**"Create demo workspace"** button in the UI, or call the API directly:

```bash
curl -X POST http://127.0.0.1:27800/demo
```

(The Docker launcher also offers this as menu item `7) demo`.)

This creates a workspace/experiment with 36 generated sample images
(`circle` / `square` / `triangle`, 12 each), a matching label schema,
and imports them so you can start labeling right away.
The UI also offers a "Create demo workspace" button when no workspace exists.

Sample images are **generated at runtime, not bundled** — no third-party assets,
no binary blobs in the repository, and reproducible (fixed seed).

---

### Authentication Mode

`xima-core` has three authentication modes, selected with `XIMA_AGENT_AUTH_MODE`.

| Mode | Purpose | How it works |
| --- | --- | --- |
| `local` **(default)** | Single-user desktop use | Binds to loopback only, restricts CORS to its own origin, and requires a per-start secret header (`X-Xima-Local-Key`). The secret is stored in `state/.local_secret` (mode 0600) and handed to the UI same-origin via `/local/session`. |
| `open` | Development / CI | No authentication. Trusted environments only. |
| `external` | Reserved for a future control plane | JWT validation against an external issuer. **Currently dormant** — kept so that a future manager can become the token issuer without a rewrite. |

`scripts/run-local.sh` starts in `local` mode. You normally do not need to set
this variable.

```bash
# Development / CI, no auth
XIMA_AGENT_AUTH_MODE=open ./scripts/run-local.sh
```

`local` mode protects against one specific threat: a web page in your browser
reaching your `localhost` API cross-origin. It cannot protect against another
process running as the same OS user — that would require OS-level sandboxing.

#### External mode (dormant)

Kept for a future control plane; there is no issuer shipped with xima-core today.
When enabled, nginx uses `auth_request` against `/_auth`, and a Redis whitelist
cache is available:

- `XIMA_AGENT_AUTH_WHITELIST_ENABLED` (default: `1`)
- `XIMA_AGENT_AUTH_WHITELIST_TTL_SECONDS` (default: `300`)
- `XIMA_AGENT_AUTH_WHITELIST_REDIS_URL` (default: `redis://redis:6379/1`)
- `XIMA_AGENT_AUTH_WHITELIST_KEY_PREFIX` (default: `xima:auth:whitelist:v1`)

---

## Usage

xima-core can be used in three ways:

1. **HTTP API** (recommended)
2. **CLI / scripts**
3. **Custom UI or automation**

The official xima UI communicates with xima-core over this same local API —
it has no privileged access, so anything the UI can do is scriptable.

---

## Developer Tests (Canonical Head Types)

For local pipeline contract checks, use a project-local venv:

```bash
python3 -m venv .venv
.venv/bin/pip install -r api/requirements.base.txt pytest
.venv/bin/pytest -q api/app/test_label_input_contract.py api/pipeline/test_pipeline_minimal_e2e.py
```

These tests validate:

- `PUT /label-input` schema-based normalization/validation
- minimal pipeline flow (`apply_label_mapping -> train_epoch -> infer_heads`)
- `single_class` compatibility and `multi_label` contract handling

---

## Core Concepts

Before using xima-core, it is recommended to understand:

- Workspace
- Experiment
- labels.json
- dataset vs cache
- filesystem as truth

See:

- `docs/00_overview.md`
- `docs/01_concepts.md`
- `docs/90_decisions.md`

---

## GPU Support

xima-core supports GPU execution.

For NVIDIA GPU (Docker profile):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu-nvidia.yml up -d --build
```

For Apple Silicon (Metal / PyTorch MPS):

- Use Metal profile compose overlay for `redis/nginx`:
  - `docker compose -f docker-compose.yml -f docker-compose.gpu-metal.yml up -d --build`
- Run API/worker natively from `xima-core/.venv` (managed by `./scripts/agent-launcher.sh` in `gpu-metal` profile).
- If `xima-core/.venv` is missing (or dependencies are missing), launcher auto-creates the venv and installs `api/requirements.base.txt` + `api/requirements.jobs.txt`.
- Set job `device` to `metal` (or `mps`) for `train_epoch` / `infer_heads`.
- `auto` now resolves in order: `cuda -> mps -> cpu`.

---

## Stability & Support

- xima-core is provided **as-is**
- The core APIs and pipelines are stable
- Internal implementation details may change
- Backward compatibility is best-effort

For issues and discussions:

- GitHub Issues (recommended)

---

## License

`xima-core` is licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE)
for the full terms and [`NOTICE`](NOTICE) for attribution and third-party notices.

---

## Relationship to xima

xima is a **local-first desktop product**. There is no server component and no
account: everything runs on your machine.

| Part | License | Where it lives |
| --- | --- | --- |
| **xima-core** (this repository) — API, ML pipelines, workspaces | Apache-2.0 | Public |
| **xima app** — the graphical UI | Proprietary | Prebuilt bundle fetched by `setup.sh` |

The app talks to xima-core over the same local HTTP API documented here — it has no
privileged access. **xima-core is fully usable on its own**, via the API or the CLI,
and every core capability is available without the UI.

Paid ("Plus") capabilities, if and when they exist, are unlocked by a signed
entitlement verified locally. They add automation on top of the loop; they do not
gate the core pipeline.
