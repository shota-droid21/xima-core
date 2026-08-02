# xima-core

**xima-core** is the local-first core of the xima ecosystem.  
It runs entirely on your machine and provides labeling, dataset preparation,
and training pipelines via a local API.

This repository contains **only the agent** (API + pipelines).  
The SaaS UI and cloud services are **not included** here.

---

## What xima-core is

- A **local ML agent** for image labeling and training
- Runs on your own machine (CPU / GPU)
- Exposes a local HTTP API (FastAPI)
- Uses the **filesystem as the source of truth**
- Can be used via **API / CLI only** (no UI required)

xima-core is designed for:

- ML engineers
- individual developers
- hobbyists running experiments locally
- users who want full ownership of their data

---

## What xima-core is NOT

- ❌ A cloud service
- ❌ A hosted training platform
- ❌ A no-code ML tool
- ❌ A data-collecting application

Your images, labels, and models **never leave your machine**.

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
- Agent containers are replaceable
- Copying `workspaces/` enables full recovery on a new machine
- Docker mounts `workspaces/` as a volume by design

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
git clone https://github.com/<your-org>/xima-core.git
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

### Start the Agent (Docker / distributed)

Only needed for distributed execution. Interactive launcher:

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

To try labeling and training without preparing your own images:

```bash
./scripts/agent-launcher.sh   # choose: 7) demo
```

Or call the API directly:

```bash
curl -X POST http://127.0.0.1:27800/demo
```

This creates a workspace/experiment with 36 generated sample images
(`circle` / `square` / `triangle`, 12 each), a matching label schema,
and imports them so you can start labeling right away.
The UI also offers a "Create demo workspace" button when no workspace exists.

Sample images are **generated at runtime, not bundled** — no third-party assets,
no binary blobs in the repository, and reproducible (fixed seed).

---

### Authentication Mode

`xima-core` supports two authentication modes controlled by `XIMA_AGENT_AUTH_MODE`:

- `open` (default): no authentication. nginx does **not** call `auth_request`.
- `external`: JWT validation with an external identity provider. nginx uses `auth_request` to call the agent's `/_auth`.

Examples:

```bash
# Open mode (default)
docker compose up -d --build

# External mode (keeps the existing auth_request behavior)
XIMA_AGENT_AUTH_MODE=external docker compose up -d --build
```

Note:

- In `open` mode, nginx will attach dummy identity headers such as `X-User-Id` for downstream compatibility.
- `open` mode is intended for trusted/local environments only.

External mode JWT auth can use Redis whitelist cache (enabled by default):

- `XIMA_AGENT_AUTH_WHITELIST_ENABLED` (default: `1`)
- `XIMA_AGENT_AUTH_WHITELIST_TTL_SECONDS` (default: `300`)
- `XIMA_AGENT_AUTH_WHITELIST_REDIS_URL` (default: `redis://redis:6379/1`)
- `XIMA_AGENT_AUTH_WHITELIST_KEY_PREFIX` (default: `xima:auth:whitelist:v1`)

### External Auth Test: Multi-Agent (4 nodes)

For local reproducibility tests with multiple external-auth-mode agents:

```bash
docker compose -f docker-compose.ext-test-multi.yml up -d --build
```

- agent-1: `http://127.0.0.1:27901`
- agent-2: `http://127.0.0.1:27902`
- agent-3: `http://127.0.0.1:27903`
- agent-4: `http://127.0.0.1:27904`

Each node has isolated runtime data:

- workspace: `workspaces-ext-test-{1..4}`
- state: `api/state-ext-test-{1..4}`

---

## Pairing (OTP) (external mode only)

Pairing is an additional safety step for **external mode** to prove that the user can interact with the agent machine.

This repository implements **Step A** and **Step B** only:

- Step A: generate an OTP challenge and save it to `workspaces/.xima/pairing.json`
- Step B: verify OTP via REST and return a short-lived signed proof (exp=90s)

**TODO (not implemented in this session):**

- Step C+: SaaS backend must verify the proof signature, bind agent↔user, and persist in DB.

### Step A: start pairing on the agent machine

Run the pairing start script (inside the api container is the easiest):

```bash
docker exec -it xima-agent-api python -m app.tools.pair_start --workspaces /data/workspaces
```

It prints:

- `OTP` (you will type this into the UI)
- `expires_at` (OTP is valid for 90 seconds)

If an unused challenge still exists, `pair_start` invalidates it and issues a new OTP.

It also writes state under workspaces:

- `workspaces/.xima/pairing.json` (challenge state, otp_hash only)
- `workspaces/.xima/keys/agent_ed25519_private.pem`
- `workspaces/.xima/keys/agent_ed25519_public.pem`

### Step B: prove OTP via REST

In **external mode** only:

```bash
XIMA_AGENT_AUTH_MODE=external docker compose up -d --build
```

Then call:

```bash
curl -sS -X POST http://127.0.0.1:27801/auth/pair/prove \
   -H 'Content-Type: application/json' \
   -d '{"otp":"123-456","client_name":"chrome-macbook"}' | jq
```

Success returns a signed proof payload + signature.

Note:

- `/auth/pair/prove` is intentionally **unauthenticated** (OTP-only) and is excluded from nginx `auth_request`.
- The expected workflow is: user performs Step A/B locally → send the proof body to SaaS → SaaS verifies and then issues API JWTs for subsequent agent API calls.

Replay/expiry rules:

- Reusing the same OTP after successful prove => **409**
- After `expires_at` => **410**

By default:

- API: `http://127.0.0.1:27800`
- Static image access (nginx): `http://127.0.0.1:27801`

Health check:

```bash
curl http://127.0.0.1:27800/health
```

---

## Usage

xima-core can be used in three ways:

1. **HTTP API** (recommended)
2. **CLI / scripts**
3. **Custom UI or automation**

The official xima UI (Plus features) communicates with xima-core
via the same local API.

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

- **xima-core**: public, free, local core
- **xima (SaaS/UI)**: account management, Plus UI features, convenience tooling

xima-core will always remain usable on its own.
