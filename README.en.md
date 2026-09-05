# xima-core

> This is a short English entry point. **The full documentation is in Japanese:
> [README.md](README.md)** (and [API.md](API.md) for the HTTP API).

**xima-core** is a local ML engine for image labeling, dataset preparation, and
training. It runs on hardware you control and exposes everything over a local HTTP API.

This repository contains the **core engine** (API + ML pipelines), released under
Apache-2.0. The graphical UI is a separate, proprietary bundle fetched at setup time —
it is **not part of this repository**. Everything here works without it via the HTTP
API and CLI.

There is no account, no API key, and no telemetry. In the default configuration the
API binds to `127.0.0.1`, and every input and output is a file under `workspaces/`.

The one outbound request the engine makes is a **one-time download of the CLIP weights**
on the first training or inference run (cached under `~/.cache/clip`). If the machine
has no network access, fetch the weights separately and place them in that cache.

---

## What xima-core is

- A **local ML engine** for image labeling and training
- Runs on your own machine (CPU / GPU)
- Exposes a local HTTP API (FastAPI)
- Uses the **filesystem as the source of truth**
- Can be used via **API / CLI only** (no UI required)
- Saves each trained head as a PyTorch checkpoint that also carries the class order,
  the head type, the encoder name, and the calibrated temperature — readable without
  xima running

## What xima-core is NOT

- A hosted or managed training platform
- A no-code ML tool
- A dataset marketplace or model registry
- A serving API. By default it binds to loopback and is meant for working on your own
  machine
- A trainer for CLIP itself. Only the linear head on top of its embeddings is trained

The scope is deliberately narrow: label images, build a dataset, train a linear head
on CLIP embeddings, run inference. Anything past that is out of scope.

---

## Quick Start

**Docker is not required for single-machine use.** You need Python 3.12+ and a
virtualenv. Jobs (training / inference / embedding) run in-process inside the API, so
there is no broker and no separate worker process.

```bash
git clone https://github.com/shota-droid21/xima-core.git
cd xima-core
./scripts/setup.sh
./scripts/run-local.sh
```

Then open <http://127.0.0.1:27800> in your browser.

`setup.sh` creates `core/.venv`, installs dependencies, fetches the prebuilt UI, and
initialises `workspaces/` and `state/`. It is **idempotent** and never writes into
`workspaces/`.

| Option | Effect |
| --- | --- |
| `--no-ui` | API / CLI only. No UI is fetched or built. |
| `--no-ml` | Skip torch / CLIP. Training and inference are unavailable. |
| `--help` | Show usage. |

**Note:** `setup.sh` and `run-local.sh` print their messages in Japanese.

To update, run `git pull` (core) followed by `./scripts/setup.sh` (UI bundle).
There is no automatic update and no update check.

Everything else — the workspace layout, authentication modes, GPU setup, distributed
execution, and the full HTTP API — is documented in [README.md](README.md) and
[API.md](API.md), in Japanese.

---

## Questions and bug reports

Please use [GitHub Discussions](https://github.com/shota-droid21/xima-core/discussions).
Issues are disabled on this repository. Japanese and English are both fine.

---

## License

`xima-core` is licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for
the full terms and [`NOTICE`](NOTICE) for attribution and third-party notices.

The **xima app** (the graphical UI) is proprietary and distributed as a prebuilt bundle
fetched by `setup.sh`. It talks to xima-core over the same local HTTP API documented
here and has no privileged access.
