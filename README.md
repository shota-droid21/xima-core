# xima-core

**xima-core** は、画像のラベリング・データセット作成・学習を、**手元のマシンだけで**
実行するためのローカル ML エンジンです。すべての機能をローカル HTTP API 越しに提供します。

このリポジトリには**エンジン本体（API + ML パイプライン）**が入っており、Apache-2.0 で
公開しています。グラフィカル UI は**別配布のバンドル**で、セットアップ時に取得されます
（このリポジトリには含まれません）。**UI が無くても、HTTP API と CLI だけで同じことが
できます。**

アカウントも API キーもテレメトリもありません。既定の構成では API は `127.0.0.1` に
バインドし、入力も出力もすべて `workspaces/` 配下のファイルです。

エンジンが行う唯一の外向き通信は、**初回の学習または推論のときに一度だけ実行される
CLIP 重みのダウンロード**です（`~/.cache/clip` にキャッシュされます）。ネットワークに
出られないマシンで使う場合は、重みを別途取得してこのキャッシュに置いてください。

> English: see [README.en.md](README.en.md).

---

## できること

- 画像のラベリングと、**類似画像のクラスタリング**（ラベルが 1 枚も無い状態から使えます）
- ラベルからの**学習用データセット生成**
- **CLIP の埋め込みに線形分類器を載せた学習**と、検証精度の確認
- 未ラベル画像への**一括推論**（クラスごとのスコアを JSON で書き出します）
- 自分のマシン（CPU / GPU）で動作
- **ファイルシステムを正本**とする
- **API / CLI だけで完結**できる（UI は必須ではありません）

想定している利用者は、ML エンジニア、個人開発者、手元で実験を回す人です。

## できないこと

- ホスト型・マネージドの学習プラットフォームではありません
- ノーコード ML ツールではありません
- データセットの取引所でもモデルレジストリでもありません

**スコープは意図的に狭くしてあります。** 画像にラベルを付け、データセットを作り、CLIP の
埋め込みに線形ヘッドを学習させ、推論する。ここから先は対象外です。

---

## 使い方の流れ

1. **まとめてラベルを付ける** — 似た画像どうしが自動でまとまるので、1 枚ずつではなく
   クラスタごとに付けられます。ラベルが 1 枚も無い状態から始められます
2. **学習して精度を見る** — 付けたラベルで分類モデルを学習します。CLIP の埋め込みに
   線形分類器を載せる方式で、検証精度が確認できます
3. **残りをまとめて推論する** — 学習したモデルで未ラベル画像を推論し、クラスごとの
   スコアを `experiments/<name>/eval/scores_<run>.json` に書き出します

いずれの段階でも、生成物はすべて `workspaces/` 配下のファイルとして残ります。

---

## はじめる

### 必要なもの

**1 台で使う場合、Docker は不要です。**

- Python 3.12 以上と virtualenv（`core/.venv`）
- ジョブ（学習 / 推論 / 埋め込み）は **API プロセス内で実行される**ため、ブローカー（redis）も
  別プロセスのワーカーも必要ありません

任意 — 分散実行（GPU ワーカー / 複数マシン）を行う場合のみ:

- Docker と Docker Compose
- （任意）NVIDIA GPU と Docker の GPU サポート
- `XIMA_JOB_BACKEND=celery` と `XIMA_CELERY_BROKER_URL` の設定

> **API を止めると、実行中のジョブも止まります。** 次回起動時、それらのジョブは `error` として
> 記録されます。

### インストールと起動

```bash
git clone https://github.com/shota-droid21/xima-core.git
cd xima-core
./scripts/setup.sh
./scripts/run-local.sh
```

ブラウザで <http://127.0.0.1:27800> を開いてください。

`setup.sh` は `core/.venv` を作り、依存をインストールし、ビルド済み UI を取得し、
`workspaces/` と `state/` を初期化します。**冪等**です — 再実行しても変わっていないものは
スキップされ、`workspaces/` には一切書き込みません。

| オプション | 効果 |
| --- | --- |
| `--no-ui` | API / CLI のみ。UI を取得もビルドもしません |
| `--no-ml` | torch / CLIP を入れません。学習と推論は使えなくなります |
| `--help` | 使い方を表示します |

関係する環境変数:

| 変数 | 用途 |
| --- | --- |
| `XIMA_PYTHON` | 使用する `python3` を指定 |
| `XIMA_APP_DIST_URL` | UI バンドルの取得元 URL を明示 |
| `XIMA_WORKSPACES_ROOT` | workspaces の置き場所（既定 `core/workspaces`） |
| `XIMA_PORT` | API のポート（既定 `27800`） |

> UI はビルド済みバンドルとして配布され、**この Apache-2.0 リポジトリには含まれません**。
> API / CLI だけが必要なら `--no-ui` を使ってください。

### デモデータで試す

自分の画像を用意しなくても、ラベリングと学習を試せます。UI の
**「Create demo workspace」**ボタンを押すか、API を直接呼んでください。

```bash
curl -X POST http://127.0.0.1:27800/demo
```

36 枚のサンプル画像（`circle` / `square` / `triangle` を 12 枚ずつ）と、対応するラベル
スキーマを持つワークスペース / 実験が作られ、すぐラベリングを始められる状態になります。

サンプル画像は**実行時に生成されます（同梱ではありません）**。第三者の素材を含まず、
リポジトリにバイナリも入らず、シード固定で再現可能です。

### 更新する

**xima は自動更新しません。** 2 つの部品が別々に配布されるため、更新は 2 手順です。

```bash
git pull                # core（このリポジトリ）
./scripts/setup.sh      # UI バンドル（最新リリースを取り直します）
```

`setup.sh` は毎回 UI バンドルを取り直すので、UI の修正はこれで反映されます。Python の依存は
`requirements` が変わっていなければスキップされ、`workspaces/` には触れません。

**いま動いている版を確認する。** `setup.sh` は終了時に両方の版を表示します。

```
セットアップが完了しました。

  core: 0.1.2
  app : 0.1.2
```

`git pull` が core を、`setup.sh` が UI を更新するため、両者は食い違うことがあります。
だから別々に表示しています。動作中の core の版は API からも取得できます。

```bash
curl -s http://127.0.0.1:27800/health
```

`agent_version` が現在動いている版、`installed_agent_version` は `workspaces/` を最初に
作った版の記録です。

**自動更新も更新チェックもありません。** `setup.sh` を自分で実行しない限り、
サーバと通信することはありません。

---

## 使い方

xima-core の使い方は 3 通りあります。

1. **HTTP API**（推奨）— [API.md](API.md) に全経路をまとめてあります
2. **CLI / スクリプト**
3. **自作の UI や自動化**

公式の xima UI も、同じローカル API 越しに xima-core と通信しています。**UI に特権的な
経路はありません**。したがって UI にできることは、すべてスクリプトから実行できます。

---

## ワークスペースの構成

```
workspaces/
└─ <workspace_name>/            # 利用者が決める、利用者の資産
   ├─ source_images/            # 元画像（正本。purge ジョブのみが対象ファイルを削除する）
   │  └─ ... （任意のサブディレクトリ）
   └─ experiments/
      └─ <experiment_name>/     # 利用者が決める
         ├─ label_input/
         │  ├─ labels.json
         │  ├─ label_schema.json
         │  └─ history/
         ├─ dataset/            # 生成物（train / val）
         ├─ models/             # 生成物（.pt など）
         ├─ eval/               # 生成物（推論スコアなど）
         └─ cache/              # 生成物（サムネイルなど）
```

### 利用者の資産としての `workspaces/`

- `workspaces/` が正本です
- **エンジンは差し替え可能で、データは特定のランタイムに縛られません**
- `workspaces/` をコピーすれば、別のマシンで完全に復元できます
- 任意の Docker 構成では、`workspaces/` はボリュームとしてマウントされます

### `source_images` の扱い

- ファイルの追加・削除は自由です
- **すでにラベルが付いたファイルの移動は禁止**です
- 追加・削除のあとは `make_label_list` を実行し直してください
- `purge_deleted_images` を実行した場合は、続けて `make_label_list`（その後 `apply_label`）を
  実行して、ラベルとデータセットを同期させてください

---

## 用語

| 語 | 意味 |
| --- | --- |
| **workspace** | 画像とその実験一式を収める単位。利用者の資産であり、正本 |
| **experiment** | 1 つの workspace の中で、ラベル体系・データセット・モデルを分離する単位 |
| **`labels.json`** | ラベルの正本。`label_input/` に置かれ、履歴は `history/` に残る |
| **dataset と cache** | `dataset/` は学習に使う生成物、`cache/` はサムネイル等の再生成可能な派生物。どちらも消しても作り直せる |
| **ファイルシステムが正本** | DB を持たない。状態はすべてファイルとして存在し、外から読める |

---

## 認証モード

`xima-core` には 3 つの認証モードがあり、`XIMA_AGENT_AUTH_MODE` で選択します。

| モード | 用途 | 仕組み |
| --- | --- | --- |
| `local` **（既定）** | 1 人でのデスクトップ利用 | ループバックのみにバインドし、CORS を自身のオリジンに限定し、起動ごとの秘密ヘッダ（`X-Xima-Local-Key`）を要求します。秘密は `state/.local_secret`（モード 0600）に保存され、同一オリジンの `/local/session` 経由で UI に渡されます |
| `open` | 開発 / CI | 認証なし。信頼できる環境でのみ使ってください |
| `external` | 将来のコントロールプレーン用に予約 | 外部発行者に対する JWT 検証。**現在は休止中** — 将来のマネージャがトークン発行者になれるよう、書き直しを避けるために残してあります |

`scripts/run-local.sh` は `local` モードで起動します。通常この変数を設定する必要はありません。

```bash
# 開発 / CI（認証なし）
XIMA_AGENT_AUTH_MODE=open ./scripts/run-local.sh
```

`local` モードが守るのは**ひとつの脅威だけ**です。ブラウザで開いている web ページが、
クロスオリジンであなたの `localhost` API に到達すること。**同じ OS ユーザで動いている別の
プロセスは防げません** — それには OS レベルのサンドボックスが必要です。

### `external` モード（休止中）

将来のコントロールプレーンのために残してあります。**現在の xima-core に発行者は同梱されて
いません。** 有効化した場合、nginx が `/_auth` に対して `auth_request` を行い、Redis の
ホワイトリストキャッシュが利用できます。

- `XIMA_AGENT_AUTH_WHITELIST_ENABLED`（既定 `1`）
- `XIMA_AGENT_AUTH_WHITELIST_TTL_SECONDS`（既定 `300`）
- `XIMA_AGENT_AUTH_WHITELIST_REDIS_URL`（既定 `redis://redis:6379/1`）
- `XIMA_AGENT_AUTH_WHITELIST_KEY_PREFIX`（既定 `xima:auth:whitelist:v1`）

---

## GPU

xima-core は GPU 実行に対応しています。

**NVIDIA GPU（Docker プロファイル）**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu-nvidia.yml up -d --build
```

**Apple Silicon（Metal / PyTorch MPS）**

- `redis` / `nginx` は Metal プロファイルの compose overlay を使います
  - `docker compose -f docker-compose.yml -f docker-compose.gpu-metal.yml up -d --build`
- API / worker は `xima-core/.venv` からネイティブに実行します
  （`gpu-metal` プロファイルの `./scripts/agent-launcher.sh` が管理します）
- `xima-core/.venv` が無い場合や依存が足りない場合、launcher が venv を作り
  `api/requirements.base.txt` と `api/requirements.jobs.txt` を導入します
- `train_epoch` / `infer_heads` のジョブの `device` に `metal`（または `mps`）を指定します
- `auto` は `cuda -> mps -> cpu` の順で解決されます

---

## 分散実行（任意・Docker）

**通常の利用では不要です。** ジョブを別の GPU ワーカーや複数マシンで実行する場合
（`XIMA_JOB_BACKEND=celery`）にだけ使います。

対話型の launcher:

```bash
./scripts/agent-launcher.sh
```

メニューは `start` / `stop` / `restart` / `rebuild` / `log` / `health` です。実行時の設定は
`xima-core/.launcher-config.env` に保存されます。

- 設定ファイルが無ければ、launcher が既定値で作成します
- `start` / `restart` は保存済み設定で起動します（`up -d --no-recreate`。既存コンテナを保持）
- `stop` は `docker compose stop`（コンテナは削除されません）
- `rebuild` は実行時オプションを聞き直し、設定を上書きしてから `--build` で起動します
- `start` / `restart` / `rebuild` は起動後にヘルスチェック（`:27800/health`、`:27801/health`）を行います

手動で起動する場合:

```bash
docker compose up -d --build
```

---

## 開発者向けテスト

パイプラインの契約を手元で確認する場合は、プロジェクト内の venv を使ってください。

```bash
python3 -m venv .venv
.venv/bin/pip install -r api/requirements.base.txt pytest
.venv/bin/pytest -q api/app/test_label_input_contract.py api/pipeline/test_pipeline_minimal_e2e.py
```

検証される内容:

- `PUT /label-input` のスキーマに基づく正規化・検証
- 最小のパイプライン（`apply_label_mapping -> train_epoch -> infer_heads`）
- `single_class` の互換性と `multi_label` の契約

---

## 状態とサポート

- xima-core は **現状のまま（as-is）**提供されます
- core の API とパイプラインは安定しています
- 内部の実装詳細は変わることがあります
- 後方互換性はベストエフォートです

**質問・不具合報告・要望は [GitHub Discussions](https://github.com/shota-droid21/xima-core/discussions) へどうぞ。**
このリポジトリの Issues は使っていません。

---

## ライセンス

`xima-core` は **Apache License 2.0** です。全文は [`LICENSE`](LICENSE)、帰属表示と
第三者ライセンスの告知は [`NOTICE`](NOTICE) を参照してください。

---

## xima 本体との関係

xima は**ローカルファーストのデスクトップ製品**です。サーバコンポーネントもアカウントも
なく、すべてが自分のマシンで動きます。

| 部品 | ライセンス | 所在 |
| --- | --- | --- |
| **xima-core**（このリポジトリ）— API・ML パイプライン・workspaces | Apache-2.0 | 公開 |
| **xima app** — グラフィカル UI | プロプライエタリ | `setup.sh` が取得するビルド済みバンドル |

app は、ここに書かれているのと同じローカル HTTP API 越しに xima-core と通信します。
**特権的な経路はありません。** xima-core は単体で完全に使え、UI が無くても core の機能は
すべて利用できます。
