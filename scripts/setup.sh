#!/usr/bin/env bash
#
# xima セットアップ（導入を 1 コマンドに）。
#
#   ./scripts/setup.sh          # 通常（UI あり・ML 依存あり）
#   ./scripts/setup.sh --no-ui  # API / CLI のみ（UI を用意しない）
#   ./scripts/setup.sh --no-ml  # torch/CLIP を入れない（同期系のみ・開発用）
#   ./scripts/setup.sh --help
#
# やること:
#   1. python3 の preflight
#   2. core/.venv の作成（既存なら再利用）
#   3. 依存の install（requirements のハッシュが変わっていなければスキップ）
#   4. app dist（UI）の用意
#   5. workspaces / state の初期化
#
# 生成物（.venv / app-dist / state）はすべて再生成可能な派生物であり、
# workspaces（ユーザ資産）には一切書き込まない。再実行は安全（冪等）。
#
# 完了後: ./scripts/run-local.sh
#
# 本スクリプトは core と一緒に配布される（subtree split に含まれる）ため、
# 2 つのレイアウトで動く必要がある:
#
#   公開 core リポジトリ:  <repo>/scripts/setup.sh   → CORE_ROOT=<repo>
#   開発 monorepo:        <repo>/core/scripts/setup.sh → CORE_ROOT=<repo>/core
#
# どちらも「スクリプトの 1 つ上」が core のルートなので CORE_ROOT の求め方は共通。
# 違いは **app/（UI のソース）が隣にあるかどうか**だけで、これが app dist の
# 入手経路（ローカルビルド or Releases）を分ける。
#
# 設計正本: docs/90_decisions.md Decision 035
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV="$CORE_ROOT/.venv"
REQ_DIR="$CORE_ROOT/api"
APP_DIST_DIR="$CORE_ROOT/app-dist"
WS_ROOT="${XIMA_WORKSPACES_ROOT:-$CORE_ROOT/workspaces}"
STATE_DIR="$CORE_ROOT/api/state"

# monorepo なら app/ のソースが core の隣にある（公開 core には存在しない）。
APP_SRC_DIR="$CORE_ROOT/../app"
IS_MONOREPO=0
[ -f "$APP_SRC_DIR/package.json" ] && IS_MONOREPO=1

# app dist の取得元。monorepo ではローカルビルドにフォールバックする。
#
# 既定は**公開 core リポジトリ**。この分岐に来るのは公開 core を clone した利用者だけで
# あり、開発 monorepo（private・匿名では落とせない）を向けても取得できないため。
APP_DIST_URL="${XIMA_APP_DIST_URL:-}"
RELEASE_REPO="${XIMA_RELEASE_REPO:-shota-droid21/xima-core}"

WANT_UI=1
WANT_ML=1

PY_MIN_MAJOR=3
PY_MIN_MINOR=12

usage() {
  sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --no-ui) WANT_UI=0 ;;
    --no-ml) WANT_ML=0 ;;
    -h|--help) usage ;;
    *) echo "error: unknown option: ${1} (--help を参照)" >&2; exit 2 ;;
  esac
  shift
done

step() { echo; echo "▸ $*"; }
info() { echo "  $*"; }
die()  { echo; echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------- 1) preflight

step "[1/5] 前提を確認します"

PY_BIN="${XIMA_PYTHON:-}"
if [ -z "$PY_BIN" ]; then
  PY_BIN="$(command -v python3 || true)"
fi
if [ -z "$PY_BIN" ]; then
  die "python3 が見つかりません。
  Python ${PY_MIN_MAJOR}.${PY_MIN_MINOR} 以上を入れてから再実行してください。
    macOS:  brew install python@3.12
    Linux:  apt install python3 python3-venv  など
  別の Python を使う場合: XIMA_PYTHON=/path/to/python3 ./scripts/setup.sh"
fi

PY_VER="$("$PY_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if ! "$PY_BIN" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= ($PY_MIN_MAJOR, $PY_MIN_MINOR) else 1)"; then
  die "Python ${PY_VER} は古すぎます（${PY_MIN_MAJOR}.${PY_MIN_MINOR} 以上が必要）。
  新しい Python を入れるか、XIMA_PYTHON で明示してください。"
fi
info "python: $PY_BIN (${PY_VER})"

if ! "$PY_BIN" -c "import venv" >/dev/null 2>&1; then
  die "python3 の venv モジュールが使えません。
    Debian/Ubuntu:  apt install python3-venv"
fi

# ------------------------------------------------------------------ 2) venv

step "[2/5] Python 環境を用意します"

if [ -x "$VENV/bin/python" ]; then
  info "既存の venv を再利用: $VENV"
else
  info "venv を作成: $VENV"
  "$PY_BIN" -m venv "$VENV"
fi
VPY="$VENV/bin/python"

# ------------------------------------------------------------- 3) dependencies

step "[3/5] 依存をインストールします"

REQ_FILES=("$REQ_DIR/requirements.base.txt")
if [ "$WANT_ML" = "1" ]; then
  REQ_FILES+=("$REQ_DIR/requirements.jobs.txt")
else
  info "--no-ml: ML 依存（torch / CLIP）を省略します（学習・推論は使えません）"
fi

for f in "${REQ_FILES[@]}"; do
  [ -f "$f" ] || die "requirements が見つかりません: $f"
done

HASH_FILE="$VENV/.requirements.hash"
CURRENT_HASH="$("$VPY" - "${REQ_FILES[@]}" <<'PY'
import hashlib, sys
h = hashlib.sha256()
for path in sys.argv[1:]:
    with open(path, "rb") as fp:
        h.update(fp.read())
print(h.hexdigest())
PY
)"

if [ -f "$HASH_FILE" ] && [ "$(cat "$HASH_FILE")" = "$CURRENT_HASH" ]; then
  info "依存に変更なし。インストールをスキップします"
else
  info "pip を更新中…"
  "$VPY" -m pip install --upgrade --quiet --disable-pip-version-check pip
  for f in "${REQ_FILES[@]}"; do
    info "install: $(basename "$f")"
    if [ "$(basename "$f")" = "requirements.jobs.txt" ]; then
      info "torch / CLIP を含みます。ダウンロードは数百 MB、venv は 700 MB 前後になり、回線により数分かかります"
    fi
    # --quiet を付けない。付けると pip の進捗が完全に消え、torch の取得中は数分間
    # 無出力になる。実際には進んでいても停止と区別できず、中断されれば結果は
    # 「手順が通らない」と同じになる（#167）。
    "$VPY" -m pip install --disable-pip-version-check --prefer-binary -r "$f"
  done
  echo "$CURRENT_HASH" > "$HASH_FILE"
fi

# -------------------------------------------------------------- 4) app dist

step "[4/5] UI（app）を用意します"

# 取得した app dist の版を表示する。setup.sh は releases/latest を参照するため、
# これが無いと再実行しても「更新がかかったのか」を利用者が判別できない（#174）。
report_app_dist_version() {
  if [ -f "$APP_DIST_DIR/VERSION" ]; then
    info "app のバージョン: $(cat "$APP_DIST_DIR/VERSION")"
  else
    # v0.1.2 以前のアセットには VERSION が入っていない。
    info "app のバージョン: 不明（この版の書庫にはバージョン情報がありません）"
  fi
}

fetch_and_extract() {
  # $1: URL
  local url="$1"
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  info "取得中: $url"
  if ! curl -fsSL "$url" -o "$tmp/app-dist.tar.gz"; then
    return 1
  fi
  rm -rf "$APP_DIST_DIR"
  mkdir -p "$APP_DIST_DIR"
  if ! tar -xzf "$tmp/app-dist.tar.gz" -C "$APP_DIST_DIR" --strip-components=1 2>/dev/null; then
    # --strip-components が合わない書庫（直下に index.html）にも対応する
    rm -rf "$APP_DIST_DIR"
    mkdir -p "$APP_DIST_DIR"
    tar -xzf "$tmp/app-dist.tar.gz" -C "$APP_DIST_DIR"
  fi
  [ -f "$APP_DIST_DIR/index.html" ]
}

if [ "$WANT_UI" = "0" ]; then
  info "--no-ui: UI を用意しません（API / CLI のみで利用できます）"
elif [ -n "$APP_DIST_URL" ]; then
  fetch_and_extract "$APP_DIST_URL" \
    || die "app dist を取得できませんでした: $APP_DIST_URL
  URL を確認するか、--no-ui で API / CLI のみの構成にしてください。"
  info "配置: $APP_DIST_DIR"
  report_app_dist_version
elif [ "$IS_MONOREPO" = "1" ]; then
  # monorepo: app/ のソースがあるのでローカルビルド
  command -v npm >/dev/null 2>&1 \
    || die "app/ はありますが npm が見つかりません。
  Node.js を入れるか、--no-ui で API / CLI のみの構成にしてください。"
  info "monorepo を検出。app をローカルビルドします"
  ( cd "$APP_SRC_DIR" && npm install --silent && npm run build )
  [ -f "$APP_SRC_DIR/dist/index.html" ] || die "app のビルドに失敗しました（app/dist/index.html がありません）"
  info "ビルド完了: $APP_SRC_DIR/dist"
  info "app のバージョン: $(cat "$CORE_ROOT/VERSION" 2>/dev/null || echo 不明)（ローカルビルド）"
else
  # 公開 core: Releases から取得
  url="https://github.com/${RELEASE_REPO}/releases/latest/download/xima-app-dist.tar.gz"
  if fetch_and_extract "$url"; then
    info "配置: $APP_DIST_DIR"
    report_app_dist_version
  else
    die "app dist を Releases から取得できませんでした。
    取得元: $url
  ネットワークを確認するか、次のいずれかで回避できます:
    - XIMA_APP_DIST_URL=<url> ./scripts/setup.sh
    - ./scripts/setup.sh --no-ui   （API / CLI のみ）"
  fi
fi

# ------------------------------------------------------------ 5) directories

step "[5/5] ディレクトリを初期化します"

# workspaces はユーザ資産。既存には触れず、無い場合だけ作る。
if [ -d "$WS_ROOT" ]; then
  info "既存の workspaces を使用: $WS_ROOT"
else
  mkdir -p "$WS_ROOT"
  info "workspaces を作成: $WS_ROOT"
fi
mkdir -p "$STATE_DIR/jobs"
info "state: $STATE_DIR"

# --------------------------------------------------------------------- done

# core と app は別々に更新される（core = git pull / app = setup.sh 再実行）ため、
# 版はばらつきうる（Decision 039-3）。1 つにまとめず両方を出す（#174）。
CORE_VERSION_TEXT="$(cat "$CORE_ROOT/VERSION" 2>/dev/null || echo 不明)"
if [ "$WANT_UI" = "1" ] && [ -f "$APP_DIST_DIR/VERSION" ]; then
  APP_VERSION_TEXT="$(cat "$APP_DIST_DIR/VERSION")"
elif [ "$WANT_UI" = "1" ]; then
  APP_VERSION_TEXT="不明"
else
  APP_VERSION_TEXT="（--no-ui）"
fi

cat <<EOS

セットアップが完了しました。

  core: ${CORE_VERSION_TEXT}
  app : ${APP_VERSION_TEXT}

  次のコマンドで起動します:

    ./scripts/run-local.sh

  起動後、ブラウザで http://127.0.0.1:27800 を開いてください。
EOS

if [ "$WANT_UI" = "1" ]; then
  cat <<'EOS'
  はじめて使う場合は、Workspaces 画面の「デモを作成」から試せます。
EOS
else
  cat <<'EOS'
  --no-ui で構成したため UI はありません。API / CLI で利用してください（core/API.md）。
EOS
fi

if [ "$WANT_ML" = "0" ]; then
  cat <<'EOS'

  注意: --no-ml で構成したため、学習・推論・埋め込みは実行できません。
        有効にするには --no-ml なしで再実行してください。
EOS
fi
echo
