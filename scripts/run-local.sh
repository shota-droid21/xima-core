#!/usr/bin/env bash
#
# xima ローカル起動（1 起動・フルスタック）。
#
# app をビルドし、core API（local モード・app を同一オリジン配信）を起動する。
# 非同期ジョブ（学習 / 推論 / 埋め込み）は API プロセス内の in-process ワーカーが
# 実行するため、**Docker / redis / celery worker は不要**（Decision 036）。
# ユーザは 1 コマンドで「ラベリング → 学習 → 精度 → クラスタ一括付与」まで完走できる
# （docs/design/local-auth-and-launch.md の Exit）。
#
#   ./scripts/run-local.sh
#
# 環境変数:
#   XIMA_PORT                  API 待受ポート（既定 27800）
#   XIMA_WORKSPACES_ROOT       workspaces のルート（既定 <core>/workspaces）
#   XIMA_APP_DIST              app dist を明示（既定は app-dist → app/dist の順に探索）
#   XIMA_SKIP_BUILD=1          app のビルドを省略（monorepo の再起動用）
#   XIMA_LOCAL_JOB_CONCURRENCY 同時実行ジョブ数（既定 1）
#
# 前提: ./scripts/setup.sh を実行済みであること（.venv と app dist）。Docker は不要。
# 停止: Ctrl+C。**実行中のジョブも一緒に停止する**（次回起動時に error として記録される）。
#
# 本スクリプトは core と一緒に配布される（subtree split に含まれる）ため、
# 公開 core リポジトリと開発 monorepo の両方で動く（setup.sh と同じ判定）。
#
# 分散実行（GPU worker / 複数マシン）が要る場合は docker compose 構成を使い、
# XIMA_JOB_BACKEND=celery と XIMA_CELERY_BROKER_URL を設定する。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${XIMA_PORT:-27800}"
WS_ROOT="${XIMA_WORKSPACES_ROOT:-$CORE_ROOT/workspaces}"

# monorepo なら app/ のソースが core の隣にある（公開 core には存在しない）。
APP_SRC_DIR="$CORE_ROOT/../app"
IS_MONOREPO=0
[ -f "$APP_SRC_DIR/package.json" ] && IS_MONOREPO=1

PY="$CORE_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

# 0) 依存が古くないかを見る
#
#    **`git pull` だけして `setup.sh` を忘れると、core は起動しない。** 新しい依存が
#    requirements に入った版へ上げた場合、import の時点で落ちる。そのとき出るのは
#    uvicorn のトレースバックの中の ModuleNotFoundError で、「setup.sh を実行せよ」
#    とは読めない。v0.4.2 で zstandard を足したときに、この形で踏める状態になった。
#
#    `setup.sh` は入れた requirements のハッシュを `.venv/.requirements.hash` に
#    残している。ここではそれと突き合わせるだけで、**何もインストールしない**。
#
#    `--no-ml` で入れた場合は base だけのハッシュになるため、**両方の組み合わせを
#    候補にして、どちらかに一致すれば黙る**。こうすると設置形態を覚えておく必要が無い。
#
#    **止めない。** requirements のコメントを直しただけでもハッシュは変わるので、
#    そこで起動を拒むと、動く環境を塞ぐことになる。起動直前に出しておけば、
#    後から落ちたときに理由がすぐ上に出ている。
check_requirements_hash() {
  local venv_hash_file="$CORE_ROOT/.venv/.requirements.hash"
  [ -f "$venv_hash_file" ] || return 0
  local req_dir="$CORE_ROOT/api"
  local base="$req_dir/requirements.base.txt"
  local jobs="$req_dir/requirements.jobs.txt"
  [ -f "$base" ] || return 0

  local stored current_base current_full
  stored="$(cat "$venv_hash_file" 2>/dev/null || true)"
  current_base="$("$PY" - "$base" <<'PYEOF' 2>/dev/null || true
import hashlib, sys
h = hashlib.sha256()
for path in sys.argv[1:]:
    with open(path, "rb") as fp:
        h.update(fp.read())
print(h.hexdigest())
PYEOF
)"
  current_full="$current_base"
  if [ -f "$jobs" ]; then
    current_full="$("$PY" - "$base" "$jobs" <<'PYEOF' 2>/dev/null || true
import hashlib, sys
h = hashlib.sha256()
for path in sys.argv[1:]:
    with open(path, "rb") as fp:
        h.update(fp.read())
print(h.hexdigest())
PYEOF
)"
  fi

  [ -z "$stored" ] && return 0
  [ "$stored" = "$current_base" ] && return 0
  [ "$stored" = "$current_full" ] && return 0

  echo
  echo "warning: 依存（requirements）が、いま入っているものと違います。" >&2
  echo "         新しい依存が足りないと core は起動しません。次を実行してください:" >&2
  echo "           ./scripts/setup.sh" >&2
  echo
}
check_requirements_hash

export XIMA_WORKSPACES_ROOT="$WS_ROOT"

# 1) app dist の解決
#    優先順: XIMA_APP_DIST（明示） > <core>/app-dist（setup.sh が Releases から配置）
#            > app/dist（monorepo のローカルビルド）
APP_DIST="${XIMA_APP_DIST:-}"
USE_LOCAL_BUILD=0
if [ -z "$APP_DIST" ]; then
  if [ -f "$CORE_ROOT/app-dist/index.html" ]; then
    APP_DIST="$CORE_ROOT/app-dist"
  elif [ "$IS_MONOREPO" = "1" ]; then
    APP_DIST="$APP_SRC_DIR/dist"
    USE_LOCAL_BUILD=1
  else
    APP_DIST="$CORE_ROOT/app-dist"
  fi
fi

# monorepo で app/dist を使う場合のみビルドする（配布物には app/ が無い）。
if [ "$USE_LOCAL_BUILD" = "1" ]; then
  if [ "${XIMA_SKIP_BUILD:-0}" != "1" ]; then
    echo "[1/2] building app (npm run build)…"
    ( cd "$APP_SRC_DIR" && npm run build )
  else
    echo "[1/2] skip build (XIMA_SKIP_BUILD=1)"
  fi
else
  echo "[1/2] using prebuilt app: $APP_DIST"
fi

if [ ! -f "$APP_DIST/index.html" ]; then
  echo "error: app が見つかりません（探索: ${APP_DIST}）" >&2
  echo "       ./scripts/setup.sh を実行してください（UI 不要なら setup.sh --no-ui）。" >&2
  exit 1
fi

# 2) core API（local モード・app を同一オリジン配信・ジョブは in-process・フォアグラウンド）
echo "[2/2] starting xima (local) on http://127.0.0.1:${PORT}"
echo "      workspaces: $WS_ROOT"
echo "      jobs: in-process (backend=local, docker 不要)"
echo "      → ブラウザで http://127.0.0.1:${PORT} を開く"
cd "$CORE_ROOT/api"
XIMA_AGENT_AUTH_MODE=local \
XIMA_JOB_BACKEND="${XIMA_JOB_BACKEND:-local}" \
XIMA_APP_DIST="$APP_DIST" \
XIMA_SERVE_FILES_NATIVE=1 \
"$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
