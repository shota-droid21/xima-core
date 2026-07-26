#!/bin/sh
set -eu

# XIMA_AGENT_AUTH_MODE=open|external (default: open)
mode="${XIMA_AGENT_AUTH_MODE:-open}"
mode="$(printf '%s' "$mode" | tr '[:upper:]' '[:lower:]')"

case "$mode" in
  external|saas)
    src="/etc/nginx/xima/nginx.conf"
    ;;
  open|"")
    src="/etc/nginx/xima/nginx.open.conf"
    ;;
  *)
    echo "[nginx] Unknown XIMA_AGENT_AUTH_MODE='$mode'; defaulting to open" >&2
    src="/etc/nginx/xima/nginx.open.conf"
    ;;
esac

cp "$src" /etc/nginx/nginx.conf
