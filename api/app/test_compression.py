"""応答の圧縮（#363）。

**この検査が守るもの。**

1. 大きい JSON は圧縮される（これが目的）
2. **画像は圧縮されない。** Starlette の `GZipMiddleware` は `content-type` を見ないので、
   そのまま入れるとサムネイルまで再圧縮する。**ここが落ちたら、その事故が起きている**
3. 小さい応答は圧縮しない
4. `Accept-Encoding` に gzip が無ければ圧縮しない
5. 中身が変わらない（解凍すると元と同じ）
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.testclient import TestClient

from .compression import (
    DEFAULT_COMPRESS_LEVEL,
    DEFAULT_MINIMUM_SIZE,
    TypeAwareGZipMiddleware,
    is_compressible,
)

BIG = {"items": [{"id": i, "path": f"images/{i:05}.jpg"} for i in range(2000)]}


def _app() -> TestClient:
    app = FastAPI()
    app.add_middleware(TypeAwareGZipMiddleware)

    @app.get("/big-json")
    def big_json():
        return JSONResponse(BIG)

    @app.get("/small-json")
    def small_json():
        return JSONResponse({"status": "ok"})

    @app.get("/thumb")
    def thumb():
        # 実際には webp のバイト列。ここでは「大きくて image/* である」ことが要点。
        return Response(content=b"\x00" * 200_000, media_type="image/webp")

    @app.get("/native-file")
    def native_file():
        return Response(content=b"\x00" * 200_000, media_type="application/octet-stream")

    @app.get("/page")
    def page():
        return Response(content="<html>" + "a" * 50_000 + "</html>", media_type="text/html")

    @app.get("/stream")
    def stream():
        def chunks():
            for _ in range(50):
                yield b"x" * 1000

        return StreamingResponse(chunks(), media_type="application/json")

    return TestClient(app)


GZIP = {"Accept-Encoding": "gzip"}
NO_GZIP = {"Accept-Encoding": "identity"}


# --- 1. 大きい JSON は圧縮される -----------------------------------------------


def test_big_json_is_compressed():
    res = _app().get("/big-json", headers=GZIP)
    assert res.headers.get("content-encoding") == "gzip"
    assert res.json() == BIG, "解凍すると元と同じ"


def test_compression_actually_shrinks_it():
    client = _app()
    plain = client.get("/big-json", headers=NO_GZIP)
    packed = client.get("/big-json", headers=GZIP)
    raw = len(plain.content)
    sent = int(packed.headers["content-length"])
    assert sent * 4 < raw, f"{raw} -> {sent} では縮んでいない"


def test_html_is_compressed():
    res = _app().get("/page", headers=GZIP)
    assert res.headers.get("content-encoding") == "gzip"


def test_streaming_json_is_compressed():
    res = _app().get("/stream", headers=GZIP)
    assert res.headers.get("content-encoding") == "gzip"
    assert res.content == b"x" * 50_000


# --- 2. 画像は圧縮されない（事故の検知） ---------------------------------------


def test_images_are_not_compressed():
    """**Starlette の GZipMiddleware をそのまま入れると、ここが落ちる。**

    あちらは `content-type` を見ない（除外は `text/event-stream` のみ）。xima は
    サムネイルを大量に配信するので、既に圧縮済みのものを再圧縮して CPU だけ使う。
    """
    res = _app().get("/thumb", headers=GZIP)
    assert res.headers.get("content-encoding") is None
    assert len(res.content) == 200_000


def test_unknown_binary_is_not_compressed():
    """`application/octet-stream` は利用者の実ファイル。**中身が分からないなら触らない。**"""
    res = _app().get("/native-file", headers=GZIP)
    assert res.headers.get("content-encoding") is None


# --- 3 / 4. 圧縮しない場面 ------------------------------------------------------


def test_small_response_is_not_compressed():
    res = _app().get("/small-json", headers=GZIP)
    assert res.headers.get("content-encoding") is None


def test_client_without_gzip_gets_plain_bytes():
    res = _app().get("/big-json", headers=NO_GZIP)
    assert res.headers.get("content-encoding") is None
    assert res.json() == BIG


def test_vary_is_set_either_way():
    """キャッシュが圧縮の有無を取り違えないようにする。"""
    client = _app()
    for headers in (GZIP, NO_GZIP):
        res = client.get("/big-json", headers=headers)
        assert "accept-encoding" in res.headers.get("vary", "").lower()


# --- 5. 型の判定 ---------------------------------------------------------------


def test_is_compressible_says_yes_to_text_and_structured():
    for value in (
        "application/json",
        "application/json; charset=utf-8",
        "text/html; charset=utf-8",
        "text/css",
        "text/javascript",
        "application/manifest+json",
        "image/svg+xml",
        "APPLICATION/JSON",
    ):
        assert is_compressible(value), value


def test_is_compressible_says_no_to_binary():
    for value in (
        "image/webp",
        "image/jpeg",
        "image/png",
        "video/mp4",
        "font/woff2",
        "application/zip",
        "application/pdf",
        "application/octet-stream",
        "",
    ):
        assert not is_compressible(value), value


def test_defaults_are_the_measured_ones():
    """docstring の実測と設定がずれないようにする。"""
    assert DEFAULT_COMPRESS_LEVEL == 6, "level 9 は 27 ms 余分に使って 1% しか縮まない"
    assert DEFAULT_MINIMUM_SIZE == 1024


# --- 6. 実際のアプリに入っていること -------------------------------------------


def test_the_real_app_compresses_json():
    from .main import app

    client = TestClient(app)
    res = client.get("/openapi.json", headers=GZIP)
    assert res.status_code == 200
    assert res.headers.get("content-encoding") == "gzip"
    assert json.loads(res.content)["openapi"]
