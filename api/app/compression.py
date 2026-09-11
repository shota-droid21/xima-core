"""応答を gzip する。**ただし中身を見て決める**（#363）。

画面は experiment を開くたびに labels.json を全件取る。実データ規模では
**5.28 MB が非圧縮で流れていた**。圧縮すると 0.71 MB になる（**7.4 分の 1**）。

| 回線 | 5.28 MB | 0.71 MB |
| --- | --- | --- |
| 20 Mbps | 2.1 s | 0.3 s |
| 5 Mbps | 8.4 s | 1.1 s |
| 2 Mbps | 21.1 s | 2.8 s |

app 側の timeout は 30 秒なので、**遅い回線では当たりうる距離にあった**。localhost
では 0.1 秒で終わるため体感できず、**別マシンの agent を使うときだけ効く**。

**Starlette の `GZipMiddleware` をそのまま入れてはいけない。** あれは
`content-type` を見ず（除外するのは `text/event-stream` だけ）、**サムネイルの
webp や jpeg まで圧縮しようとする**。既に圧縮済みのものを再圧縮しても縮まらず、
CPU だけ使う。xima はサムネイルを大量に配信するので、ここは無視できない。

そこで**圧縮してよい型を挙げて、それ以外はやらない**。許可制にするのは、
`application/octet-stream`（利用者の実ファイル）のような**中身の分からないものを
既定で圧縮しない**ためである。

**圧縮率は level 6 にする。** 実データで測った結果:

| level | 後 | 倍率 | 圧縮時間 |
| --- | --- | --- | --- |
| 1 | 0.92 MB | 5.7x | 16 ms |
| **6** | **0.71 MB** | **7.4x** | **42 ms** |
| 9（Starlette の既定） | 0.70 MB | 7.6x | 69 ms |

9 は 6 より **27 ms 余分に使って 1% しか縮まない**。6 で止める。

**nginx とは二重にならない。** `core/nginx/*.conf` に `gzip` の指定は無く、圧縮は
ここだけで行う。仮に nginx 側を有効にしても、既に `Content-Encoding` が付いた応答は
nginx が再圧縮しない。
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware, GZipResponder, IdentityResponder
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: これより小さい応答は圧縮しない。gzip のヘッダだけで 20 バイト近くあり、
#: `/health` のような小さな応答では割に合わない。
DEFAULT_MINIMUM_SIZE = 1024

#: 実データで測って決めた（モジュールの docstring）。
DEFAULT_COMPRESS_LEVEL = 6

#: 丸ごと圧縮してよい型。
_COMPRESSIBLE_PREFIXES = ("text/",)

#: `text/` ではないが圧縮したい型。
_COMPRESSIBLE_TYPES = frozenset(
    {
        "application/json",
        "application/javascript",
        "application/xml",
        "application/xhtml+xml",
        "application/manifest+json",
        "application/wasm",
        "image/svg+xml",  # 画像だが中身は XML である
    }
)

#: `+json` / `+xml` で終わる型は、名前が何であれ構造化テキストである。
_COMPRESSIBLE_SUFFIXES = ("+json", "+xml")


def is_compressible(content_type: str) -> bool:
    """その `Content-Type` を圧縮してよいか。

    **知らない型は圧縮しない。** `application/octet-stream` は利用者の実ファイルに
    使われており、中身は画像かもしれない。既に圧縮済みのものを再圧縮しても縮まらず、
    CPU だけ使う。
    """
    base = (content_type or "").split(";", 1)[0].strip().lower()
    if not base:
        return False
    if base.startswith(_COMPRESSIBLE_PREFIXES):
        return True
    if base in _COMPRESSIBLE_TYPES:
        return True
    return base.endswith(_COMPRESSIBLE_SUFFIXES)


class _TypeAwareGZipResponder(GZipResponder):
    """`content-type` を見て、圧縮しないものに印を付ける。

    親は `http.response.start` を**記録するだけで送らない**ので、記録させたあとに
    印を上書きできる。圧縮そのものは親の実装をそのまま使う —— streaming の扱いを
    書き直す理由が無い。

    **親の内部名に触っている。** Starlette を上げたときに黙って壊れないよう、
    `test_compression.py` が「画像が圧縮されないこと」を検査している。
    """

    async def send_with_compression(self, message: Message) -> None:
        await super().send_with_compression(message)
        if message["type"] != "http.response.start":
            return
        content_type = Headers(raw=message["headers"]).get("content-type", "")
        if not is_compressible(content_type):
            self.content_type_is_excluded = True


class TypeAwareGZipMiddleware(GZipMiddleware):
    """`GZipMiddleware` と同じだが、**圧縮してよい型だけ**を圧縮する。"""

    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int = DEFAULT_MINIMUM_SIZE,
        compresslevel: int = DEFAULT_COMPRESS_LEVEL,
    ) -> None:
        super().__init__(app, minimum_size=minimum_size, compresslevel=compresslevel)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        accept_encoding = Headers(scope=scope).get("Accept-Encoding", "")
        if "gzip" in accept_encoding:
            responder: ASGIApp = _TypeAwareGZipResponder(
                self.app, self.minimum_size, compresslevel=self.compresslevel
            )
        else:
            # 圧縮しないときも `Vary: Accept-Encoding` は要る（親と同じ扱い）。
            responder = IdentityResponder(self.app, self.minimum_size)

        await responder(scope, receive, send)
