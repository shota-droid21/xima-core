"""ユーザ資産ファイルを壊さずに書き換えるための I/O。

なぜ必要か（#151）:
    `Path.write_text()` は「truncate してから書く」ため、書き込み中に中断すると
    **元の内容を失ったうえに、切り詰められた不正なファイルが残る**。
    `workspaces/` 配下は labels.json / label_schema.json といったユーザ資産であり、
    ここが壊れるとラベリング作業がまるごと失われる（Decision 002-A）。

    labels.json は実データで 2.7 MB あり、書き込みは一瞬ではない。中断の契機も
    現実に存在する（Ctrl+C で止める運用・Decision 036 / スリープ / ディスク満杯）。

やっていること:
    同一ディレクトリに一時ファイルを書き、fsync してから `os.replace` で差し替える。

    - **同一ディレクトリ**に作るのは、`os.replace` がアトミックなのが
      同一ファイルシステム内に限られるため。
    - **fsync** してから replace するのは、順序を保証しないと「rename は済んだが
      中身がまだディスクに無い」状態が起こりうるため。
    - 一時ファイル名を一意にするのは、同じ対象への並行書き込みが互いの一時ファイルを
      壊さないようにするため。

意図的にやっていないこと:
    親ディレクトリの fsync は行わない。これを省くと電源断で **rename 自体**が
    失われうるが、その場合に残るのは**古い正しいファイル**であって壊れたファイル
    ではない。守りたいのは「壊れないこと」なので、コストに見合わないと判断した。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """`path` を `text` で置き換える。失敗しても元のファイルは残る。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        # 中断・失敗時に一時ファイルを残さない（KeyboardInterrupt も拾う）。
        tmp_path.unlink(missing_ok=True)
        raise


def write_json_atomic(
    path: Path | str,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """JSON を `write_text_atomic` で書く。

    シリアライズを先に済ませるので、`data` が JSON にできない場合は
    **既存ファイルに一切触れずに**例外になる。
    """
    write_text_atomic(path, json.dumps(data, ensure_ascii=ensure_ascii, indent=indent))
