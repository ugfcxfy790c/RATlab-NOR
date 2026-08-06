"""
Writes a minimal solid-color placeholder PNG with no external
dependencies (Pillow may not be installed on whatever machine is doing
the build) -- used by build_linux.sh for the AppImage's required icon
file, so the Linux build doesn't hard-fail just because nobody's made a
real icon yet. Safe to ignore/replace: drop a real packaging/icon.png
and the build scripts will use that instead of generating this.
"""

from __future__ import annotations

import struct
import sys
import zlib


def write_png(path: str, size: int = 256, rgba: tuple[int, int, int, int] = (70, 110, 180, 255)) -> None:
    width = height = size
    row = bytes(rgba) * width
    raw = b"".join(b"\x00" + row for _ in range(height))  # filter byte 0 ("None") per scanline
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))  # 8-bit RGBA
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(png)


if __name__ == "__main__":
    write_png(sys.argv[1] if len(sys.argv) > 1 else "icon.png")
