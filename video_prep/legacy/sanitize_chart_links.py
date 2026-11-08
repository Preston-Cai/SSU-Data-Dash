"""Remove dead external Excel links while retaining complete chart caches."""

from __future__ import annotations

import sys
import zipfile
import re
from decimal import Decimal
from pathlib import Path


def convert_chart_xml(data: bytes) -> bytes:
    """Preserve Office prefixes while converting complete caches to literals."""
    text = data.decode("utf-8")
    text = re.sub(
        r"<c:externalData\b[^>]*>.*?</c:externalData>",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"<c:tx><c:strRef>.*?<c:strCache>.*?<c:pt\b[^>]*><c:v>(.*?)</c:v></c:pt>.*?</c:strCache></c:strRef></c:tx>",
        lambda match: f"<c:tx><c:v>{match.group(1)}</c:v></c:tx>",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"<c:strRef>.*?<c:strCache>(.*?)</c:strCache></c:strRef>",
        lambda match: f"<c:strLit>{match.group(1)}</c:strLit>",
        text,
        flags=re.DOTALL,
    )

    def numeric_literal(match: re.Match[str]) -> str:
        body = match.group(1)

        def normalize_value(value_match: re.Match[str]) -> str:
            normalized = Decimal(value_match.group(1)).quantize(Decimal("0.0000000001"))
            return f"<c:v>{format(normalized.normalize(), 'f')}</c:v>"

        body = re.sub(r"<c:v>([-+0-9.Ee]+)</c:v>", normalize_value, body)
        return f"<c:numLit>{body}</c:numLit>"

    text = re.sub(
        r"<c:numRef>.*?<c:numCache>(.*?)</c:numCache></c:numRef>",
        numeric_literal,
        text,
        flags=re.DOTALL,
    )
    return text.encode("utf-8")


def main(source: str, destination: str) -> None:
    src = Path(source).resolve()
    dst = Path(destination).resolve()
    if dst.exists():
        raise FileExistsError(dst)

    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(
        dst, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)

            if info.filename.startswith("ppt/charts/chart") and info.filename.endswith(".xml"):
                data = convert_chart_xml(data)

            if info.filename.startswith("ppt/charts/_rels/chart") and info.filename.endswith(".rels"):
                continue

            zout.writestr(info, data)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: sanitize_chart_links.py SOURCE DESTINATION")
    main(sys.argv[1], sys.argv[2])
