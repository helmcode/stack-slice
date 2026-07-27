"""Recover JSONL records from a gzip file whose stream was cut mid-member.

`gzip.open(path, "at")` appends a new member per run, and `GzipFile.flush()`
writes a zlib sync point without terminating the member: no CRC, no ISIZE
trailer. Kill the writer and that member stays unterminated. Any later append
lands behind the wound, and every standard reader stops at it, so the bytes after
it look lost when they are merely unreachable.

This walks the file member by member. Whatever a member yields before it breaks
is kept, then the scan resyncs on the next gzip magic and carries on. Lines are
validated as JSON, which discards the half-written record at each wound.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import zlib
from collections import Counter

GZIP_MAGIC = b"\x1f\x8b\x08"
CHUNK = 1 << 22


def _next_magic(data: bytes, start: int) -> int:
    """Offset of the next plausible gzip member header, or -1."""
    return data.find(GZIP_MAGIC, start)


def _refine_to_failing_byte(data: bytes, position: int, failing_chunk: int):
    """Replay a member, salvaging every byte up to the exact point it breaks.

    zlib gives no partial output from a call that raises, so a coarse chunk loses
    everything it was carrying. The member is replayed from its start: one call
    for the range already known to be good, then byte at a time through the
    chunk that failed.
    """
    decompressor = zlib.decompressobj(wbits=31)
    output = bytearray()
    if failing_chunk > position:
        output += decompressor.decompress(data[position:failing_chunk])
        if decompressor.eof:
            # `unused_data` counts only what was fed, never the whole file.
            return bytes(output), False, failing_chunk - len(decompressor.unused_data)

    cursor = failing_chunk
    while cursor < len(data):
        try:
            output += decompressor.decompress(data[cursor:cursor + 1])
        except zlib.error:
            break
        cursor += 1
        if decompressor.eof:
            return bytes(output), False, cursor
    return bytes(output), True, cursor


def _decompress_member(data: bytes, position: int, chunk: int = CHUNK):
    """Return (payload, truncated, consumed_offset) for the member at `position`."""
    decompressor = zlib.decompressobj(wbits=31)
    output = bytearray()
    cursor = position
    while cursor < len(data):
        fed_end = min(cursor + chunk, len(data))
        try:
            output += decompressor.decompress(data[cursor:fed_end])
        except zlib.error:
            return _refine_to_failing_byte(data, position, cursor)
        if decompressor.eof:
            # `unused_data` holds the tail of what was FED, not of the file, so
            # the consumed offset is relative to fed_end. Using len(data) here
            # pushed the resync point to the end of the file and silently hid
            # every member after the first.
            return bytes(output), False, fed_end - len(decompressor.unused_data)
        cursor = fed_end

    # Input exhausted with no trailer: the file itself ends mid-member.
    try:
        output += decompressor.flush()
    except zlib.error:
        pass
    return bytes(output), True, len(data)


def iter_member_payloads(data: bytes, chunk: int = CHUNK):
    """Yield (decompressed_bytes, member_offset, was_truncated) per member."""
    position = _next_magic(data, 0)
    while position != -1 and position < len(data):
        payload, truncated, consumed = _decompress_member(data, position, chunk)
        yield payload, position, truncated
        # Resync from where decompression stopped: for a wound, that is normally
        # the first byte of the next member's header, since reading that header
        # as deflate data is what broke the stream.
        position = _next_magic(data, max(position + 1, consumed - 64))


def repair_file(source: str, destination: str) -> dict:
    """Rewrite `source` as a single clean gzip member, keeping every good line."""
    with open(source, "rb") as handle:
        data = handle.read()

    stats: Counter = Counter()
    shards: set[int] = set()
    seen: set[tuple] = set()
    leftover = b""

    with gzip.open(destination, "wt", encoding="utf-8") as out:
        for payload, offset, truncated in iter_member_payloads(data):
            stats["members"] += 1
            if truncated:
                stats["members_truncated"] += 1
            if not payload:
                stats["members_empty"] += 1
                continue

            # A member boundary can split a line; stitch the tail onto the head
            # of the next member before parsing.
            buffer = leftover + payload
            lines = buffer.split(b"\n")
            leftover = lines.pop() if not buffer.endswith(b"\n") else b""

            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    stats["lines_unparseable"] += 1
                    continue
                key = (
                    record.get("repo_path"),
                    record.get("unit_prefix"),
                    record.get("unit_type"),
                    record.get("commit_id"),
                )
                if key in seen:
                    stats["lines_duplicate"] += 1
                    continue
                seen.add(key)
                if isinstance(record.get("shard"), int):
                    shards.add(record["shard"])
                out.write(line.decode("utf-8", "replace") + "\n")
                stats["lines_recovered"] += 1

        if leftover.strip():
            stats["lines_unparseable"] += 1

    return {
        "source": os.path.basename(source),
        "source_bytes": len(data),
        "output_bytes": os.path.getsize(destination),
        "shards_seen": len(shards),
        "members": stats["members"],
        "members_truncated": stats["members_truncated"],
        "members_empty": stats["members_empty"],
        "lines_recovered": stats["lines_recovered"],
        "lines_duplicate": stats["lines_duplicate"],
        "lines_unparseable": stats["lines_unparseable"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="damaged .jsonl.gz files")
    parser.add_argument("--suffix", default=".repaired", help="output suffix")
    args = parser.parse_args()

    for source in args.files:
        destination = source.replace(".jsonl.gz", f"{args.suffix}.jsonl.gz")
        result = repair_file(source, destination)
        print(json.dumps(result), flush=True)
        print(
            f"  {result['source']:<32} "
            f"{result['lines_recovered']:>10,} lines  "
            f"{result['members']} members "
            f"({result.get('members_truncated', 0)} truncated)  "
            f"{result['shards_seen']:>5} shards  "
            f"dupes={result.get('lines_duplicate', 0):,}  "
            f"bad={result.get('lines_unparseable', 0):,}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
