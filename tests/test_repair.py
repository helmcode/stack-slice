"""Tests for recovering records from a gzip stream cut mid-member."""

import gzip
import json
import zlib

from stackslice.repair import iter_member_payloads, repair_file


def record(name, shard=1, unit_type="dockerfile"):
    return json.dumps({
        "unit_type": unit_type,
        "repo_path": name,
        "unit_prefix": "Dockerfile",
        "commit_id": f"sha-{name}",
        "shard": shard,
        "files": [{"path": "Dockerfile", "content": "FROM alpine\n"}],
    })


def complete_member(lines: list[str]) -> bytes:
    payload = ("\n".join(lines) + "\n").encode()
    compressor = zlib.compressobj(9, zlib.DEFLATED, 31)
    return compressor.compress(payload) + compressor.flush()


def truncated_member(lines: list[str]) -> bytes:
    """A member flushed with Z_SYNC_FLUSH and never terminated, as a kill leaves it."""
    payload = ("\n".join(lines) + "\n").encode()
    compressor = zlib.compressobj(9, zlib.DEFLATED, 31)
    return compressor.compress(payload) + compressor.flush(zlib.Z_SYNC_FLUSH)


def test_reads_a_single_clean_member(tmp_path):
    source = tmp_path / "a.jsonl.gz"
    source.write_bytes(complete_member([record("one"), record("two")]))
    result = repair_file(str(source), str(tmp_path / "a.repaired.jsonl.gz"))
    assert result["lines_recovered"] == 2
    assert result.get("members_truncated", 0) == 0


def test_recovers_data_written_after_a_truncated_member(tmp_path):
    """The wound must not hide the members that follow it."""
    source = tmp_path / "b.jsonl.gz"
    source.write_bytes(
        complete_member([record("seeded")])
        + truncated_member([record("killed-run-1"), record("killed-run-2")])
        + complete_member([record("after-1"), record("after-2"), record("after-3")])
    )

    # A standard reader stops at the wound and sees only part of the data.
    with gzip.open(source, "rb") as handle:
        try:
            naive = len(handle.read().splitlines())
        except (OSError, EOFError, zlib.error):
            naive = -1
    assert naive < 6, "the premise is that plain gzip cannot read past the wound"

    result = repair_file(str(source), str(tmp_path / "b.repaired.jsonl.gz"))
    assert result["lines_recovered"] == 6
    assert result["members_truncated"] >= 1

    with gzip.open(tmp_path / "b.repaired.jsonl.gz", "rt") as handle:
        names = [json.loads(line)["repo_path"] for line in handle]
    assert names == [
        "seeded", "killed-run-1", "killed-run-2", "after-1", "after-2", "after-3"
    ]


def test_output_is_a_single_readable_member(tmp_path):
    source = tmp_path / "c.jsonl.gz"
    source.write_bytes(
        truncated_member([record("x")]) + complete_member([record("y")])
    )
    destination = tmp_path / "c.repaired.jsonl.gz"
    repair_file(str(source), str(destination))
    with gzip.open(destination, "rt") as handle:
        assert len(handle.read().splitlines()) == 2


def test_drops_a_half_written_record(tmp_path):
    """A kill mid-line leaves invalid JSON, which must not reach the output."""
    payload = (record("good") + "\n" + record("cut")[:40]).encode()
    compressor = zlib.compressobj(9, zlib.DEFLATED, 31)
    source = tmp_path / "d.jsonl.gz"
    source.write_bytes(
        compressor.compress(payload) + compressor.flush(zlib.Z_SYNC_FLUSH)
    )
    result = repair_file(str(source), str(tmp_path / "d.repaired.jsonl.gz"))
    assert result["lines_recovered"] == 1
    assert result["lines_unparseable"] == 1


def test_deduplicates_identical_units(tmp_path):
    source = tmp_path / "e.jsonl.gz"
    source.write_bytes(
        complete_member([record("dup"), record("dup"), record("unique")])
    )
    result = repair_file(str(source), str(tmp_path / "e.repaired.jsonl.gz"))
    assert result["lines_recovered"] == 2
    assert result["lines_duplicate"] == 1


def test_counts_distinct_shards(tmp_path):
    source = tmp_path / "f.jsonl.gz"
    source.write_bytes(
        complete_member([record("a", shard=1), record("b", shard=2), record("c", shard=2)])
    )
    result = repair_file(str(source), str(tmp_path / "f.repaired.jsonl.gz"))
    assert result["shards_seen"] == 2


def test_line_split_across_members_is_stitched(tmp_path):
    """A record straddling a member boundary must survive as one line."""
    line = record("straddler")
    head, tail = line[:30], line[30:]
    source = tmp_path / "g.jsonl.gz"
    source.write_bytes(
        complete_member_raw(head.encode()) + complete_member_raw((tail + "\n").encode())
    )
    result = repair_file(str(source), str(tmp_path / "g.repaired.jsonl.gz"))
    assert result["lines_recovered"] == 1
    assert result.get("lines_unparseable", 0) == 0


def complete_member_raw(payload: bytes) -> bytes:
    compressor = zlib.compressobj(9, zlib.DEFLATED, 31)
    return compressor.compress(payload) + compressor.flush()


def test_empty_file_yields_nothing(tmp_path):
    source = tmp_path / "h.jsonl.gz"
    source.write_bytes(b"")
    result = repair_file(str(source), str(tmp_path / "h.repaired.jsonl.gz"))
    assert result["lines_recovered"] == 0
    assert list(iter_member_payloads(b"")) == []
