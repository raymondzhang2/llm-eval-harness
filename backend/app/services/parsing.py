"""Parse and validate uploaded datasets (CSV or JSONL) into item dicts."""

import csv
import io
import json
from dataclasses import dataclass

REQUIRED = ("input", "output")
OPTIONAL = "reference"


class DatasetParseError(ValueError):
    """Raised with a user-readable message when an upload is invalid."""


@dataclass
class ParsedItem:
    input: str
    output: str
    reference: str | None
    metadata: dict | None


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise DatasetParseError("File must be UTF-8 encoded text") from e


def _rows(filename: str, text: str) -> list[dict]:
    name = filename.lower()
    if name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            raise DatasetParseError("File is empty")
        missing = [c for c in REQUIRED if c not in reader.fieldnames]
        if missing:
            raise DatasetParseError(f"Missing required column(s): {', '.join(missing)}")
        return list(reader)
    if name.endswith((".jsonl", ".ndjson")):
        rows = []
        for n, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetParseError(f"Line {n}: invalid JSON") from e
            if not isinstance(row, dict):
                raise DatasetParseError(f"Line {n}: expected a JSON object")
            rows.append(row)
        return rows
    raise DatasetParseError("Unsupported file type; upload a .csv or .jsonl file")


def parse_dataset(filename: str, raw: bytes) -> list[ParsedItem]:
    text = _decode(raw)
    if not text.strip():
        raise DatasetParseError("File is empty")
    rows = _rows(filename, text)
    if not rows:
        raise DatasetParseError("File contains no items")

    items = []
    for n, row in enumerate(rows, start=1):
        for col in REQUIRED:
            value = row.get(col)
            if value is None or not str(value).strip():
                raise DatasetParseError(f"Item {n}: '{col}' is missing or empty")
        reference = row.get(OPTIONAL)
        reference = str(reference) if reference not in (None, "") else None
        extra = {k: v for k, v in row.items() if k not in (*REQUIRED, OPTIONAL) and k is not None}
        items.append(
            ParsedItem(
                input=str(row["input"]),
                output=str(row["output"]),
                reference=reference,
                metadata=extra or None,
            )
        )
    return items
