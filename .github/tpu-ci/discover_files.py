#!/usr/bin/env python3
"""Build CI matrix rows whose names are always derived from test file paths."""

import argparse
import json

from test_names import names_from_path


def parse_entry(value: str) -> dict:
    parts = value.split(",")
    path = parts[0]
    row = {"file": path, **names_from_path(path)}
    for item in parts[1:]:
        key, separator, metadata_value = item.partition("=")
        if not separator or not key:
            raise argparse.ArgumentTypeError(
                f"metadata must use key=value syntax: {item!r}"
            )
        row[key] = metadata_value
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", type=parse_entry, required=True)
    args = parser.parse_args()
    print(json.dumps(args.entry))


if __name__ == "__main__":
    main()
