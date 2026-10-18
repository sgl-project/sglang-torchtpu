"""Redact external infrastructure identifiers before displaying or storing pod logs.

GitHub's log masking does not sanitize the files uploaded to artifacts or GCS.
This filter covers configured cluster/project/nodepool values, GKE node hostnames,
and IP addresses. It is not a general-purpose credential or personal-data scanner.
"""

import ipaddress
import os
import re
import sys

GKE_NODE = re.compile(r"\bgke-[\w.-]+", re.IGNORECASE)
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
IPV6 = re.compile(r"(?<![\w:])[\da-fA-F]*:[\da-fA-F:.]+(?:%[\w.-]+)?")


def redact_ip(match):
    value = match.group()
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return value
    return "[REDACTED_IP]"


def sensitive_values(environ):
    return sorted(
        {
            value
            for key, value in environ.items()
            if value
            and (
                key in {"EXTERNAL_CLUSTER", "EXTERNAL_PROJECT"}
                or key.startswith("EXTERNAL_NODEPOOL_")
            )
        },
        key=len,
        reverse=True,
    )


def sanitize(line, values):
    # Match full node names before replacing a cluster/nodepool substring in them.
    line = GKE_NODE.sub("[REDACTED_NODE]", line)
    line = IPV6.sub(redact_ip, line)
    line = IPV4.sub(redact_ip, line)
    for value in values:
        line = line.replace(value, "[REDACTED]")
    return line


def main():
    values = sensitive_values(os.environ)
    for line in sys.stdin:
        print(sanitize(line, values), end="", flush=True)


if __name__ == "__main__":
    main()
