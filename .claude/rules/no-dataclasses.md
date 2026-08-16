---
paths:
  - "**/*.py"
---

# Use `msgspec.Struct`, not `@dataclass`

Define data containers as `msgspec.Struct`. Do not add new
`dataclasses.dataclass` (or `attrs`) — they weaken strict type checking and
don't translate cleanly for multi-language support (e.g. the planned Rust migration).

```python
import msgspec

class LoadSnapshot(msgspec.Struct):   # prefer frozen= and omit_defaults=; kw_only= as needed
    dp_rank: int = 0
    tokens: list[int] = []            # mutable defaults are safe
```

- Methods / `@classmethod` constructors go on the `Struct`; see
  `python/sglang/srt/managers/load_snapshot.py`.
- New code only. Existing `@dataclass` is grandfathered — migrate opportunistically
  while editing the file, not in drive-by sweeps.
- **Exception — compiled regions.** A container constructed inside
  torch.compile-traced code (anything reachable from a fullgraph-compiled
  forward; on TPU no graph break is possible) must be a `NamedTuple` or plain
  tuple: Dynamo cannot instantiate `msgspec.Struct`
  (`object.__new__(X) is not safe`). Reading fields of a Struct built outside
  the traced region remains fine.
