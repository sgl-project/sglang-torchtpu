# Long-context dispatch audit: context capacity changes the kernel path

No code/config/remote job changed.

Production plain-token prefill dispatch in mla_sparse_fused.py:304–336 requires
num_tokens>=QBLOCK_MIN_TOKENS, token count divisible128, power-of-two page size
multiple128, and static pages_per_seq<=256. Otherwise it falls back to the
per-query fused kernel. This limit is on rectangular metadata capacity, not
the live kv_lens or selected topk2048 token count.

attention_pallas_rpa_base.py:594–596 sets max_context_len to configured model
context + request-table slack; lines914 and995–1011 construct the rectangular
page table with ceil(max_context_len/page_stride). No DCP is enabled here,
so page_stride=page_size.

Consequences (non-spec slack4 as in accepted smoke):
- accepted context4096/page128 ->33 pages: warmup2048 eligible for qblock;
- minimum frozen context61456/page128 ->481 pages: qblock ineligible;
- historical context135168/page64 ->2113 pages, and page64 independently
  fails the multiple128 predicate: per-query path;
- context65536/page256 ->257 pages: STILL ineligible because slack crosses
  the256-page boundary (naive65536/256 calculation misses it);
- context61456/page256 ->241 pages: statically eligible, but page256 layout,
  larger metadata and two-request local batches are NOT hardware validated
  by the accepted H64/T2048/page128/33-page kernel gates.

Thus simply restoring long context on the accepted page128 config silently
changes from qblock warmup to per-query prefill. Increasing page size to256
may retain qblock but is a new kernel/layout experiment, not a safe equivalent
configuration. Do not bypass the256-page guard or change page size without a
bounded compiler/numerical gate. Matched performance planning must state
which path is being measured and cannot attribute it solely to the selector
fix. First establish reference context/page semantics and precision; then
choose an explicit minimal path-specific diagnostic.
