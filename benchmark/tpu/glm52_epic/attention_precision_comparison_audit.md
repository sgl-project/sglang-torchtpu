# Attention precision comparison gap (source audit, not device instruction census)

JAX reference result records FP8 KV, fp8_q=true, fp8_pv=true.
Current successful Torch probes allocate BF16 Q/QR and FP8 KV. The production
per-query path concatenates Q and QR, unpacks byte cache to float8_e4m3fn
(kernel.py:395–418), computes QK einsum with preferred_element_type=float32
(lines233–246), and passes unnormalized exponential P directly into PV
(lines252–269,600–618), again requesting float32 accumulation.

No explicit FP8 Q conversion or FP8 probability quantization is present in
that audited fallback attention path. This is a source-level numerical
contract gap versus the reference's recorded switches. XLA may insert casts
or choose implementation-specific dot operands; source alone does NOT prove
final MXU dtype/instruction cost. Runtime HLO/LLO or equivalent verified export
is needed before assigning actual instruction precision or speedup potential.

Therefore matching model checkpoint and FP8 KV is insufficient for an
apples-to-apples performance ratio. Do not silently turn on new Torch
quantization, loosen numeric tolerances, or reinterpret reference booleans as
verified device instructions. Options for a later controlled comparison are
an explicitly common precision mode on both frameworks, or a clearly labeled
native-mode comparison with numerical accuracy and operand-type evidence.
Neither option is implemented or measured here.
