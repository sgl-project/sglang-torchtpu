"""End-to-end TPU smoke test for a tiny *dense* hybrid Qwen3.5 text model.

Builds a small dense (non-MoE) Qwen3.5 text config — a few Gated-DeltaNet
linear-attention layers interleaved with full-attention layers, dense MLP —
writes it to a temp model dir, and runs the SGLang offline ``Engine`` with
random weights (``load_format="dummy"``) on TPU. No checkpoint / tokenizer /
network needed: token IDs are passed directly via ``skip_tokenizer_init``.

Uses ``model_type="qwen3_5_text"`` + architecture ``Qwen3_5ForCausalLM`` — the
text-only dense backbone (dense MLP is selected for that model_type), which
avoids the multimodal VL wrapper's image processor. This is the smallest
vehicle that exercises the GDN path end-to-end on device: the TPU GDN Pallas
kernel + bridge op, the TPU recurrent state cache, and the RPA v3 full-attention
backend, composed by ``HybridLinearAttnBackend``.

Success = the engine starts, runs prefill + several decode steps, and emits
finite, in-range token IDs without crashing (dummy weights => output *quality*
is not checked).

Requires a free TPU. Run directly:

    SGLANG_PLATFORM=tpu python test/manual/tpu/smoke_qwen35_tpu_dummy.py
"""

import json
import os
import tempfile

# head_dim for the GDN linear heads must be a multiple of the TPU lane count
# (128) for the kernel's strided VMEM loads; keep those at 128 while everything
# else stays tiny for fast iteration.
TINY_CONFIG = {
    "architectures": ["Qwen3_5TextForCausalLM"],
    "model_type": "qwen3_5_text",  # dense MLP selected for this model_type
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "full_attention_interval": 2,  # layers 1,3 full-attn; 0,2 GDN
    "attn_output_gate": True,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000.0,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "num_experts": None,  # dense
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "max_position_embeddings": 4096,
    "torch_dtype": "bfloat16",
}
TINY_TEXT_CONFIG = TINY_CONFIG  # vocab range used in the output-range check


def _write_model_dir(path: str):
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(TINY_CONFIG, f, indent=2)


def main():
    import sglang as sgl

    with tempfile.TemporaryDirectory() as model_dir:
        _write_model_dir(model_dir)

        engine = sgl.Engine(
            model_path=model_dir,
            device="tpu",
            dtype="bfloat16",
            load_format="dummy",
            skip_tokenizer_init=True,
            disable_radix_cache=True,  # recurrent state has no prefix sharing yet
            attention_backend="pallas_tpu_rpa_v3",
            trust_remote_code=True,
            mem_fraction_static=0.3,
            max_running_requests=4,
            context_length=512,
        )
        try:
            prompt_ids = [1, 5, 9, 13, 17, 21, 25, 29]
            out = engine.generate(
                input_ids=[prompt_ids],
                sampling_params={"max_new_tokens": 16, "temperature": 0.0},
            )
            result = out[0] if isinstance(out, list) else out
            token_ids = result.get("token_ids") or result["output_ids"]
            print("Generated token ids:", token_ids)
            vocab = TINY_TEXT_CONFIG["vocab_size"]
            assert len(token_ids) > 0, "no tokens generated"
            assert all(
                0 <= t < vocab for t in token_ids
            ), f"token id out of range [0,{vocab})"
            print("SMOKE PASS: tiny dense hybrid Qwen3.5 ran E2E on TPU.")
        finally:
            engine.shutdown()


if __name__ == "__main__":
    main()
