# Inkling2 Architecture Guide

Inkling2 keeps Inkling's decoder-only backbone, learned relative attention, short causal convolutions, and shared-sink MoE routing. The main changes are Block Attention Residuals with separate full-attention Q/K/V inputs, gated attention with a learned sink, dynamic K/V convolutions, a Perception Encoder–based image encoder, and a DSpark-style MTP draft.

Inkling2 dimensions below describe Inkling2-Small. See [PROMPTING.md](PROMPTING.md) for the message format and [README.md](README.md) for serving instructions.

## Overview

The outer model structure is unchanged:

```text
normalized token embeddings and image embeddings
  -> repeated attention and MoE sublayers
  -> final RMSNorm
  -> LM head
```

Inkling's ordinary residual additions are replaced by Block Attention Residuals (Block AttnRes): addition within blocks, with learned attention over block representations before each sublayer.

```text
completed residual blocks + current partial block
  -> Block AttnRes -> RMSNorm
  -> attention -> attention-output sconv
  -> Block AttnRes including the attention output -> RMSNorm
  -> MoE -> MoE-output sconv
  -> update the current residual block

after the last layer:
  -> final Block AttnRes -> final RMSNorm -> LM head
```

The table compares the public [Inkling-Small configuration](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/config.json) with Inkling2-Small:

| Component | Inkling-Small | Inkling2-Small |
| --- | --- | --- |
| Decoder layers / hidden width | 42 / 4096 | 48 / 4608 |
| Full attention Q / KV heads | 32 / 8 | 48 / 8 |
| Local attention Q / KV heads | 32 / 8 | 48 / 8 |
| Attention head width | 128 | 128, unchanged |
| Local attention window | 512 tokens | 1024 tokens |
| Full relative extent / relative feature width | 1024 / 16 | 1024 / 16, unchanged |
| Full attention placement | Every sixth layer | Every sixth layer, unchanged |
| Routed / active routed / shared experts | 256 / 6 / 2 | 382 / 6 / 2 |
| Expert intermediate width | 2048 | 2304 |
| Feed-forward layout | Two initial dense layers, then MoE | MoE in every layer |
| Image encoder | Per-patch HMLP | 23-layer PE-Lang-style vision transformer |
| MTP | Successive prediction-depth layers | Five-layer draft proposing a block of seven tokens |

Both configurations retain a 1,048,576-token target context limit, length-scaling floor of 128,000, and length-scaling coefficient of 0.1. The optional Inkling2 draft has its own shorter context limit.

## Block Attention Residuals

Inkling2 uses the Block AttnRes formulation in [Attention Residuals](https://arxiv.org/pdf/2603.15031#page=4) (§3.2, Figure 2). Each attention or MLP sublayer has a learned **pseudo-query** that attends over completed block representations and the current intra-block partial sum. This is attention across depth, independently at each token position.

As in the paper, scoring uses RMS-normalized keys while values remain unnormalized in Inkling2-Small. For one Inkling2 pseudo-query of width `H`:

```python
keys = sources / sqrt(mean(sources**2, dim=-1, keepdim=True) + 1e-6)
scores = keys @ (query / sqrt(H))
weights = softmax(scores, dim="sources")
mixture = sum(weights[i] * sources[i] for i in range(len(sources)))
```

Here, `n_layers_per_attnres_block=6` counts complete transformer layers, or 12 attention/MLP sublayers in the paper's notation. The 48-layer configuration has eight blocks plus the embedding source. The final output aggregation follows Block AttnRes as well.

### Inkling2's Q/K/V Extension

The main departure from the paper is in full attention: [Inkling2](layers/attn_res.py) uses **three pseudo-queries**, producing separate Q, K, and V inputs, each with its own input norm. The paper supplies one aggregated input to an attention sublayer. Local attention and MoE retain one pseudo-query each.

```text
residual sources
  -> Q mixture -> Q input norm -> Q, R, output gate, sink projections
  -> K mixture -> K input norm -> K projection and K convolution weights
  -> V mixture -> V input norm -> V projection and V convolution weights
```

Q, K, and V can therefore draw different proportions of information from earlier residual blocks. R, the output gate, and the sink use the Q-side source.

## Normalization And L2 Clipping

Inkling2 retains embedding RMSNorm and pre-norm attention/MoE sublayers. Its sublayer and final norms additionally support a learned scale **before** RMS normalization, enabled in Inkling2-Small:

```python
u = pre_weight * x
y = weight * u / sqrt(mean(u**2) + eps)
```

The pre-scale changes each channel's contribution to the denominator. The embedding norm remains an ordinary RMSNorm.

Within attention, Inkling's Q/K RMSNorm is replaced by **L2 clipping**. Q and R first apply learned per-head, per-channel scales; K has no such learned scale. Each vector is then clipped only if its norm exceeds its threshold:

```python
u = learned_scale * x  # Q and R; use x directly for K
y = u * min(1.0, threshold / (l2_norm(u) + 1e-6))
```

Clipping preserves vectors below the threshold. Inkling2-Small's Q/K threshold is approximately 22.6274, and the R threshold is 8. K clipping follows K sconv; optional V normalization is disabled in this configuration.

## Relative Attention And Attention Layout

**Same as Inkling:** the target uses learned relative logits rather than RoPE or absolute position embeddings. A query-side feature `R` of width `d_rel` produces one learned logit per causal distance and head:

```text
rel_logits[i, h, d] = R[i, h] @ rel_projection[:, d]

score(i, j, h) = dot(Q[i, h], K[j, h]) / sqrt(head_dim)
                 + rel_logits[i, h, i - j]
```

Outside the relative extent, the relative term is zero; full attention still reaches older tokens through the content score. **The content-score scale changes:** Inkling used `1 / head_dim` after Q/K RMSNorm; Inkling2 uses `1 / sqrt(head_dim)` with its L2-clipped vectors.

The stack still repeats five local causal layers followed by one full causal layer. Local attention includes the current token and the previous `sliding_window_size - 1` tokens; its relative extent matches that window.

Full layers retain Inkling's length-dependent scaling:

```python
tau = 1 + alpha * log(max((position + 1) / n_floor, 1))
```

Inkling2 applies `tau` to the clipped Q and R vectors in full-attention layers only.

### Attention Output Gate And Sink

**New relative to Inkling:** the elementwise sigmoid gate uses the post-attention, pre-output-projection placement called `G1` in [Gated Attention for Large Language Models](https://arxiv.org/html/2505.06708v1#S2). Inkling2 additionally projects a query-dependent sink logit per head from its Q-side source. With attention output `output` and score log-sum-exp `lse`:

```python
output = output * sigmoid(gate)
output = output * sigmoid(lse - sink)
```

The sink factor acts like an extra attention entry with logit `sink` and a zero value: it absorbs probability mass. It is query-conditioned and distinct from the shared-expert sink in MoE routing.

The complete attention sublayer is:

```text
normalized source(s)
  -> Q/K/V/R, gate, and sink projections
  -> K sconv and V sconv
  -> Q/K/R L2 clipping, optional V norm, full-layer length scaling
  -> relative attention, returning output and log-sum-exp
  -> elementwise gate and sink scaling
  -> output projection -> attention-output sconv
```

## Sconv

**Same placement as Inkling:** each decoder layer has four short causal convolution streams: K, V, attention output, and MoE output. Inkling2-Small's kernel width is still four, and each stream caches its previous three input states per request:

```text
[num_request_slots, kernel_size - 1, stream_width]
```

Local K/V and both output streams use static per-channel convolution weights. Inkling2 includes a fixed output scale, so the static operation for channel `c` is:

```python
y[t, c] = (x[t, c] + sum(w[c, W - 1 - lag] * x[t - lag, c] for lag in range(W))) / sqrt(W + 1)
```

Missing positions contribute zero. The fixed divisor is new relative to Inkling's unscaled sconv.

### Dynamic Convolutions In Full Attention

**Different from Inkling:** full-attention K/V use dynamic weights in addition to static channel weights. The normalized K-side or V-side source produces `W` additional weights for each KV head at each current token:

```python
dynamic[t, head, :] = weight_proj(source[t])
weight[t, head, channel, lag] = static[lag, head, channel] + dynamic[t, head, lag]
y[t, head, channel] = (
    x[t, head, channel]
    + sum(weight[t, head, channel, lag] * x[t - lag, head, channel] for lag in range(W))
) / sqrt(2 * W + 1)
```

The dynamic weights are shared within each KV head; static weights remain channel-specific. K and V have independent projections. The caches still hold the previous input states.

## MoE And Shared Expert Sink

**Same routing rule as Inkling:** the router scores routed and shared experts, selects top-k only among routed experts, and normalizes the selected routed experts together with every shared expert. Gate bias changes selection, not mixture weights:

```python
scores = router(x)
routed_ids = top_k(sigmoid(scores[:num_routed]) + gate_bias, k)
active_scores = concat(scores[routed_ids], scores[num_routed:])
weights = softmax(logsigmoid(active_scores)) * route_scale

output = sum(weights[i] * active_expert[i](x) for i in range(k + num_shared))
```

Shared experts remain always active but compete with the selected routed experts for mixture weight. Both use SwiGLU MLPs.

**Different from Inkling:** every Inkling2 decoder layer uses MoE, including layer 0. Inkling2-Small selects six of 382 routed experts and uses two shared experts. It retains `route_scale=8`, but does not apply Inkling's additional learned global router scale.

## Image Embeddings

**Same interface as Inkling:** image embeddings replace image placeholder positions and pass through the same decoder as text.

**Different encoder:** Inkling's HMLP encoded each patch independently. Inkling2 follows the language-aligned vision-transformer design from [Perception Encoder (PE)](https://arxiv.org/html/2504.13181v1#S4), allowing each patch to attend across its image before entering the language model:

```text
preprocessed image patches
  -> pixel renormalization and temporal mean
  -> native-patch linear embedding + learned position embeddings + CLS token
  -> pre-LayerNorm
  -> vision transformer layers with bidirectional attention
  -> remove CLS and concatenate neighboring spatial features
  -> linear -> GELU -> linear to decoder hidden width
```

### Native Patches And Decoder Tokens

The processor supplies BF16 patches with shape:

```text
[num_image_tokens, temporal_patch_size,
 native_patch_size * adapter_merge_size,
 native_patch_size * adapter_merge_size, 3]
```

For Inkling2-Small, that is `[num_image_tokens, 2, 40, 40, 3]`. The temporal dimension contains two copies of the same still image, averaged by the encoder.

Each 40×40 patch is split into four native 20×20 patches. The image grid describes the **merged decoder-token grid**, so an image with grid `[1, h, w]` has `h * w` decoder tokens but `4 * h * w + 1` vision-transformer positions, including CLS.

### PE Backbone And Inkling2 Adaptations

The backbone matches the [public PE-Lang-L configuration](https://github.com/facebookresearch/perception_models/blob/65a4a17a587b6a89eaa0dc08f231c2cb1c285f9f/core/vision_encoder/config.py#L169-L177) in width (1024), depth (23), heads (16), and MLP ratio (4). It retains PE's CLS token, learned 32×32 position grid interpolated to each image, 2D RoPE, pre-LayerNorm, GELU MLPs, and LayerScale. Like PE-Lang, it outputs patch features without a final norm or global pooling head.

The [Inkling2 encoder](vision.py) adapts this design for its decoder:

- **Patch size:** native patches are 20×20 instead of PE-Lang-L's 14×14.
- **Decoder adapter:** after removing CLS, neighboring 2×2 features are concatenated. A `4096 -> 4096 -> 4608` GELU MLP produces one embedding per 40×40 image placeholder.
- **Input interface:** Inkling's patch format is retained, with pixel values renormalized to PE's `[-1, 1]` convention and duplicate temporal slices averaged before encoding.

The architecture is specified in a flat `vision_config`. Inkling2 supports images; Inkling's audio path is not included.

## MTP And Speculative Decoding

**Different from Inkling's successive prediction-depth layers:** Inkling2 uses the RNN variant of [DSpark's semi-autoregressive design](https://arxiv.org/html/2607.05147v1#S3.SS1). A parallel transformer backbone computes base logits for a block; a lightweight RNN then conditions proposals on earlier sampled tokens. The target verifies the proposals.

```text
anchor token embedding + seven learned mask embeddings
  -> five-layer draft transformer attending to target-derived context
  -> final norm -> shared target LM head -> base logits
  -> recurrent hidden correction and sequential proposal sampling
  -> target verification -> commit accepted prefix
```

### Differences From Public DSpark

The [Inkling2 draft](mtp.py) changes the paper's context mapping and [public RNN head](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/dspark/markov_head.py):

- **Target context:** [DFlare's adaptive layer fusion and heterogeneous KV projections](https://arxiv.org/html/2606.02091v2#S4) replace DSpark's shared feature projection. Each draft layer mixes target residual-block features and the final normalized state with learned softmax weights; context and draft positions use separate K/V projections.
- **RNN head:** `rnn_hidden` adds a separately mixed target-context vector to the recurrent input. Its rank-256 correction passes through target hidden space and the shared LM head, instead of an independent vocabulary projection.
- **Block layout:** eight input positions contain one anchor and seven masks, yielding seven proposals. The paper also predicts at the anchor position.
- **Verification:** Inkling2 omits the paper's [confidence head and prefix scheduler](https://arxiv.org/html/2607.05147v1#S3.SS2), using fixed-length verification.

Inkling2-Small's draft backbone has one full-attention layer and four local layers, with bidirectional attention within the draft block and local windows anchored at its start. Unlike the target, it uses ordinary residual additions, dense SwiGLU MLPs, Q/K RMSNorm, and RoPE.

The draft shares the target's normalized token embeddings and output head. Its BF16 weights live in `mtp.safetensors`; its 131,072-token context limit also limits speculative serving.

## Precision And Output Logits

The checkpoint's `hf_quant_config.json` specifies computation precision. Expert layers can use MXFP8 or NVFP4, with matching activation quantization. MXFP8 stores FP8 E4M3 values and one UE8M0 scale byte per 32 values. NVFP4 stores packed E2M1 values, FP8 scales per 16 values, and an outer scale per expert; activation scales use per-token `fouroversix_mae` selection. FP32 parameter groups and the BF16 draft retain their respective precision. The output head can retain BF16 weights while producing FP32 logits when specified by the checkpoint.

**Different from Inkling-Small:** Inkling2 uses final logit softcapping at 15 and does not apply Inkling's final hidden-width multiplier division:

```python
logits = lm_head(final_hidden)  # FP32 output projection
logits = 15 * tanh(logits / 15)
logits[..., unpadded_vocab_size:] = -inf
```

The output head has 201,024 rows; padding rows at and above token ID 200,058 are masked after softcapping.
