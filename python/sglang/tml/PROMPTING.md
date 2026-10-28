# Inkling2 Prompting Guide

Inkling2 retains Inkling's typed message format, reasoning blocks, and tool-call
syntax. The main prompting differences are the text tokenizer and image
input pipeline: Inkling2 supports text and images, with a vision transformer and
explicit image-grid metadata. It does not include Inkling's audio input path.

| Area | Relationship to Inkling |
|------|-------------------------|
| Text tokenization and message framing | H200k replaces o200k; role/content token IDs are unchanged. |
| Reasoning and tool calling | Same typed blocks, effort prompt, tool declarations, and `name`/`args` payload. |
| Images | Same logical image block; different preprocessing contract and vision encoder. |
| Audio | Unsupported. |
| Constrained JSON responses | Shared SGLang support for Inkling and Inkling2; the JSON constraint applies inside the final text block. |

The architecture behind these differences is described in
[ARCHITECTURE.md](ARCHITECTURE.md). Server launch instructions are in
[README.md](README.md).

## Tokenizer

Inkling2 uses the `h200k` byte-level BPE tokenizer, replacing Inkling's
`o200k_base`. The vocabulary size and structural special-token IDs are unchanged,
but the BPE merges differ: ordinary text can encode to different token IDs.
Additional special tokens mark message structure. Use the checkpoint's tokenizer
files.

The renderer encodes payload text as ordinary text and inserts structural token
IDs separately. A user writing the literal string `<|end_message|>` therefore
does not close their message. The spellings in the examples below display actual
special tokens for readability; line breaks between blocks are also for display.

### Special Tokens

The supported special tokens are:

| Token | ID | Meaning |
|-------|---:|---------|
| `<\|endoftext\|>` | 199999 | End-of-text token. |
| `<\|message_user\|>` | 200000 | Starts a user message block. |
| `<\|message_model\|>` | 200001 | Starts an assistant message block. |
| `<\|message_system\|>` | 200002 | Starts a system message block. |
| `<\|message_tool\|>` | 200003 | Starts a tool-result block. |
| `<\|content_text\|>` | 200004 | Starts visible text. |
| `<\|content_image\|>` | 200005 | Starts image content. |
| `<\|content_model_end_sampling\|>` | 200006 | Ends an assistant turn; standalone. |
| `<\|content_thinking\|>` | 200008 | Starts reasoning content. |
| `<\|end_message\|>` | 200010 | Ends one typed message block. |
| `<\|content_tool_error\|>` | 200022 | Starts model-emitted tool-error text. |
| `<\|content_xml\|>` | 200024 | Starts structured content such as tool declarations. |
| `<\|begin_of_text\|>` | 200028 | Beginning-of-text token. |
| `<\|content_invoke_tool_json\|>` | 200049 | Starts a structured JSON tool call. |
| `<\|content_invoke_tool_text\|>` | 200057 | Starts a raw text tool invocation. |

## Message Blocks And Conversation History

The shared block format is:

```text
<|message_ROLE|>optional_name<|content_KIND|>payload<|end_message|>
```

Names, when present, are ordinary text. For example:

```text
<|message_system|><|content_text|>You are a concise assistant.<|end_message|>
<|message_user|><|content_text|>What is 2 + 2?<|end_message|>
<|message_model|><|content_text|>4<|end_message|>
<|content_model_end_sampling|>
<|message_user|><|content_text|>And 3 + 3?<|end_message|>
```

An assistant turn can contain several blocks. `end_message` closes one block;
`content_model_end_sampling` closes the turn and has no trailing `end_message`.

Multiple content parts preserve their order and become separate blocks with the
same role:

```text
<|message_user|><|content_text|>Describe this image:<|end_message|>
<|message_user|><|content_image|><image feature span><|end_message|>
<|message_user|><|content_text|>Focus on the center.<|end_message|>
```

The earlier Inkling renderer appended `message_model` before sampling by default.
The Inkling2 completion renderer appends no generation prefix; current HTTP
serving also omits it for both models. The model emits its own `message_model`
and content header, as shown in the examples here.

### Standalone Rendering And Parsing

`sglang.tml.rendering` exposes typed messages, a renderer, and both complete and
incremental parsers. A text-only prompt can be constructed as follows:

```python
from sglang.tml.rendering import EncodedTextTokenSpan, Message, Renderer
from sglang.tml.rendering.tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("/path/to/checkpoint")
renderer = Renderer(tokenizer)
history = [
    Message.system_text("You are a concise assistant."),
    Message.user_text("What is 2 + 2?"),
]
spans, parser = renderer.render_for_completion_with_effort(history, effort=0.8)
assert all(isinstance(span.span, EncodedTextTokenSpan) for span in spans)
prompt_ids = [token for span in spans for token in span.span.tokens]
stop_ids = renderer.stop()

# After sampling a completion from prompt_ids:
# result = parser.parse_tokens(sampled_token_ids)
# result.messages contains typed Message objects.
```

For subsequent turns, `bridge_to_next_turn` appends new messages while preserving
the existing token/image prefix. Image spans require preprocessing before they
can be used as model inputs; see [Image Inputs](#image-inputs).

### OpenAI Chat Requests

The server selects Inkling2 rendering and the `inkling` reasoning/tool parsers
automatically. No Jinja chat template or `processor_config.json` is needed.
OpenAI `assistant` maps to `message_model`, and `developer` maps to
`message_system`. An assistant's `reasoning_content` precedes its visible content
and tool calls.

With `continue_final_message=true`, a final assistant message containing only a
text string is treated as an unfinished answer:

```text
<|message_model|><|content_text|>The answer is
```

Its closing tokens are omitted so generation continues the text.

## Reasoning

The reasoning format is unchanged from Inkling. Reasoning and visible text are
separate blocks within the assistant turn:

```text
<|message_model|><|content_thinking|>Add the two numbers.<|end_message|>
<|message_model|><|content_text|>4<|end_message|>
<|content_model_end_sampling|>
```

Reasoning can precede a tool call in the same way. The standalone parser returns
`Thinking` and `Text` separately. The chat API's reasoning parser exposes them
as `reasoning_content` and `content` when `separate_reasoning=true`.

### Reasoning Effort

As with Inkling, effort is an ordinary system text block:

```text
<|message_system|><|content_text|>Thinking effort level: 0.8<|end_message|>
```

`render_for_completion_with_effort` inserts it after the leading system messages
and before the first non-system message. Add it once when constructing the
conversation.

The shared Inkling/Inkling2 chat API accepts numeric effort in `[0, 0.99]` or these
named values:

| Request value | Prompt value |
|---------------|-------------:|
| `none` | 0 |
| `minimal` | 0.1 |
| `low` | 0.2 |
| `medium` | 0.7 |
| `high` | 0.9 |
| `xhigh`, `max` | 0.99 |

The default is `0.9` unless the server's default effort is configured otherwise.
Effort is a prompt control, not a hard limit on reasoning-token count.

## Tool Calling

The declaration and call formats are the same as Inkling. A complete tool round
trip has this shape:

```text
<|message_system|>tool_declare<|content_xml|>[{"description":"Get weather information","name":"get_weather","parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"},"type":"function"}]<|end_message|>
<|message_system|><|content_text|>Thinking effort level: 0.8<|end_message|>
<|message_user|><|content_text|>What should I wear in SF today?<|end_message|>
<|message_model|><|content_thinking|>Check the current weather.<|end_message|>
<|message_model|><|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"SF"}}<|end_message|>
<|content_model_end_sampling|>
<|message_tool|>get_weather<|content_text|>{"temperature":58,"condition":"fog"}<|end_message|>
<|message_model|><|content_text|>Wear a light jacket.<|end_message|>
<|content_model_end_sampling|>
```

### Declarations And Structured Calls

Declarations use a system message named `tool_declare`. Despite the
`content_xml` marker, its payload is a JSON array. The chat API accepts the usual
OpenAI `{"type":"function","function":{...}}` tool specs and renders flattened
objects with `description`, `name`, `parameters`, and `type`, as above. The
renderer handles the JSON formatting.

A structured call has exactly this payload:

```json
{"name":"get_weather","args":{"city":"SF"}}
```

`name` must be a string and `args` an object. Multiple call blocks represent
multiple calls. Tool-call IDs are API metadata, not part of the model-visible
JSON object.

The role header can also include the tool name:

```text
<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"SF"}}<|end_message|>
```

### Results And Tool Choice

A result uses `message_tool` and ordinary text, optionally preceded by the tool
name. Use `tool_call_id` in API message metadata to associate results with calls.

Unlike Inkling's automatic structural grammar, non-strict `tool_choice="auto"`
leaves Inkling2 generation unconstrained. Required, named, and strict-tool
requests use SGLang's tool constraints; the declaration syntax stays the same.

### Raw Tool Text And Errors

The shared raw invocation form is headerless:

```text
<|content_invoke_tool_text|>search for weather in SF<|end_message|>
```

It has no `name`/`args` object. Standalone OpenAI conversion rejects it; HTTP
parsing returns an unnamed tool call with `{"text": ...}` arguments.
Model-emitted tool errors use a normal assistant block:

```text
<|message_model|><|content_tool_error|>The requested tool is unavailable.<|end_message|>
```

This is distinct from an external tool returning an error string in a
`message_tool` result.

## Constrained JSON Responses

JSON response framing is shared with Inkling. For chat
`response_format={"type":"json_object"}` or `json_schema`, the grammar preserves
optional reasoning and constrains the final text payload:

```text
<|message_model|><|content_thinking|>Compute the result.<|end_message|>
<|message_model|><|content_text|>{"answer":4}<|end_message|>
```

The API returns `{"answer":4}` as visible content, without the message framing.
Reasoning is free text except for reserved framing spellings; the final answer
must match the JSON constraint.

This mode requires `--grammar-backend xgrammar`, the `inkling` reasoning parser,
and `separate_reasoning=true`. Inkling2 selects the reasoning parser
automatically.

## Image Inputs

The logical image block is shared with Inkling:

```text
<|message_user|><|content_image|><image feature span><|end_message|>
```

`<image feature span>` is not literal text. It reserves positions for the image
embeddings and carries separate pixel features. In the chat API, provide
`image_url` content parts in the desired order; the server loads the images and
expands their placeholders. Enable multimodal serving as described in
[README.md](README.md).

### Preprocessing Differences

Inkling's image encoder embedded patch features independently with an hMLP.
Inkling2 feeds a vision transformer, so the grid layout and original image size
are part of the input contract. Load its processor from the checkpoint's
`image_preprocessing.json`:

```python
from sglang.tml.multimodal import Inkling2ImageProcessor

image_processor = Inkling2ImageProcessor.from_pretrained("/path/to/checkpoint")
features = image_processor.preprocess(["example.png"], return_tensors="pt")

vision_patches = features["vision_patches_bthwc"]
vision_grid = features["vision_grid_thw"]
original_size = features["vision_original_thw"]
num_tokens = features["num_tokens"]
```

The returned `BatchFeature` contains:

```text
vision_patches_bthwc: bfloat16 tensor [sum(num_tokens), 2, 40, 40, 3]
vision_grid_thw:      int64 tensor [num_images, 3], rows [1, grid_rows, grid_columns]
vision_original_thw:  int64 tensor [num_images, 3], rows [1, height, width]
num_tokens:          list[int], one token count per image
```

For a width `W` and height `H`, the grid is `ceil(H / 40)` by `ceil(W / 40)`,
with one language-model image token per grid cell. A 41-by-39 image therefore
uses two image tokens. The two temporal slices repeat the same still image;
they do not add two tokens per cell or enable video input.

The processor's 40-pixel grid is different from the vision encoder's 20-pixel
patch size. Each 40-by-40 feature row is split into four native patches for the
vision transformer; the spatial adapter combines their outputs into one
language-model embedding. The processor does not return Inkling's separate
`num_patches` field.

Preprocessing preserves the original dimensions by default and pads to the
40-pixel grid. Transparent inputs are composited onto a checkerboard before RGB
normalization, so preserve transparency when passing a PIL image. Byte inputs,
PIL images, paths, HTTP(S) URLs, and base64 data URIs are supported.

## Incremental Parsing And Boundaries

The standalone parser consumes token IDs, preserving the distinction between
framing tokens and identical-looking ordinary text. `parser.parse_tokens(...)`
requires complete blocks. For incremental consumption, use
`parser.parse_stream()`, feed `TokenSpan` objects, and call `finish()` at the end.
Text and thinking produce incremental content updates; structured calls are
returned once their complete JSON block closes.

A token limit can stop generation mid-block. Use `parse_partial(...)` to recover
completed messages and partial text from a truncated response.
