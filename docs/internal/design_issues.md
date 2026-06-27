# Design Issues

## Text Encoder Class Token And Padding

### Problem

`TransformerEncoder` can append a learned class token to the text sequence and use
`pool_type="last"` to produce the text embedding. This is used by CoCa-style text encoders.

With fixed-length tokenization, the embedded sequence can look like:

```text
[SOT, token, token, EOT, PAD, PAD, PAD, CLS]
```

If attention is not padded correctly (like standard causal masking), the appended `CLS` token can attend to every preceding slot.
That includes padding slots. Without a padding mask, the pooled text
embedding becomes dependent on padding embeddings and padding positions.
This is different from CLIP-style EOS pooling, where the pooled EOS token appears before
future padding and therefore cannot see it under causal attention.

### Why We Do Not Just Inject Tokenizer IDs Everywhere

The most attractive API would be to construct the model with a tokenizer and
automatically inject `eos_token_id`, `pad_token_id`, and similar tokenizer
metadata. That would keep model and tokenizer config aligned.

However, token-id padding masks are not always safe. Some tokenizers use the same
id for padding and EOS. SigLIP v1 is one example: `PAD == EOS`. If we set
`pad_token_id` for such a tokenizer and build masks with `text != pad_token_id`,
we also mask real EOS tokens. In those cases, the correct token-id-only model
setting is `pad_token_id=None`.

### Current Design

`TransformerEncoder` rejects `class_token=True` unless `pad_token_id` is configured.

This makes the risky class-token setup fail at construction instead of silently
training with class-token embeddings that attend to padding. It also keeps the
rule at the right layer: the issue is not specific to CoCa, it is a property of a
text encoder that appends and pools a class token after a padded sequence.

Known configs whose tokenizer uses a reliable padding id should set
`pad_token_id` explicitly. Generic configs should not guess.

### OpenVision v1 Compatibility

OpenVision v1 is an intentional exception to the usual padding-mask guidance.
Its exported OpenCLIP configs use a custom WordPiece tokenizer that lays out text as:

```text
[BOS, token, token, EOS, PAD, PAD, PAD, CLS]
```

Those configs use `pool_type="last"` and `no_causal_mask=true`, and the text
tower pools the final sequence slot without a padding attention mask.
Birder's `openvision_v1` tokenizer mirrors that layout with `BOS=1`, `EOS=2`, `PAD=0` and `CLS=101`.

For OpenVision v1 pretrained compatibility, do not add `pad_token_id` to the
registered text encoder configs just because the tokenizer has a distinct
padding id. Adding a padding attention mask would change the pretrained text
tower's behavior. New non-causal class-token configs should still decide
explicitly whether they need a padding mask.

## Text Encoder `pool_type="last"` Semantics

`TransformerEncoder` and `HFTextEncoder` both expose `pool_type="last"`, but
the name maps to different legacy-compatible behavior.

`TransformerEncoder` pools the final sequence slot. This is needed for text
stacks that place a meaningful token at the end of the fixed-length sequence,
for example an appended class token.

`HFTextEncoder` pools the last non-padding token when `pad_token_id` is known,
and falls back to the final sequence slot when it is not. This matches common
Hugging Face text-model usage, where the final fixed-length slot is usually
padding after right-padding.

The two meanings are intentionally not normalized at the base API layer because
registered checkpoints and tokenizer conventions differ. Configs that rely on
last non-padding pooling should use `HFTextEncoder` with a valid `pad_token_id`,
or avoid `pool_type="last"` on `TransformerEncoder` unless the final fixed
sequence position is semantically meaningful.
