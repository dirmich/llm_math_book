# Book-to-code verification map

This table identifies the language-neutral implementation and regression test for
the mathematical contracts shared by the Korean, English, and Japanese editions.
Notebook prose may be translated, but these calculations must not be reimplemented
differently by language.

| Book topic | Notebook chapter | Shared symbol | Verification |
|---|---|---|---|
| stable softmax and probability sum | Ch05, Ch14 | `stable_softmax` | `test_softmax_is_stable_and_normalized` |
| cross-entropy equals negative log likelihood | Ch05, Ch08 | `manual_cross_entropy` | `test_cross_entropy_matches_pytorch` |
| numerical and backpropagated gradients | Ch03, Ch07 | `finite_difference_gradient` | `test_finite_difference_matches_backpropagation` |
| attention scaling by `sqrt(d_k)` | Ch14 | `scaled_dot_product_attention` | `test_attention_uses_sqrt_dk_and_causal_mask` |
| causal masking prevents future-token leakage | Ch14, Ch18 | `scaled_dot_product_attention` | `test_attention_uses_sqrt_dk_and_causal_mask` |
| rotary encoding preserves vector norms | Ch16 | `apply_rope` | `test_rope_preserves_norm_and_depends_on_position` |
| LoRA starts with zero update | Ch26 | `lora_delta` | `test_lora_initial_delta_and_dpo_preference_direction` |
| DPO rewards preferred ordering | Ch22 | `dpo_loss` | `test_lora_initial_delta_and_dpo_preference_direction` |

The test names are in `tests/unit/test_math_reference.py`. Public releases must run
them on CPU. GPU measurements are performance records, not alternate correctness
definitions.
