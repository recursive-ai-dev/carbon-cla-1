# Carbon CLA-1 (`gemini_carbon_25_11_2026`)

**The Sub-Kilobyte Looped-Attention HTML Template Engine — Not an LLM, Built 100% From LLM Components.**

`carbon_html` is an ultra-compact deterministic tensor virtual machine (`d_model = 32`, `d_ff = 49`, **`213` nonzero matrix weights = `852 bytes` sparse `fp16`**, `5,818` dense params, `1,915 bytes` compressed PyTorch checkpoint) that both **synthesizes reusable HTML templates** from concise blueprints and **renders HTML templates with up to 3-level nested loops** purely inside PyTorch LLM building blocks—with zero natural-language training corpus, zero hallucinations, and zero external template libraries.

## LLM Components Used

| Component | Origin in Modern LLMs | Exact Role in `carbon_html` |
|---|---|---|
| **`RotaryEmbedding` (RoPE)** | Su et al. 2021 (Llama, Qwen, Gemma) | Encodes `PC`, 3D loop counters `(i0, i1, i2)`, scope/field IDs, and HTML tag stack depth on multi-frequency unit circles so `Q · K` peaks uniquely at exact index matches up to 4,096 steps. |
| **`RMSNorm` (QK-Norm)** | Zhang & Sennrich 2019 (Gemma 2/3, Qwen 3) | Normalizes query, key, and subword embeddings to invariant root-mean-square scale prior to dot-product attention. |
| **Scaled Dot-Product Attention + `KVCache`** | Vaswani et al. 2017 | **Head 0** fetches template micro-instructions; **Head 1** fetches scoped loop variables; **Head 2** pops open HTML tags; **Router** matches blueprint intents to HTML archetypes. |
| **Null Attention Sink** | Xiao et al. 2023 (*StreamingLLM*) | Position-0 `[NULL_SINK]` token in the Data KV-Cache catches out-of-bounds loop termination (`idx == len(list)`) and missing optional fields with a deterministic `4.98725` logit threshold. |
| **`ALiBiRecencyBias`** | Press et al. 2022 (*ALiBi*) | Breaks ties among historical open HTML tags at the same nesting depth so `</>` and `</*>` always close the **most recently opened** tag in exact LIFO order. |
| **`SwiGLU` FFN** | Shazeer 2020 (Llama-3) | Bias-free `(silu(x @ W_gate) * (x @ W_up)) @ W_down` executing exact bipolar `k`-input `AND` multiplexers, loop counter increments/resets, and PC jumps in residual space. |
| **`SubwordNGramEmbedding`** | FastText / Char-Subword Tokenizers | Hashes character 2/3-grams + subwords into quasi-orthogonal basis embeddings so unseen component names (`kpi_grid`, `datagrid`, `hero_banner`) route cleanly to HTML archetypes. |

## Template Syntax

- `{{ expr }}` — HTML-escaped variable interpolation (`{{ user.name }}`, `{{ @num }}`, `{{ @index }}`)
- `{! expr }` — Raw unescaped HTML slot
- `{@for item in list} ... {@end}` — Loop over arrays/lists (supports empty lists `[]` and up to 3 nested levels, plus `@index`, `@num`, `@first`, `@last`, `@not_last`, `@even`, `@odd`)
- `{@if cond} ... {@else} ... {@end}` — Truthiness & negated (`{@if !cond}` / `{@if not cond}`) branches
- `</>` — Auto-close innermost open HTML tag via ALiBi + RoPE Stack Attention
- `</*>` — Loop in-place to unwind and close **all** remaining open HTML tags in LIFO order

## Quickstart

```bash
cd /home/woez/testing_grounds/gemini_carbon_25_11_2026
.venv/bin/pytest -v
.venv/bin/python demos/run_showcase.py
.venv/bin/carbon-html stats
.venv/bin/carbon-html render '<ul>{@for x in xs}<li>{{@num}}: {{x}}</>{@end}</*>' -d '{"xs":["RoPE","ALiBi","SwiGLU"]}' --trace
```
