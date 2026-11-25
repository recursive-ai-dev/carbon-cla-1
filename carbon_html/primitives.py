"""
LLM Architectural Primitives for the Carbon Looped-Attention HTML Engine.

Every module in this file is a genuine, miniature PyTorch building block used in
modern Transformer LLMs (Llama, Gemma, Qwen, StreamingLLM), with zero template
library dependencies:
  - RMSNorm (Root Mean Square Layer Normalization, Zhang & Sennrich 2019; QK-Norm)
  - RotaryEmbedding & apply_rotary_pos_emb (Multi-frequency RoPE, Su et al. 2021)
  - ALiBiRecencyBias (Attention with Linear Biases, Press et al. 2022)
  - KVCache (Static prefill + dynamic autoregressive stack KV-cache)
  - SwiGLU (Bias-free gated linear MLP: F.silu(x @ W_gate) * (x @ W_up) @ W_down, Shazeer 2020)
  - SubwordNGramEmbedding (Character n-gram subword bag embedding for fuzzy semantic routing)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Standard Root Mean Square Normalization (as used in Llama / Gemma / Qwen3 QK-Norm)."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (Llama-style RoPE layout)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Applies rotary position embeddings to query or key tensors."""
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    Multi-frequency Rotary Position Embedding (RoPE).
    Maps integer/float positions into multi-scale 2D phase rotations so that
    Q(pos_q) . K(pos_k) attains a sharp unique maximum at pos_q == pos_k.
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        freqs: tuple[float, ...] = (
            math.pi / 8.0,
            math.pi / 128.0,
            math.pi / 2048.0,
        ),
    ) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor(freqs, dtype=torch.float32))
        self.dim = len(freqs) * 2

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (...) -> angles: (..., n_freqs) -> cos, sin: (..., 2 * n_freqs)
        angles = positions.unsqueeze(-1) * self.inv_freq
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos(), emb.sin()

    def encode_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to the canonical unit-norm carrier [1, ..., 1, 0, ..., 0] / sqrt(F).
        Equivalent to Llama's apply_rotary_pos_emb on a unit query/key basis vector.
        """
        n_freqs = self.inv_freq.shape[0]
        scale = 1.0 / math.sqrt(float(n_freqs))
        carrier = torch.zeros(
            (*positions.shape, self.dim), dtype=positions.dtype, device=positions.device
        )
        carrier[..., :n_freqs] = scale
        cos, sin = self.cos_sin(positions)
        return apply_rotary_pos_emb(carrier, cos, sin)


class ALiBiRecencyBias(nn.Module):
    """
    Attention with Linear Biases (Press et al., 2022).
    Adds -slope * (current_step - entry_step) to attention logits so that among
    all open HTML tags at the target nesting depth, the most recently opened tag
    wins deterministically in LIFO stack order.
    """

    slope: torch.Tensor

    def __init__(self, slope: float = 20.0) -> None:
        super().__init__()
        self.register_buffer("slope", torch.tensor(slope, dtype=torch.float32))

    def forward(self, current_step: torch.Tensor, key_steps: torch.Tensor) -> torch.Tensor:
        # Returns shape (..., n_keys) negative linear distance bias
        return -self.slope * (current_step - key_steps)


@dataclass
class KVCache:
    """
    Holds prefilled static KV memories (Template instructions & Data records)
    plus the dynamically growing Stack KV-Cache of open HTML tags during decode.
    """

    k_tmpl: torch.Tensor          # (1, 1, L_tmpl, d_instr_k)
    v_tmpl: torch.Tensor          # (1, 1, L_tmpl, d_instr_v)
    k_data: torch.Tensor          # (1, 1, L_data, d_data_k)
    v_data: torch.Tensor          # (1, 1, L_data, d_data_v)
    k_stack: torch.Tensor         # (1, 1, L_stack, d_stack_k)
    v_stack: torch.Tensor         # (1, 1, L_stack, d_stack_v)
    stack_steps: torch.Tensor     # (1, 1, 1, L_stack)
    push_counter: int = 0

    def push_open_tag(
        self, k_entry: torch.Tensor, v_entry: torch.Tensor
    ) -> None:
        """Appends a newly opened HTML tag to the Stack KV-Cache."""
        self.push_counter += 1
        step_val = torch.tensor(
            [[[[float(self.push_counter)]]]],
            dtype=self.stack_steps.dtype,
            device=self.stack_steps.device,
        )
        self.k_stack = torch.cat((self.k_stack, k_entry), dim=2)
        self.v_stack = torch.cat((self.v_stack, v_entry), dim=2)
        self.stack_steps = torch.cat((self.stack_steps, step_val), dim=3)


class SwiGLU(nn.Module):
    """
    Exact Llama-style bias-free SwiGLU Feed-Forward Network:
        SwiGLU(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down
    Acts as a differentiable boolean multiplexer and exact bilinear arithmetic unit.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SubwordNGramEmbedding(nn.Module):
    """
    Lightweight Subword Character N-Gram Embedding + RMSNorm.
    Maps arbitrary component/intent identifiers (including unseen variations like
    'kpi_cards', 'card_grid', 'datagrid', 'hero_banner') into a compact embedding
    space using deterministic hashed character 2/3-grams and orthogonal basis vectors.
    """

    basis: torch.Tensor

    def __init__(self, num_buckets: int = 256, embed_dim: int = 32) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.embed_dim = embed_dim
        self.norm = RMSNorm(embed_dim)
        # Deterministic quasi-orthogonal sinusoidal basis (0 trainable params needed, or buffer)
        idx = torch.arange(num_buckets, dtype=torch.float32).unsqueeze(1)
        freqs = torch.arange(1, embed_dim + 1, dtype=torch.float32).unsqueeze(0)
        basis = torch.cos(idx * freqs * 0.61803398875 * math.pi)
        basis = F.normalize(basis, dim=-1)
        self.register_buffer("basis", basis)

    @staticmethod
    def _ngrams(text: str) -> list[int]:
        clean = f"^{text.strip().lower().replace('-', '_')}$"
        grams: list[str] = []
        # Whole token + underscore parts + character 2-grams and 3-grams
        for part in clean.strip("^$").split("_"):
            if part:
                grams.append(f"#{part}#")
        for n in (2, 3):
            for i in range(max(0, len(clean) - n + 1)):
                grams.append(clean[i : i + n])
        buckets: list[int] = []
        for g in grams:
            # FNV-1a deterministic 32-bit hash -> bucket
            h = 2166136261
            for ch in g.encode("utf-8"):
                h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
            buckets.append(h)
        return buckets

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.basis.device
        rows: list[torch.Tensor] = []
        for t in texts:
            raw_ids = self._ngrams(t)
            if not raw_ids:
                rows.append(torch.zeros(self.embed_dim, device=device))
                continue
            idx = torch.tensor(
                [b % self.num_buckets for b in raw_ids], dtype=torch.long, device=device
            )
            pooled = self.basis.index_select(0, idx).sum(dim=0)
            rows.append(pooled)
        stacked = torch.stack(rows, dim=0)
        return self.norm(stacked)
