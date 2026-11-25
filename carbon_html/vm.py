"""
CarbonLoopedVM: The Weight-Shared Looped-Attention HTML Template Engine.

Executes HTML templates containing loops (including 3-level nested loops),
conditionals ({@if} / {@else}), escaped & raw variable interpolation ({{x}}, {!x}),
built-in loop metadata (@index, @num, @first, @last, @even, @odd, @not_last),
and stack-based HTML auto-closing (</> and </*>) purely via PyTorch LLM components:
  1. InstructionFetchAttention (Multi-frequency RoPE + QK-RMSNorm + SDPA)
  2. DataAndStackAttention (Tuple RoPE + Attention Sink + ALiBi + QK-RMSNorm + SDPA)
  3. SwiGLU State-Machine FFN (exact Llama F.silu(x @ W_gate) * (x @ W_up) @ W_down)
  4. Autoregressive residual stream loop: h_{t+1} = CarbonBlock(h_t, kv_cache)

Zero external template libraries are used; loop control flow happens entirely inside
the recurrent tensor state and KV-cache attention heads.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import ALiBiRecencyBias, KVCache, RMSNorm, RotaryEmbedding, SwiGLU

# ---------------------------------------------------------------------------
# Residual Stream Channel Map (d_model = 32, power-of-two)
# ---------------------------------------------------------------------------
D_MODEL = 32

# Persistent machine registers (channels 0..5)
CH_ONE = 0            # Constant 1.0 (homogeneous bias reference for bias-free Linear layers)
CH_PC = 1             # Template instruction pointer
CH_I0 = 2             # Loop depth-0 index
CH_I1 = 3             # Loop depth-1 index
CH_I2 = 4             # Loop depth-2 index
CH_DEPTH = 5          # Open HTML tag stack depth

# Instruction microcode scratch channels written by Head 0 (channels 6..24)
CH_OP_LIT = 6
CH_OP_VAR = 7
CH_OP_FOR = 8
CH_OP_NEXT = 9
CH_OP_IF = 10
CH_OP_JMP = 11
CH_OP_CLOSE = 12
CH_OP_HALT = 13
CH_LD0 = 14
CH_LD1 = 15
CH_LD2 = 16
CH_CLOSE_ALL = 17
CH_JUMP_DELTA = 18
CH_SCOPE_ID = 19
CH_FIELD_ID = 20
CH_LIT_SYM = 21
CH_CLOSE_SYM_PUSH = 22
CH_TAG_DELTA = 23
CH_TAG_IS_OPEN = 24

# Data & Stack scratch channels written by Head 1 & Head 2 (channels 25..29)
CH_DATA_EXISTS = 25
CH_DATA_TRUTHY = 26
CH_DATA_SYM = 27
CH_STACK_SYM = 28
CH_DEPTH_POS = 29

# Step output latches written by SwiGLU (channels 30..31)
CH_EMIT_SYM = 30
CH_PUSH_FLAG = 31

# Reserved symbol IDs in the dynamic symbol table
SYM_NOP = 0
SYM_EOS = 1
SYM_EMPTY = 2

# HTML5 void elements that do not push a closing tag onto the open-tag stack
VOID_HTML_TAGS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img",
        "input", "link", "meta", "param", "source", "track", "wbr",
    }
)

# Opcode constants
OP_LIT = 0
OP_VAR = 1
OP_FOR = 2
OP_NEXT = 3
OP_IF = 4
OP_JMP = 5
OP_CLOSE = 6
OP_HALT = 7

OP_NAMES = ("LIT", "VAR", "FOR", "NEXT", "IF", "JMP", "CLOSE", "HALT")


@dataclass
class StepTrace:
    """Mechanistic telemetry recorded at a single recurrent decode step."""

    step: int
    pc: int
    opcode: str
    loop_indices: tuple[int, int, int]
    tag_depth: int
    instr_attn_peak: float
    data_attn_peak: float
    data_attn_sink: bool
    stack_attn_peak: float
    active_gates: int
    emitted: str


@dataclass
class RenderResult:
    """Output of CarbonLoopedVM execution including optional mechanistic traces."""

    html: str
    steps: int
    emitted_tokens: int
    traces: list[StepTrace] = field(default_factory=list)


@dataclass
class MicroInstruction:
    """Single compiled template micro-instruction stored in Template KV-Cache."""

    opcode: int
    loop_depth: int = 0
    close_all: bool = False
    jump_delta: int = 0
    scope_name: str = "$root"
    field_name: str = ""
    raw_output: bool = False
    literal_text: str = ""
    close_tag_text: str = ""
    tag_delta: int = 0
    tag_is_open: bool = False


# ---------------------------------------------------------------------------
# Attention Sublayers & CarbonBlock (100% PyTorch LLM Primitives)
# ---------------------------------------------------------------------------
class InstructionFetchAttention(nn.Module):
    """
    Head 0: Queries the Template KV-Cache at the current Program Counter (CH_PC)
    using multi-frequency RoPE + QK-RMSNorm + F.scaled_dot_product_attention.
    """

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.rope = RotaryEmbedding(
            freqs=(math.pi / 8.0, math.pi / 128.0, math.pi / 2048.0)
        )
        self.head_dim = self.rope.dim  # 6
        self.val_dim = 19              # channels 6..24 inclusive
        self.q_proj = nn.Linear(d_model, 1, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(self.val_dim, d_model, bias=False)
        self.attn_scale = 800.0
        self._compile_weights()

    def _compile_weights(self) -> None:
        with torch.no_grad():
            self.q_proj.weight.zero_()
            self.q_proj.weight[0, CH_PC] = 1.0
            self.o_proj.weight.zero_()
            for i in range(self.val_dim):
                self.o_proj.weight[CH_OP_LIT + i, i] = 1.0

    def forward(
        self, h: torch.Tensor, kv: KVCache, record_attn: bool = False
    ) -> tuple[torch.Tensor, float]:
        # Extract scalar PC via linear projection and apply multi-frequency RoPE
        pc = self.q_proj(h).squeeze(-1)  # (B, 1)
        q = self.rope.encode_positions(pc).unsqueeze(1)  # (B, 1, 1, 6)
        q = self.q_norm(q)
        k = self.k_norm(kv.k_tmpl)
        if record_attn:
            scores = torch.matmul(q, k.transpose(-2, -1)) * (
                self.attn_scale / float(self.head_dim)
            )
            weights = F.softmax(scores, dim=-1)
            peak = float(weights.max().item())
            v_out = torch.matmul(weights, kv.v_tmpl)
        else:
            v_out = F.scaled_dot_product_attention(
                q, k, kv.v_tmpl, scale=self.attn_scale / float(self.head_dim)
            )
            peak = 1.0
        return self.o_proj(v_out.squeeze(1)), peak


class DataAndStackAttention(nn.Module):
    """
    Head 1 (Data KV-Cache Fetch with StreamingLLM-style Null Attention Sink) +
    Head 2 (HTML Open-Tag Stack Fetch with RoPE Depth + ALiBi Recency Bias).
    """

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.data_rope = RotaryEmbedding(
            freqs=(math.pi / 8.0, math.pi / 128.0, math.pi / 2048.0)
        )
        self.stack_rope = RotaryEmbedding(
            freqs=(math.pi / 4.0, math.pi / 32.0, math.pi / 256.0)
        )
        self.alibi = ALiBiRecencyBias(slope=20.0)

        # Data head dimensions: 5 RoPE blocks * 6 + 1 bias + 2 orthogonal norm-pad = 33
        self.d_data_k = 33
        self.d_data_v = 3   # [exists, truthy, data_sym]
        self.d_stack_k = 6
        self.d_stack_v = 2  # [stack_sym, depth_pos]

        # Linear extraction of (scope_id, field_id, i0, i1, i2, one) and (tag_depth)
        self.q_data_idx = nn.Linear(d_model, 6, bias=False)
        self.q_stack_idx = nn.Linear(d_model, 1, bias=False)

        self.q_data_norm = RMSNorm(self.d_data_k)
        self.k_data_norm = RMSNorm(self.d_data_k)
        self.q_stack_norm = RMSNorm(self.d_stack_k)
        self.k_stack_norm = RMSNorm(self.d_stack_k)

        self.o_proj = nn.Linear(self.d_data_v + self.d_stack_v, d_model, bias=False)
        self.data_scale = 2500.0 * float(self.d_data_k) / 25.0
        self.stack_scale = 4_000_000.0 / float(self.d_stack_k)
        self._compile_weights()

    def _compile_weights(self) -> None:
        with torch.no_grad():
            self.q_data_idx.weight.zero_()
            self.q_data_idx.weight[0, CH_SCOPE_ID] = 1.0
            self.q_data_idx.weight[1, CH_FIELD_ID] = 1.0
            self.q_data_idx.weight[2, CH_I0] = 1.0
            self.q_data_idx.weight[3, CH_I1] = 1.0
            self.q_data_idx.weight[4, CH_I2] = 1.0
            self.q_data_idx.weight[5, CH_ONE] = 1.0

            self.q_stack_idx.weight.zero_()
            self.q_stack_idx.weight[0, CH_DEPTH] = 1.0

            self.o_proj.weight.zero_()
            # Data outputs -> channels 25, 26, 27
            self.o_proj.weight[CH_DATA_EXISTS, 0] = 1.0
            self.o_proj.weight[CH_DATA_TRUTHY, 1] = 1.0
            self.o_proj.weight[CH_DATA_SYM, 2] = 1.0
            # Stack outputs -> channels 28, 29
            self.o_proj.weight[CH_STACK_SYM, 3] = 1.0
            self.o_proj.weight[CH_DEPTH_POS, 4] = 1.0

    def forward(
        self, u: torch.Tensor, kv: KVCache, record_attn: bool = False
    ) -> tuple[torch.Tensor, float, bool, float]:
        # --- Head 1: Data KV-Cache Attention ---
        extracted = self.q_data_idx(u)  # (B, 1, 6)
        five_pos = extracted[..., :5]   # (B, 1, 5): scope, field, i0, i1, i2
        one_val = extracted[..., 5:6]   # (B, 1, 1)
        rope_five = self.data_rope.encode_positions(five_pos).flatten(-2)  # (B, 1, 30)
        # Active squared norm of Q is 5 * 1.0 + 1.0^2 = 6.0; pad slot 31 with sqrt(19.0) -> ||Q||^2 = 25.0
        q_pad = one_val * math.sqrt(19.0)
        k_pad_zero = torch.zeros_like(one_val)
        q_data = torch.cat((rope_five, one_val, q_pad, k_pad_zero), dim=-1).unsqueeze(1)
        q_data_n = self.q_data_norm(q_data)
        k_data_n = self.k_data_norm(kv.k_data)

        if record_attn:
            d_scores = torch.matmul(q_data_n, k_data_n.transpose(-2, -1)) * self.data_scale
            d_weights = F.softmax(d_scores, dim=-1)
            data_peak = float(d_weights.max().item())
            hit_sink = int(d_weights.argmax(dim=-1).item()) == 0
            v_data_out = torch.matmul(d_weights, kv.v_data).squeeze(1)
        else:
            v_data_out = F.scaled_dot_product_attention(
                q_data_n, k_data_n, kv.v_data, scale=self.data_scale
            ).squeeze(1)
            data_peak = 1.0
            hit_sink = False

        # --- Head 2: Stack KV-Cache Attention (RoPE depth + ALiBi recency) ---
        depth = self.q_stack_idx(u).squeeze(-1)  # (B, 1)
        q_stack = self.stack_rope.encode_positions(depth).unsqueeze(1)  # (B, 1, 1, 6)
        q_stack_n = self.q_stack_norm(q_stack)
        k_stack_n = self.k_stack_norm(kv.k_stack)

        cur_step = torch.tensor(
            float(kv.push_counter + 1), dtype=u.dtype, device=u.device
        )
        alibi_bias = self.alibi(cur_step, kv.stack_steps)  # (1, 1, 1, L_stack)

        if record_attn:
            s_scores = (
                torch.matmul(q_stack_n, k_stack_n.transpose(-2, -1)) * self.stack_scale
                + alibi_bias
            )
            s_weights = F.softmax(s_scores, dim=-1)
            stack_peak = float(s_weights.max().item())
            v_stack_out = torch.matmul(s_weights, kv.v_stack).squeeze(1)
        else:
            v_stack_out = F.scaled_dot_product_attention(
                q_stack_n,
                k_stack_n,
                kv.v_stack,
                attn_mask=alibi_bias,
                scale=self.stack_scale,
            ).squeeze(1)
            stack_peak = 1.0

        combined = torch.cat((v_data_out, v_stack_out), dim=-1)
        return self.o_proj(combined), data_peak, hit_sink, stack_peak


class CarbonBlock(nn.Module):
    """
    Single Weight-Shared Looped-Attention Block.
    Contains:
      - InstructionFetchAttention (RoPE + QK-RMSNorm + SDPA)
      - DataAndStackAttention (Tuple RoPE + Null Sink + ALiBi + QK-RMSNorm + SDPA)
      - SwiGLU (Llama bias-free gated MLP compiling exact loop & stack transitions)
    """

    D_FF = 49
    N_LOGIC_GATES = 23
    GATE_SCALE = 40.0

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.d_model = d_model
        self.instr_attn = InstructionFetchAttention(d_model)
        self.mem_attn = DataAndStackAttention(d_model)
        self.ffn = SwiGLU(d_model, self.D_FF)
        self._compile_swiglu()

    def _compile_swiglu(self) -> None:
        """
        Analytically compiles the 8-opcode looped virtual machine transitions into
        Llama SwiGLU matrices (W_gate, W_up, W_down) with zero biases.

        Uses the exact k-input bipolar AND identity:
            G = S * (sum_{m=1}^k s_m * b_m - (k - 1) * ONE),  S = 40.0
        For b_m in {-1, +1}:
          - When all literals hold: G = +40.0 -> silu(+40.0) / 40.0 = 1.0000000
          - When >= 1 literal fails: G <= -40.0 -> silu(<= -40.0) / 40.0 = 0.0000000
        """
        S = self.GATE_SCALE
        with torch.no_grad():
            self.ffn.w_gate.weight.zero_()
            self.ffn.w_up.weight.zero_()
            self.ffn.w_down.weight.zero_()

            cursor = 0

            def add_rule(
                conds: list[tuple[int, float]],
                up_ch: int,
                up_scale: float,
                outputs: list[tuple[int, float]],
            ) -> None:
                nonlocal cursor
                n = cursor
                cursor += 1
                k = len(conds)
                if k == 0:
                    # Unconditional active gate: G = +S * ONE
                    self.ffn.w_gate.weight[n, CH_ONE] = S
                else:
                    for ch, sign in conds:
                        self.ffn.w_gate.weight[n, ch] += S * sign
                    self.ffn.w_gate.weight[n, CH_ONE] += -S * float(k - 1)

                self.ffn.w_up.weight[n, up_ch] = up_scale
                for out_ch, out_weight in outputs:
                    self.ffn.w_down.weight[out_ch, n] = out_weight / S

            # --- 1. Program Counter (CH_PC), Stack Depth, & Output Symbol ---
            # OP_LIT: PC += 1, EMIT_SYM = LIT_SYM, DEPTH += TAG_DELTA
            add_rule([(CH_OP_LIT, +1.0)], CH_ONE, 1.0, [(CH_PC, +1.0)])
            add_rule([(CH_OP_LIT, +1.0)], CH_LIT_SYM, 1.0, [(CH_EMIT_SYM, +1.0)])
            add_rule([(CH_OP_LIT, +1.0)], CH_TAG_DELTA, 1.0, [(CH_DEPTH, +1.0)])
            add_rule(
                [(CH_OP_LIT, +1.0), (CH_TAG_IS_OPEN, +1.0)],
                CH_ONE,
                1.0,
                [(CH_PUSH_FLAG, +1.0)],
            )

            # OP_VAR: PC += 1, EMIT_SYM = DATA_SYM
            add_rule([(CH_OP_VAR, +1.0)], CH_ONE, 1.0, [(CH_PC, +1.0)])
            add_rule([(CH_OP_VAR, +1.0)], CH_DATA_SYM, 1.0, [(CH_EMIT_SYM, +1.0)])

            # OP_FOR: if DATA_EXISTS == +1 -> PC += 1; else -> PC += JUMP_DELTA and reset I_d
            add_rule(
                [(CH_OP_FOR, +1.0), (CH_DATA_EXISTS, +1.0)],
                CH_ONE,
                1.0,
                [(CH_PC, +1.0)],
            )
            add_rule(
                [(CH_OP_FOR, +1.0), (CH_DATA_EXISTS, -1.0)],
                CH_JUMP_DELTA,
                1.0,
                [(CH_PC, +1.0)],
            )
            for ld_ch, idx_ch in ((CH_LD0, CH_I0), (CH_LD1, CH_I1), (CH_LD2, CH_I2)):
                add_rule(
                    [(CH_OP_FOR, +1.0), (CH_DATA_EXISTS, -1.0), (ld_ch, +1.0)],
                    idx_ch,
                    -1.0,
                    [(idx_ch, +1.0)],
                )

            # OP_NEXT: PC += JUMP_DELTA, I_d += 1
            add_rule([(CH_OP_NEXT, +1.0)], CH_JUMP_DELTA, 1.0, [(CH_PC, +1.0)])
            for ld_ch, idx_ch in ((CH_LD0, CH_I0), (CH_LD1, CH_I1), (CH_LD2, CH_I2)):
                add_rule(
                    [(CH_OP_NEXT, +1.0), (ld_ch, +1.0)],
                    CH_ONE,
                    1.0,
                    [(idx_ch, +1.0)],
                )

            # OP_IF: if DATA_TRUTHY == +1 -> PC += 1; else -> PC += JUMP_DELTA
            add_rule(
                [(CH_OP_IF, +1.0), (CH_DATA_TRUTHY, +1.0)],
                CH_ONE,
                1.0,
                [(CH_PC, +1.0)],
            )
            add_rule(
                [(CH_OP_IF, +1.0), (CH_DATA_TRUTHY, -1.0)],
                CH_JUMP_DELTA,
                1.0,
                [(CH_PC, +1.0)],
            )

            # OP_JMP: PC += JUMP_DELTA
            add_rule([(CH_OP_JMP, +1.0)], CH_JUMP_DELTA, 1.0, [(CH_PC, +1.0)])

            # OP_CLOSE (</> and </*>):
            # - If stack empty (DEPTH_POS == -1): PC += 1, emit NOP
            add_rule(
                [(CH_OP_CLOSE, +1.0), (CH_DEPTH_POS, -1.0)],
                CH_ONE,
                1.0,
                [(CH_PC, +1.0)],
            )
            # - If stack non-empty (DEPTH_POS == +1): emit STACK_SYM, DEPTH -= 1
            add_rule(
                [(CH_OP_CLOSE, +1.0), (CH_DEPTH_POS, +1.0)],
                CH_STACK_SYM,
                1.0,
                [(CH_EMIT_SYM, +1.0)],
            )
            add_rule(
                [(CH_OP_CLOSE, +1.0), (CH_DEPTH_POS, +1.0)],
                CH_ONE,
                -1.0,
                [(CH_DEPTH, +1.0)],
            )
            # - Single close </> advances PC += 1; close-all </*> keeps PC += 0 so it loops!
            add_rule(
                [(CH_OP_CLOSE, +1.0), (CH_DEPTH_POS, +1.0), (CH_CLOSE_ALL, -1.0)],
                CH_ONE,
                1.0,
                [(CH_PC, +1.0)],
            )

            # OP_HALT: EMIT_SYM = SYM_EOS (1.0)
            add_rule([(CH_OP_HALT, +1.0)], CH_ONE, float(SYM_EOS), [(CH_EMIT_SYM, +1.0)])

            # --- 2. Autonomous Scratch & Latch Cleanup (channels 6..31) ---
            # Subtract u[ch] unconditionally so scratch channels reset to 0 and
            # output latches (CH_EMIT_SYM, CH_PUSH_FLAG) hold purely this step's value.
            for scratch_ch in range(CH_OP_LIT, D_MODEL):
                add_rule([], scratch_ch, -1.0, [(scratch_ch, +1.0)])

            assert cursor <= self.D_FF, f"Exceeded D_FF: {cursor} > {self.D_FF}"

    def forward(
        self, h: torch.Tensor, kv: KVCache, record_attn: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        instr_delta, instr_peak = self.instr_attn(h, kv, record_attn=record_attn)
        u1 = h + instr_delta
        mem_delta, data_peak, hit_sink, stack_peak = self.mem_attn(
            u1, kv, record_attn=record_attn
        )
        u2 = u1 + mem_delta
        if record_attn:
            gates = F.silu(self.ffn.w_gate(u2))
            # Exclude the 26 always-on cleanup gates when reporting active logic gates
            logic_gates = int((gates[..., : self.N_LOGIC_GATES] > 1.0).sum().item())
        else:
            logic_gates = 0
        h_next = u2 + self.ffn(u2)
        meta = {
            "u2": u2,
            "instr_peak": instr_peak,
            "data_peak": data_peak,
            "hit_sink": hit_sink,
            "stack_peak": stack_peak,
            "logic_gates": logic_gates,
        }
        return h_next, meta


# ---------------------------------------------------------------------------
# Template Compiler & Data KV-Cache Serializer (Prefill Stage)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"(\{#.*?#\}|\{@for\s+.*?\s+in\s+.*?\}|\{@if\s+.*?\}|\{@else\}|\{@end(?:for|if)?\}|\{\{.*?\}\}|\{!.*?\}|</\*>|</>)",
    re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"(<!--.*?-->|<!DOCTYPE[^>]*>|</?[a-zA-Z][a-zA-Z0-9:-]*(?:\s[^<>]*?)?>)", re.DOTALL | re.IGNORECASE)
_OPEN_TAG_NAME_RE = re.compile(r"^<([a-zA-Z][a-zA-Z0-9:-]*)\b")
_CLOSE_TAG_NAME_RE = re.compile(r"^</([a-zA-Z][a-zA-Z0-9:-]*)\s*>$")
_PARTIAL_OPEN_TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9:-]*)\b[^<>]*$", re.DOTALL)


def compile_template(template: str) -> list[MicroInstruction]:
    """
    Compiles a template string into a flat array of MicroInstructions ready to be
    encoded as the Template KV-Cache. Supports:
      - {@for item in expr} ... {@end} (up to 3 nested levels)
      - {@if expr} / {@if not expr} / {@if !expr} ... {@else} ... {@end}
      - {{expr}} (HTML-escaped) and {!expr} (raw HTML)
      - </> (pop & close innermost open HTML tag) and </*> (unwind all open tags)
    """
    instrs: list[MicroInstruction] = []
    # Active loop stack: list of (var_name, scope_path)
    loop_stack: list[tuple[str, str]] = []
    # Control-flow patch stack: entries for 'for' or 'if'
    ctrl_stack: list[dict[str, Any]] = []
    # Tracks an HTML open tag split across {{...}} / {!...} attribute interpolations
    pending_open_tag: str | None = None

    def resolve_expr(expr: str) -> tuple[str, str]:
        expr = expr.strip()
        if expr.startswith("@"):
            # Innermost loop metadata (@index, @num, @first, @last, @even, @odd, @not_last)
            if not loop_stack:
                return ("$root", expr)
            return (loop_stack[-1][1], expr)
        if "." in expr:
            head, tail = expr.split(".", 1)
            for var_name, scope_path in reversed(loop_stack):
                if head == var_name:
                    return (scope_path, tail)
            return ("$root", expr)
        # Bare identifier: check if it is itself a loop variable (primitive list)
        for var_name, scope_path in reversed(loop_stack):
            if expr == var_name:
                return (scope_path, "$self")
        return ("$root", expr)

    def emit_literal_html(chunk: str) -> None:
        nonlocal pending_open_tag
        if not chunk:
            return
        pos = 0
        # If an open tag started before an attribute interpolation (e.g. <a href="{{url}}">),
        # finish scanning to its closing '>' first!
        if pending_open_tag is not None:
            gt_idx = chunk.find(">")
            if gt_idx == -1:
                instrs.append(MicroInstruction(opcode=OP_LIT, literal_text=chunk))
                return
            head_piece = chunk[: gt_idx + 1]
            tag_lower = pending_open_tag.lower()
            self_closing = (
                head_piece.rstrip().endswith("/>") or tag_lower in VOID_HTML_TAGS
            )
            if self_closing:
                instrs.append(MicroInstruction(opcode=OP_LIT, literal_text=head_piece))
            else:
                instrs.append(
                    MicroInstruction(
                        opcode=OP_LIT,
                        literal_text=head_piece,
                        close_tag_text=f"</{pending_open_tag}>",
                        tag_delta=+1,
                        tag_is_open=True,
                    )
                )
            pending_open_tag = None
            pos = gt_idx + 1
            if pos >= len(chunk):
                return

        for m in _HTML_TAG_RE.finditer(chunk, pos):
            if m.start() > pos:
                instrs.append(MicroInstruction(opcode=OP_LIT, literal_text=chunk[pos : m.start()]))
            tag_str = m.group(0)
            pos = m.end()
            if tag_str.startswith("<!--") or tag_str.upper().startswith("<!DOCTYPE") or tag_str.startswith("<!"):
                instrs.append(MicroInstruction(opcode=OP_LIT, literal_text=tag_str))
                continue
            close_m = _CLOSE_TAG_NAME_RE.match(tag_str)
            if close_m:
                instrs.append(
                    MicroInstruction(opcode=OP_LIT, literal_text=tag_str, tag_delta=-1)
                )
                continue
            open_m = _OPEN_TAG_NAME_RE.match(tag_str)
            if open_m:
                tag_name = open_m.group(1).lower()
                self_closing = tag_str.rstrip().endswith("/>") or tag_name in VOID_HTML_TAGS
                if self_closing:
                    instrs.append(MicroInstruction(opcode=OP_LIT, literal_text=tag_str))
                else:
                    instrs.append(
                        MicroInstruction(
                            opcode=OP_LIT,
                            literal_text=tag_str,
                            close_tag_text=f"</{open_m.group(1)}>",
                            tag_delta=+1,
                            tag_is_open=True,
                        )
                    )
                continue
            instrs.append(MicroInstruction(opcode=OP_LIT, literal_text=tag_str))
        if pos < len(chunk):
            tail = chunk[pos:]
            partial_m = _PARTIAL_OPEN_TAG_RE.search(tail)
            if partial_m:
                pending_open_tag = partial_m.group(1)
            instrs.append(MicroInstruction(opcode=OP_LIT, literal_text=tail))

    for part in _TOKEN_RE.split(template):
        if not part or part.startswith("{#"):
            continue
        if part == "</>":
            instrs.append(MicroInstruction(opcode=OP_CLOSE, close_all=False))
        elif part == "</*>":
            instrs.append(MicroInstruction(opcode=OP_CLOSE, close_all=True))
        elif part.startswith("{{") and part.endswith("}}"):
            scope_name, field_name = resolve_expr(part[2:-2])
            instrs.append(
                MicroInstruction(
                    opcode=OP_VAR,
                    scope_name=scope_name,
                    field_name=field_name,
                    raw_output=False,
                )
            )
        elif part.startswith("{!") and part.endswith("}"):
            scope_name, field_name = resolve_expr(part[2:-1])
            instrs.append(
                MicroInstruction(
                    opcode=OP_VAR,
                    scope_name=scope_name,
                    field_name=field_name,
                    raw_output=True,
                )
            )
        elif part.startswith("{@for "):
            inner = part[6:-1].strip()
            var_name, list_expr = [s.strip() for s in inner.split(" in ", 1)]
            depth = len(loop_stack)
            if depth >= 3:
                raise ValueError("CarbonLoopedVM supports up to 3 nested loop levels (0, 1, 2).")
            parent_scope, list_field = resolve_expr(list_expr)
            child_scope = (
                list_field if parent_scope == "$root" else f"{parent_scope}.{list_field}"
            )
            for_idx = len(instrs)
            instrs.append(
                MicroInstruction(
                    opcode=OP_FOR,
                    loop_depth=depth,
                    scope_name=child_scope,
                    field_name="$exists",
                )
            )
            loop_stack.append((var_name, child_scope))
            ctrl_stack.append({"kind": "for", "for_idx": for_idx, "depth": depth})
        elif part.startswith("{@if "):
            raw_cond = part[5:-1].strip()
            negated = False
            if raw_cond.startswith("!"):
                negated = True
                raw_cond = raw_cond[1:].strip()
            elif raw_cond.startswith("not "):
                negated = True
                raw_cond = raw_cond[4:].strip()
            scope_name, field_name = resolve_expr(raw_cond)
            if field_name == "@last" and negated:
                # Direct support or negated branch swap; branch swap handles ALL negations!
                pass
            if_idx = len(instrs)
            instrs.append(
                MicroInstruction(
                    opcode=OP_IF,
                    scope_name=scope_name,
                    field_name=field_name,
                )
            )
            ctrl_stack.append(
                {"kind": "if", "if_idx": if_idx, "jmp_idx": None, "negated": negated}
            )
        elif part == "{@else}":
            if not ctrl_stack or ctrl_stack[-1]["kind"] != "if":
                raise ValueError("Unexpected {@else} without matching {@if}.")
            top = ctrl_stack[-1]
            jmp_idx = len(instrs)
            instrs.append(MicroInstruction(opcode=OP_JMP))
            top["jmp_idx"] = jmp_idx
            # False branch of OP_IF jumps right after OP_JMP
            instrs[top["if_idx"]].jump_delta = len(instrs) - top["if_idx"]
        elif part in ("{@end}", "{@endfor}", "{@endif}"):
            if not ctrl_stack:
                raise ValueError(f"Unexpected {part} with no open block.")
            top = ctrl_stack.pop()
            if top["kind"] == "for":
                loop_stack.pop()
                for_idx = top["for_idx"]
                next_idx = len(instrs)
                instrs.append(
                    MicroInstruction(
                        opcode=OP_NEXT,
                        loop_depth=top["depth"],
                        jump_delta=for_idx - next_idx,
                    )
                )
                instrs[for_idx].jump_delta = len(instrs) - for_idx
            else:
                if_idx = top["if_idx"]
                jmp_idx = top["jmp_idx"]
                end_idx = len(instrs)
                if not top["negated"]:
                    if jmp_idx is None:
                        instrs[if_idx].jump_delta = end_idx - if_idx
                    else:
                        instrs[jmp_idx].jump_delta = end_idx - jmp_idx
                else:
                    # Negated condition: swap true & false destinations cleanly!
                    if jmp_idx is None:
                        # {@if !cond} BODY {@end} -> if cond is True, skip BODY; if False, enter BODY!
                        # Rewrite as: OP_IF jumps +2 to BODY? Wait: OP_IF executes PC += 1 when True,
                        # and PC += jump_delta when False! Insert a 1-instruction OP_JMP right after OP_IF!
                        body_slice = instrs[if_idx + 1 :]
                        del instrs[if_idx + 1 :]
                        # If cond True -> PC += 1 hits OP_JMP -> skips past body!
                        # If cond False -> PC += 2 jumps straight into body_slice!
                        instrs[if_idx].jump_delta = 2
                        skip_jmp_idx = len(instrs)
                        instrs.append(MicroInstruction(opcode=OP_JMP))
                        instrs.extend(body_slice)
                        # Re-adjust any internal relative jumps? Internal jumps are strictly relative,
                        # so shifting body_slice by +1 preserves ALL internal jump_deltas identically!
                        instrs[skip_jmp_idx].jump_delta = len(instrs) - skip_jmp_idx
                    else:
                        # {@if !cond} THEN_A {@else} ELSE_B {@end} === {@if cond} ELSE_B {@else} THEN_A {@end}
                        then_slice = instrs[if_idx + 1 : jmp_idx]
                        else_slice = instrs[jmp_idx + 1 : end_idx]
                        del instrs[if_idx + 1 :]
                        instrs.extend(else_slice)
                        new_jmp_idx = len(instrs)
                        instrs.append(MicroInstruction(opcode=OP_JMP))
                        else_end = len(instrs)
                        instrs[if_idx].jump_delta = else_end - if_idx
                        instrs.extend(then_slice)
                        instrs[new_jmp_idx].jump_delta = len(instrs) - new_jmp_idx
        else:
            emit_literal_html(part)

    if ctrl_stack:
        raise ValueError(f"Unclosed control block(s) in template: {ctrl_stack}")

    instrs.append(MicroInstruction(opcode=OP_HALT))
    return instrs


class SymbolTable:
    """Interns output string chunks and maps integer symbol IDs <-> strings."""

    def __init__(self) -> None:
        self.id_to_str: list[str] = ["", "", ""]  # 0=NOP, 1=EOS, 2=EMPTY
        self.str_to_id: dict[str, int] = {"": SYM_EMPTY}

    def intern(self, text: str) -> int:
        if text == "":
            return SYM_EMPTY
        existing = self.str_to_id.get(text)
        if existing is not None:
            return existing
        idx = len(self.id_to_str)
        self.id_to_str.append(text)
        self.str_to_id[text] = idx
        return idx


class CarbonLoopedVM(nn.Module):
    """
    Complete Non-LLM Looped-Attention HTML Engine.
    Prefills compiled Template & Data KV-Caches and steps a single 32-dim
    CarbonBlock autoregressively until SYM_EOS is emitted.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.block = CarbonBlock(D_MODEL)
        self.to(device)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def parameter_report(self) -> dict[str, Any]:
        import io
        import zlib

        total_params = sum(p.numel() for p in self.parameters())
        nonzero_params = sum(int((p != 0).sum().item()) for p in self.parameters())
        matrix_nonzeros = sum(
            int((p != 0).sum().item())
            for name, p in self.named_parameters()
            if "norm" not in name
        )
        fp16_bytes = total_params * 2
        # Sparse COO storage size: (row_uint8, col_uint8, fp16_val) = 4 bytes per nonzero matrix weight
        sparse_matrix_bytes = matrix_nonzeros * 4
        buf = io.BytesIO()
        torch.save({k: v.half() for k, v in self.state_dict().items()}, buf)
        compressed_state_bytes = len(zlib.compress(buf.getvalue(), level=9))
        return {
            "d_model": D_MODEL,
            "d_ff": CarbonBlock.D_FF,
            "total_parameters": total_params,
            "nonzero_parameters": nonzero_params,
            "nonzero_matrix_weights": matrix_nonzeros,
            "dense_fp16_bytes": fp16_bytes,
            "sparse_fp16_bytes": sparse_matrix_bytes,
            "zlib_checkpoint_bytes": compressed_state_bytes,
        }

    def _prefill_kv_cache(
        self,
        instrs: list[MicroInstruction],
        data: dict[str, Any],
        symtab: SymbolTable,
    ) -> KVCache:
        device = self.device
        scope_map: dict[str, int] = {"$root": 1}
        field_map: dict[str, int] = {"$exists": 1, "$self": 2}

        def get_scope_id(name: str) -> int:
            if name not in scope_map:
                scope_map[name] = len(scope_map) + 1
            return scope_map[name]

        def get_field_id(name: str) -> int:
            if name not in field_map:
                field_map[name] = len(field_map) + 1
            return field_map[name]

        # Pre-register all scopes & fields referenced by the template
        raw_fields: set[tuple[str, str]] = set()
        escaped_fields: set[tuple[str, str]] = set()
        for ins in instrs:
            get_scope_id(ins.scope_name)
            if ins.field_name:
                get_field_id(ins.field_name)
            if ins.opcode == OP_VAR:
                if ins.raw_output:
                    raw_fields.add((ins.scope_name, ins.field_name))
                else:
                    escaped_fields.add((ins.scope_name, ins.field_name))

        # --- 1. Build Template KV-Cache (K_tmpl, V_tmpl) ---
        l_tmpl = len(instrs)
        positions = torch.arange(l_tmpl, dtype=torch.float32, device=device)
        k_tmpl = (
            self.block.instr_attn.rope.encode_positions(positions)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, L_tmpl, 6)

        v_rows: list[list[float]] = []
        for ins in instrs:
            # 8 bipolar opcodes
            ops = [-1.0] * 8
            ops[ins.opcode] = +1.0
            # 3 bipolar loop-depth indicators
            lds = [-1.0, -1.0, -1.0]
            lds[ins.loop_depth] = +1.0
            close_all = +1.0 if ins.close_all else -1.0
            s_id = float(get_scope_id(ins.scope_name))
            f_name = (
                f"$raw:{ins.field_name}"
                if (ins.opcode == OP_VAR and ins.raw_output)
                else ins.field_name
            )
            f_id = float(get_field_id(f_name)) if f_name else 0.0
            lit_sym = float(symtab.intern(ins.literal_text)) if ins.opcode == OP_LIT else 0.0
            close_sym = (
                float(symtab.intern(ins.close_tag_text)) if ins.close_tag_text else 0.0
            )
            tag_is_open = +1.0 if ins.tag_is_open else -1.0
            row = (
                ops
                + lds
                + [
                    close_all,
                    float(ins.jump_delta),
                    s_id,
                    f_id,
                    lit_sym,
                    close_sym,
                    float(ins.tag_delta),
                    tag_is_open,
                ]
            )
            v_rows.append(row)
        v_tmpl = torch.tensor(v_rows, dtype=torch.float32, device=device).view(
            1, 1, l_tmpl, 19
        )

        # --- 2. Flatten Data Context into Data KV-Cache (K_data, V_data) ---
        # Tuple entry: (scope_id, field_id, depth, i0, i1, i2, exists, truthy, sym_id)
        entries: list[tuple[int, int, int, int, int, int, float, float, float]] = []

        def add_value_entries(
            scope_name: str,
            key_name: str,
            val: Any,
            depth: int,
            indices: tuple[int, int, int],
        ) -> None:
            if isinstance(val, (list, tuple)):
                child_scope = (
                    key_name if scope_name == "$root" else f"{scope_name}.{key_name}"
                )
                if depth < 3:
                    populate_list(child_scope, list(val), depth + 1, indices)
                # Also record truthiness of the list itself (non-empty = truthy) for {@if items}
                s_id = get_scope_id(scope_name)
                f_id = get_field_id(key_name)
                truthy = +1.0 if len(val) > 0 else -1.0
                entries.append(
                    (s_id, f_id, depth, indices[0], indices[1], indices[2], +1.0, truthy, float(SYM_EMPTY))
                )
                return

            if isinstance(val, dict):
                # Support nested dotted dict access (e.g. user.profile.avatar)
                for sub_k, sub_v in val.items():
                    dotted = f"{key_name}.{sub_k}" if key_name else str(sub_k)
                    add_value_entries(scope_name, dotted, sub_v, depth, indices)
                s_id = get_scope_id(scope_name)
                f_id = get_field_id(key_name)
                truthy = +1.0 if bool(val) else -1.0
                entries.append(
                    (s_id, f_id, depth, indices[0], indices[1], indices[2], +1.0, truthy, float(SYM_EMPTY))
                )
                return

            s_id = get_scope_id(scope_name)
            truthy = +1.0 if bool(val) else -1.0
            str_val = "" if val is None or val is False else str(val)
            escaped_sym = float(symtab.intern(html.escape(str_val, quote=True)))
            entries.append(
                (
                    s_id,
                    get_field_id(key_name),
                    depth,
                    indices[0],
                    indices[1],
                    indices[2],
                    +1.0,
                    truthy,
                    escaped_sym,
                )
            )
            if (scope_name, key_name) in raw_fields:
                raw_sym = float(symtab.intern(str_val))
                entries.append(
                    (
                        s_id,
                        get_field_id(f"$raw:{key_name}"),
                        depth,
                        indices[0],
                        indices[1],
                        indices[2],
                        +1.0,
                        truthy,
                        raw_sym,
                    )
                )

        def populate_list(
            list_scope: str,
            items: list[Any],
            depth: int,
            parent_indices: tuple[int, int, int],
        ) -> None:
            s_id = get_scope_id(list_scope)
            exists_fid = get_field_id("$exists")
            n_items = len(items)
            for idx, item in enumerate(items):
                cur_indices = list(parent_indices)
                cur_indices[depth - 1] = idx
                idx_t = (cur_indices[0], cur_indices[1], cur_indices[2])

                # List item existence marker for OP_FOR
                entries.append(
                    (s_id, exists_fid, depth, idx_t[0], idx_t[1], idx_t[2], +1.0, +1.0, float(SYM_EMPTY))
                )

                # Built-in loop metadata fields (@index, @num, @first, @last, @not_last, @even, @odd)
                meta_fields: dict[str, Any] = {
                    "@index": idx,
                    "@num": idx + 1,
                    "@first": idx == 0,
                    "@last": idx == n_items - 1,
                    "@not_last": idx < n_items - 1,
                    "@even": (idx % 2) == 0,
                    "@odd": (idx % 2) == 1,
                }
                for mk, mv in meta_fields.items():
                    # Only materialize metadata fields actually referenced or cheap defaults
                    if mk in field_map or f"$raw:{mk}" in field_map:
                        add_value_entries(list_scope, mk, mv, depth, idx_t)

                if isinstance(item, dict):
                    for k, v in item.items():
                        add_value_entries(list_scope, str(k), v, depth, idx_t)
                else:
                    add_value_entries(list_scope, "$self", item, depth, idx_t)

        for top_k, top_v in data.items():
            add_value_entries("$root", str(top_k), top_v, 0, (0, 0, 0))

        # Encode Data KV-Cache tensors with constant squared norm ||K||^2 == 25.0
        # Entry 0 is the [NULL_SINK] Attention Sink: bias = 4.98725 -> ||K_active||^2 = 24.87266
        sink_bias = 4.98725
        sink_k = torch.zeros((1, 33), dtype=torch.float32, device=device)
        sink_k[0, 30] = sink_bias
        sink_k[0, 32] = math.sqrt(25.0 - sink_bias * sink_bias)
        sink_v = torch.tensor(
            [[-1.0, -1.0, float(SYM_EMPTY)]], dtype=torch.float32, device=device
        )

        if entries:
            e_tensor = torch.tensor(entries, dtype=torch.float32, device=device)
            # Columns: 0=s_id, 1=f_id, 2=depth, 3=i0, 4=i1, 5=i2, 6=exists, 7=truthy, 8=sym
            pos_5 = torch.stack(
                (
                    e_tensor[:, 0],
                    e_tensor[:, 1],
                    e_tensor[:, 3],
                    e_tensor[:, 4],
                    e_tensor[:, 5],
                ),
                dim=-1,
            )  # (N, 5)
            rope_5 = self.block.mem_attn.data_rope.encode_positions(pos_5)  # (N, 5, 6)
            depths = e_tensor[:, 2]  # (N,)
            # Zero out unused deeper loop index RoPE slots so outer/global scopes ignore inner counters
            mask_i0 = (depths >= 1.0).float().unsqueeze(-1)
            mask_i1 = (depths >= 2.0).float().unsqueeze(-1)
            mask_i2 = (depths >= 3.0).float().unsqueeze(-1)
            rope_5[:, 2, :] *= mask_i0
            rope_5[:, 3, :] *= mask_i1
            rope_5[:, 4, :] *= mask_i2
            rope_flat = rope_5.flatten(1)  # (N, 30)

            pad_bias = (3.0 - depths).unsqueeze(-1)  # (N, 1): ensures exact match Q.K == 5.0
            active_norm_sq = 2.0 + depths + (3.0 - depths).pow(2)  # <= 11.0 < 25.0
            q_orth_zero = torch.zeros_like(pad_bias)
            k_orth_pad = torch.sqrt(25.0 - active_norm_sq).unsqueeze(-1)
            k_entries = torch.cat(
                (rope_flat, pad_bias, q_orth_zero, k_orth_pad), dim=-1
            )  # (N, 33)
            v_entries = e_tensor[:, 6:9]  # (N, 3)

            k_data = torch.cat((sink_k, k_entries), dim=0).view(1, 1, -1, 33)
            v_data = torch.cat((sink_v, v_entries), dim=0).view(1, 1, -1, 3)
        else:
            k_data = sink_k.view(1, 1, 1, 33)
            v_data = sink_v.view(1, 1, 1, 3)

        # --- 3. Initialize Stack KV-Cache with [STACK_BOTTOM_SINK] at depth 0 ---
        zero_pos = torch.zeros((1,), dtype=torch.float32, device=device)
        k_stack = (
            self.block.mem_attn.stack_rope.encode_positions(zero_pos)
            .view(1, 1, 1, 6)
        )
        v_stack = torch.tensor(
            [[[[float(SYM_NOP), -1.0]]]], dtype=torch.float32, device=device
        )
        stack_steps = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=device)

        return KVCache(
            k_tmpl=k_tmpl,
            v_tmpl=v_tmpl,
            k_data=k_data,
            v_data=v_data,
            k_stack=k_stack,
            v_stack=v_stack,
            stack_steps=stack_steps,
            push_counter=0,
        )

    @torch.no_grad()
    def render(
        self,
        template: str,
        data: dict[str, Any] | None = None,
        *,
        max_steps: int = 100_000,
        trace: bool = False,
    ) -> RenderResult:
        """
        Compiles `template` and `data` into KV-caches and runs the weight-shared
        `CarbonBlock` in an autoregressive decode loop until `<EOS>` is emitted.
        """
        instrs = compile_template(template)
        symtab = SymbolTable()
        kv = self._prefill_kv_cache(instrs, data or {}, symtab)

        # Initial residual state h_0: only CH_ONE = 1.0, all counters at 0.0
        h = torch.zeros((1, 1, D_MODEL), dtype=torch.float32, device=self.device)
        h[0, 0, CH_ONE] = 1.0

        chunks: list[str] = []
        traces: list[StepTrace] = []
        emitted_tokens = 0

        for step in range(max_steps):
            pc_before = int(round(float(h[0, 0, CH_PC].item())))
            i0_before = int(round(float(h[0, 0, CH_I0].item())))
            i1_before = int(round(float(h[0, 0, CH_I1].item())))
            i2_before = int(round(float(h[0, 0, CH_I2].item())))
            depth_before = int(round(float(h[0, 0, CH_DEPTH].item())))

            h, meta = self.block(h, kv, record_attn=trace)

            # Push newly opened HTML tag to Stack KV-Cache if CH_PUSH_FLAG fired
            push_flag = float(h[0, 0, CH_PUSH_FLAG].item())
            if push_flag > 0.5:
                u2 = meta["u2"]
                new_depth_pos = torch.tensor(
                    [float(depth_before + 1)], dtype=torch.float32, device=self.device
                )
                k_entry = self.block.mem_attn.stack_rope.encode_positions(
                    new_depth_pos
                ).view(1, 1, 1, 6)
                close_sym_val = float(u2[0, 0, CH_CLOSE_SYM_PUSH].item())
                v_entry = torch.tensor(
                    [[[[close_sym_val, +1.0]]]],
                    dtype=torch.float32,
                    device=self.device,
                )
                kv.push_open_tag(k_entry, v_entry)

            sym_id = int(round(float(h[0, 0, CH_EMIT_SYM].item())))
            emitted_piece = ""
            if sym_id == SYM_EOS:
                if trace:
                    traces.append(
                        StepTrace(
                            step=step,
                            pc=pc_before,
                            opcode="HALT",
                            loop_indices=(i0_before, i1_before, i2_before),
                            tag_depth=depth_before,
                            instr_attn_peak=meta["instr_peak"],
                            data_attn_peak=meta["data_peak"],
                            data_attn_sink=meta["hit_sink"],
                            stack_attn_peak=meta["stack_peak"],
                            active_gates=meta["logic_gates"],
                            emitted="<EOS>",
                        )
                    )
                return RenderResult(
                    html="".join(chunks),
                    steps=step + 1,
                    emitted_tokens=emitted_tokens,
                    traces=traces,
                )

            if sym_id > SYM_EMPTY:
                emitted_piece = symtab.id_to_str[sym_id]
                chunks.append(emitted_piece)
                emitted_tokens += 1

            if trace:
                u2 = meta["u2"]
                op_vec = u2[0, 0, CH_OP_LIT : CH_OP_HALT + 1]
                op_idx = int(torch.argmax(op_vec).item())
                traces.append(
                    StepTrace(
                        step=step,
                        pc=pc_before,
                        opcode=OP_NAMES[op_idx],
                        loop_indices=(i0_before, i1_before, i2_before),
                        tag_depth=depth_before,
                        instr_attn_peak=meta["instr_peak"],
                        data_attn_peak=meta["data_peak"],
                        data_attn_sink=meta["hit_sink"],
                        stack_attn_peak=meta["stack_peak"],
                        active_gates=meta["logic_gates"],
                        emitted=emitted_piece if emitted_piece else "<NOP>",
                    )
                )

        raise RuntimeError(
            f"CarbonLoopedVM exceeded max_steps={max_steps} without reaching <EOS>."
        )
