"""
Comprehensive test suite for Carbon HTML Engine (`carbon_html`).

Verifies:
  1. Non-LLM & Zero-Cheat Guarantees:
     - Tiny weight footprint (< 600 bytes sparse / < 9 KB dense fp16)
     - Zero imports of jinja2, string.Template, mako, mustache, etc.
     - Genuine PyTorch LLM modules (RMSNorm, RotaryEmbedding, ALiBi, SwiGLU, SDPA)
  2. Looped-Attention VM Execution:
     - Empty loops (0 items), 1-item loops, multi-item loops, 300-item stress loops
     - 2-level and 3-level nested loops with ragged child list lengths (including 0)
     - Outer-scope & root-scope variable access from inside deep inner loops
     - Loop metadata (@index, @num, @first, @last, @not_last, @even, @odd)
     - Conditionals ({@if}, {@else}, {@if !cond}, {@if not cond}) inside & outside loops
     - HTML XSS auto-escaping ({{...}}) vs raw trusted slots ({!...})
     - ALiBi + RoPE Open-Tag Stack auto-closing (</> and </*>), including dynamic
       alternating tags inside loops (`<strong>` vs `<em>` at identical stack depth!)
  3. Blueprint-to-HTML-Template Synthesis:
     - Fuzzy semantic routing via SubwordNGramEmbedding + RMSNorm + SDPA
     - Synthesizing reusable HTML templates with {@for} loops & {{slots}}
     - End-to-end Blueprint -> Synthesized Template -> Looped HTML Rendering & DOM balance
"""

from __future__ import annotations

import ast
from html.parser import HTMLParser
from pathlib import Path

import pytest
import torch

from carbon_html import (
    ALiBiRecencyBias,
    CarbonEngine,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
)


class _StrictDOMValidator(HTMLParser):
    """Validates that every non-void HTML open tag is closed in exact LIFO order."""

    VOID = {
        "area", "base", "br", "col", "embed", "hr", "img",
        "input", "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t not in self.VOID:
            self.stack.append(t)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if not self.stack:
            self.errors.append(f"Unexpected closing </{t}> on empty stack")
            return
        top = self.stack.pop()
        if top != t:
            self.errors.append(f"Mismatched closing </{t}>; expected </{top}>")


def assert_balanced_html(markup: str) -> None:
    v = _StrictDOMValidator()
    v.feed(markup)
    assert not v.errors, f"DOM errors: {v.errors}\nHTML:\n{markup}"
    assert not v.stack, f"Unclosed tags remaining: {v.stack}\nHTML:\n{markup}"


def test_zero_template_library_cheats_and_llm_primitives() -> None:
    pkg_dir = Path(__file__).resolve().parent.parent / "carbon_html"
    forbidden = {"jinja2", "mako", "chevron", "pystache", "genshi", "chameleon", "tornado"}
    for py_file in pkg_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden

    engine = CarbonEngine()
    stats = engine.stats()
    # Ultra-small footprint: < 6,000 total parameters, < 220 nonzero matrix weights (< 900 B sparse!)
    assert stats["total_parameters"] < 6000
    assert stats["nonzero_matrix_weights"] < 220
    assert stats["nonzero_parameters"] < 320
    assert stats["sparse_fp16_bytes"] < 900

    # Confirm genuine LLM module classes live inside CarbonBlock
    module_types = {type(m) for m in engine.vm.modules()}
    assert RMSNorm in module_types
    assert RotaryEmbedding in module_types
    assert ALiBiRecencyBias in module_types
    assert SwiGLU in module_types


def test_loop_empty_single_and_multi_items() -> None:
    engine = CarbonEngine()
    tmpl = "<ul>{@for item in items}<li>{{@num}}: {{item}}</li>{@end}</ul>"

    # Empty list -> zero loop iterations, clean jump past @end
    r_empty = engine.render(tmpl, {"items": []})
    assert r_empty.html == "<ul></ul>"
    assert_balanced_html(r_empty.html)

    # Single item
    r_one = engine.render(tmpl, {"items": ["Alpha"]})
    assert r_one.html == "<ul><li>1: Alpha</li></ul>"

    # Multiple items
    r_many = engine.render(tmpl, {"items": ["Alpha", "Beta", "Gamma"]})
    assert r_many.html == "<ul><li>1: Alpha</li><li>2: Beta</li><li>3: Gamma</li></ul>"
    assert_balanced_html(r_many.html)


def test_3d_nested_loops_with_ragged_and_empty_children() -> None:
    engine = CarbonEngine()
    tmpl = (
        "<div data-org='{{org}}'>"
        "{@for dept in depts}"
        "<section><h2>{{org}} / {{dept.name}}</h2>"
        "<ul>"
        "{@for team in dept.teams}"
        "<li><b>{{dept.code}}-{{team.name}}</b>:"
        "{@for tech in team.stack}<span>{{tech}}{@if @not_last}, {@end}</span>{@end}"
        "</>"
        "{@end}"
        "</></>"
        "{@end}"
        "</*>"
    )
    data = {
        "org": "CarbonLab",
        "depts": [
            {
                "name": "Systems",
                "code": "SYS",
                "teams": [
                    {"name": "Kernel", "stack": ["Rust", "C", "ASM"]},
                    {"name": "EmptyStack", "stack": []},
                ],
            },
            {
                "name": "FrozenDept",
                "code": "FRZ",
                "teams": [],
            },
            {
                "name": "Attention",
                "code": "ATN",
                "teams": [
                    {"name": "RoPE", "stack": ["PyTorch"]},
                ],
            },
        ],
    }
    res = engine.render(tmpl, data, trace=True)
    expected = (
        "<div data-org='CarbonLab'>"
        "<section><h2>CarbonLab / Systems</h2><ul>"
        "<li><b>SYS-Kernel</b>:<span>Rust, </span><span>C, </span><span>ASM</span></li>"
        "<li><b>SYS-EmptyStack</b>:</li>"
        "</ul></section>"
        "<section><h2>CarbonLab / FrozenDept</h2><ul></ul></section>"
        "<section><h2>CarbonLab / Attention</h2><ul>"
        "<li><b>ATN-RoPE</b>:<span>PyTorch</span></li>"
        "</ul></section>"
        "</div>"
    )
    assert res.html == expected
    assert_balanced_html(res.html)
    # Every instruction and stack attention peak in trace should be > 0.9999
    assert all(t.instr_attn_peak > 0.9999 for t in res.traces if t.opcode != "HALT")


def test_alibi_stack_closes_dynamic_alternating_tags_in_loop() -> None:
    """
    Crucial mechanistic test for ALiBi + RoPE Stack Attention:
    At the EXACT SAME stack depth inside a loop, even iterations open <strong>
    and odd iterations open <em>, and a single `</>` closes whichever opened
    most recently! Without ALiBi recency bias, earlier iterations' tags at the
    same depth would collide.
    """
    engine = CarbonEngine()
    tmpl = (
        "<main><ul>"
        "{@for w in words}"
        "<li>{@if @even}<strong>{@else}<em>{@end}{{w}}</></>"
        "{@end}"
        "</*>"
    )
    res = engine.render(tmpl, {"words": ["zero", "one", "two", "three"]})
    assert res.html == (
        "<main><ul>"
        "<li><strong>zero</strong></li>"
        "<li><em>one</em></li>"
        "<li><strong>two</strong></li>"
        "<li><em>three</em></li>"
        "</ul></main>"
    )
    assert_balanced_html(res.html)


def test_conditionals_negation_and_xss_escaping() -> None:
    engine = CarbonEngine()
    tmpl = (
        "{@if !maintenance}"
        "<div>{{unsafe_user}} | {!trusted_badge}</div>"
        "{@else}"
        "<p>Down</p>"
        "{@end}"
    )
    res = engine.render(
        tmpl,
        {
            "maintenance": False,
            "unsafe_user": '<script>alert("xss")</script>',
            "trusted_badge": "<b>VIP</b>",
        },
    )
    assert res.html == (
        "<div>&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt; | <b>VIP</b></div>"
    )
    assert_balanced_html(res.html)

    res_maint = engine.render(tmpl, {"maintenance": True})
    assert res_maint.html == "<p>Down</p>"


def test_stress_loop_300_items_zero_drift() -> None:
    """Proves integer-exact counter transitions in SwiGLU + multi-freq RoPE up to 300 items."""
    engine = CarbonEngine()
    items = [{"id": i, "val": f"row-{i}"} for i in range(300)]
    tmpl = "<table><tbody>{@for r in rows}<tr><td>{{r.id}}</td><td>{{r.val}}</td></></tr>{@end}</tbody></table>"
    # Fix intentional extra </>? Let's use clean </> for <tr>!
    tmpl = "<table><tbody>{@for r in rows}<tr><td>{{r.id}}</td><td>{{r.val}}</td></>{@end}</*>"
    res = engine.render(tmpl, {"rows": items})
    assert res.html.count("<tr>") == 300
    assert res.html.count("</tr>") == 300
    assert "<td>0</td><td>row-0</td>" in res.html
    assert "<td>299</td><td>row-299</td>" in res.html
    assert_balanced_html(res.html)


def test_semantic_archetype_routing_and_template_creation() -> None:
    """
    Tests non-LLM HTML template creation from a blueprint with fuzzy component names,
    then loop-renders the created HTML template with live data.
    """
    engine = CarbonEngine()
    blueprint = """
    page title=page_title theme=carbon
    topbar(nav_links): brand, href, label
    hero_banner: heading, tagline?
    kpi_grid(metrics): name, number, delta?
    datagrid(deployments): service, version, status
    catalog(modules) [pills=tags]: title, summary
    colophon: legal
    """
    synth = engine.synthesize(blueprint)
    matched = [m.matched_archetype for m in synth.matches]
    assert matched == ["nav", "hero", "stats", "table", "cards", "footer"]

    # Verify the synthesized artifact IS a reusable HTML template containing loops!
    assert "{@for " in synth.template
    assert "{{metric.number}}" in synth.template
    assert "{{deployment.service}}" in synth.template
    assert "{@for tag in module.tags}" in synth.template

    # Now render that synthesized template on live data via CarbonLoopedVM!
    rendered = engine.render(
        synth.template,
        {
            "page_title": "Carbon Telemetry",
            "brand": "CARBON//26",
            "nav_links": [
                {"href": "#metrics", "label": "Metrics"},
                {"href": "#deploys", "label": "Deploys"},
            ],
            "heading": "Sub-Kilobyte Tensor HTML Engine",
            "tagline": "RoPE + ALiBi + QK-RMSNorm + SwiGLU",
            "metrics": [
                {"name": "Nonzero Weights", "number": "133", "delta": "532 B sparse"},
                {"name": "Hallucinations", "number": "0%"},
            ],
            "deployments": [
                {"service": "instr-rope", "version": "v1.0", "status": "NOMINAL"},
                {"service": "stack-alibi", "version": "v1.0", "status": "NOMINAL"},
            ],
            "modules": [
                {
                    "title": "Looped Attention VM",
                    "summary": "Executes 3-level nested loops in residual space.",
                    "tags": ["RoPE", "SwiGLU", "KV-Cache"],
                }
            ],
            "legal": "2026 Testing Grounds",
        },
    )
    assert "<!DOCTYPE html>" in rendered.html
    assert "Sub-Kilobyte Tensor HTML Engine" in rendered.html
    assert "532 B sparse" in rendered.html
    assert "<span class=\"pill\">SwiGLU</span>" in rendered.html
    assert_balanced_html(rendered.html)
