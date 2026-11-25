"""
Showcase & Self-Hosted Mechanistic Inspector Generator for Carbon HTML Engine.

Demonstrates both halves of the engine:
  1. Synthesizing a complete multi-section HTML template (with loops, nested pill loops,
     conditional badges, and a trace datagrid) from a 9-line Blueprint using
     SubwordNGramEmbedding + RMSNorm + Scaled Dot-Product Attention + CarbonLoopedVM.
  2. Recording real step-by-step tensor telemetry (RoPE instruction attention peaks,
     StreamingLLM Null-Sink hits, ALiBi stack attention, and SwiGLU gate counts)
     on a nested-loop sample, then rendering the ENTIRE inspector HTML report
     THROUGH CarbonLoopedVM itself!
"""

from __future__ import annotations

import json
from pathlib import Path

from carbon_html import CarbonEngine


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = CarbonEngine()
    stats = engine.stats()

    # 1. Record a real mechanistic trace on a compact nested-loop HTML snippet
    probe_template = (
        "<section><h2>{{heading}}</h2>"
        "<ul>{@for group in groups}"
        "<li><b>{{group.name}}</b>: "
        "{@for item in group.items}<span>{{item}}{@if @not_last}, {@end}</>{@end}"
        "</>{@end}</*>"""
    )
    probe_data = {
        "heading": "Tensor Loop Probe",
        "groups": [
            {"name": "Attention", "items": ["RoPE", "ALiBi", "QK-RMSNorm"]},
            {"name": "FFN", "items": ["SwiGLU"]},
        ],
    }
    probe_result = engine.render(probe_template, probe_data, trace=True)

    # 2. Synthesize a full dashboard HTML template from a 7-line declarative Blueprint
    blueprint = """
    page title=page_title theme=carbon
    topbar(links): brand, href, label
    hero_banner: headline, subheadline?
    kpi_grid(kpis): label, value, note?
    catalog(components) [pills=primitives]: name, role
    datagrid(trace_rows): step, pc, opcode, loops, depth, attn_peaks, gates, emitted
    colophon: footer_note
    """
    synthesized = engine.synthesize(blueprint)
    tmpl_path = out_dir / "synthesized_dashboard.html.tmpl"
    tmpl_path.write_text(synthesized.template, "utf-8")

    # 3. Render the synthesized dashboard template using the engine's own tensor loop!
    trace_rows = [
        {
            "step": tr.step,
            "pc": tr.pc,
            "opcode": tr.opcode,
            "loops": f"({tr.loop_indices[0]}, {tr.loop_indices[1]}, {tr.loop_indices[2]})",
            "depth": tr.tag_depth,
            "attn_peaks": (
                f"I:{tr.instr_attn_peak:.3f} "
                f"D:{'SINK' if tr.data_attn_sink else f'{tr.data_attn_peak:.3f}'} "
                f"S:{tr.stack_attn_peak:.3f}"
            ),
            "gates": tr.active_gates,
            "emitted": tr.emitted,
        }
        for tr in probe_result.traces
    ]

    dashboard_data = {
        "page_title": "Carbon CLA-1 — Non-LLM Looped-Attention HTML Engine",
        "brand": "CARBON // CLA-1",
        "links": [
            {"href": "#kpis", "label": "Footprint"},
            {"href": "#components", "label": "LLM Primitives"},
            {"href": "#trace", "label": "Tensor Trace"},
        ],
        "headline": "Sub-Kilobyte Looped-Attention HTML Engine",
        "subheadline": (
            "Not an LLM — zero text training, zero hallucinations. "
            "100% built from RoPE, ALiBi, QK-RMSNorm, KV-Cache, Attention Sinks, and SwiGLU."
        ),
        "kpis": [
            {
                "label": "Active Nonzero Weights",
                "value": str(stats["nonzero_parameters"]),
                "note": f"{stats['sparse_fp16_bytes']} bytes sparse fp16",
            },
            {
                "label": "Residual Width (d_model)",
                "value": f"{stats['d_model']} floats",
                "note": f"SwiGLU d_ff = {stats['d_ff']}",
            },
            {
                "label": "Template Meta-Synth Steps",
                "value": str(synthesized.meta_steps),
                "note": "Engine synthesized its own template!",
            },
            {
                "label": "Dashboard Render Steps",
                "value": "Live",
                "note": f"{len(trace_rows)} probe trace rows looped",
            },
        ],
        "components": [
            {
                "name": "Head 0: Instruction Fetch",
                "role": "Queries Template KV-Cache at Program Counter via 3-frequency RoPE + QK-RMSNorm.",
                "primitives": ["RoPE", "QK-RMSNorm", "SDPA"],
            },
            {
                "name": "Head 1: Tuple Data & Sink",
                "role": "Indexes 3D nested loop scopes in Data KV-Cache; Null Attention Sink catches loop exits.",
                "primitives": ["Tuple-RoPE", "Attention-Sink", "KV-Cache"],
            },
            {
                "name": "Head 2: LIFO Tag Stack",
                "role": "Matches open HTML tag depth via RoPE and resolves most recent tag via ALiBi recency.",
                "primitives": ["ALiBi", "RoPE", "Dynamic-KV"],
            },
            {
                "name": "SwiGLU Control FFN",
                "role": "Bias-free Llama FFN executing exact boolean AND gates, loop jumps, and counter resets.",
                "primitives": ["SwiGLU", "SiLU", "Residual-Stream"],
            },
            {
                "name": "Archetype Router",
                "role": "Maps fuzzy blueprint intents to HTML archetypes via Subword 3-gram Embeddings + SDPA.",
                "primitives": ["Subword-NGram", "RMSNorm", "Cross-Attention"],
            },
        ],
        "trace_rows": trace_rows,
        "footer_note": (
            f"Synthesized & loop-rendered deterministically by Carbon CLA-1 "
            f"({stats['nonzero_parameters']} nonzero weights, {stats['total_parameters']} dense params)."
        ),
    }

    rendered_dashboard = engine.render(synthesized.template, dashboard_data)
    html_path = out_dir / "inspector_dashboard.html"
    html_path.write_text(rendered_dashboard.html, "utf-8")

    summary = {
        "engine_stats": stats,
        "archetype_routing": [
            {
                "intent": m.query_intent,
                "archetype": m.matched_archetype,
                "confidence": round(m.confidence, 4),
            }
            for m in synthesized.matches
        ],
        "probe_html": probe_result.html,
        "probe_steps": probe_result.steps,
        "synthesized_template_path": str(tmpl_path),
        "rendered_dashboard_path": str(html_path),
        "dashboard_render_steps": rendered_dashboard.steps,
        "dashboard_bytes": len(rendered_dashboard.html.encode("utf-8")),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
