"""
Command-line interface for Carbon HTML Engine (`python -m carbon_html.cli`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import CarbonEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="carbon-html",
        description="Sub-kilobyte non-LLM HTML template creating & loop-rendering engine built from LLM components.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="Print parameter count and weight byte footprint.")

    p_synth = sub.add_parser("synthesize", help="Synthesize a reusable HTML template from a blueprint.")
    p_synth.add_argument("blueprint", help="Blueprint string or path to .bp file.")
    p_synth.add_argument("-o", "--output", help="Optional path to write synthesized template.")

    p_render = sub.add_parser("render", help="Render an HTML template with JSON data using CarbonLoopedVM.")
    p_render.add_argument("template", help="Template string or path to template file.")
    p_render.add_argument("-d", "--data", default="{}", help="JSON string or path to .json file.")
    p_render.add_argument("--trace", action="store_true", help="Print step-by-step attention & gate telemetry.")
    p_render.add_argument("-o", "--output", help="Optional path to write rendered HTML.")

    args = parser.parse_args(argv)
    engine = CarbonEngine()

    if args.cmd == "stats":
        print(json.dumps(engine.stats(), indent=2))
        return 0

    if args.cmd == "synthesize":
        bp_path = Path(args.blueprint)
        bp_text = bp_path.read_text("utf-8") if bp_path.is_file() else args.blueprint
        res = engine.synthesize(bp_text)
        if args.output:
            Path(args.output).write_text(res.template, "utf-8")
        else:
            sys.stdout.write(res.template + "\n")
        return 0

    if args.cmd == "render":
        t_path = Path(args.template)
        tmpl = t_path.read_text("utf-8") if t_path.is_file() else args.template
        d_path = Path(args.data)
        data = json.loads(d_path.read_text("utf-8") if d_path.is_file() else args.data)
        out = engine.render(tmpl, data, trace=args.trace)
        if args.trace:
            for tr in out.traces:
                print(
                    f"[step {tr.step:03d}] pc={tr.pc:02d} op={tr.opcode:<5} "
                    f"loop={tr.loop_indices} depth={tr.tag_depth} "
                    f"attn=({tr.instr_attn_peak:.4f},{tr.data_attn_peak:.4f},{tr.stack_attn_peak:.4f}) "
                    f"gates={tr.active_gates} -> {tr.emitted!r}",
                    file=sys.stderr,
                )
        if args.output:
            Path(args.output).write_text(out.html, "utf-8")
        else:
            sys.stdout.write(out.html + "\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
