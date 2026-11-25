"""
Blueprint-to-HTML-Template Synthesizer using Subword N-Gram Embeddings,
Scaled Dot-Product Cross-Attention, and CarbonLoopedVM Meta-Looping.

Creates reusable HTML templates (complete with {@for ...} loops, {@if ...} guards,
{{slot}} bindings, and stack-closed HTML5 tags) from ultra-compact blueprints—without
an LLM—by combining:
  1. SubwordNGramEmbedding + RMSNorm + F.scaled_dot_product_attention over an
     HTML Archetype KV-Memory Bank (semantic component routing).
  2. CarbonLoopedVM itself executing a meta-template that loops over the blueprint's
     sections, fields, and table columns to emit clean HTML template source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import SubwordNGramEmbedding
from .vm import CarbonLoopedVM


@dataclass
class ArchetypeMatch:
    """Diagnostic record showing how Cross-Attention routed a blueprint intent."""

    query_intent: str
    matched_archetype: str
    confidence: float


@dataclass
class SynthesizedTemplate:
    """Result of synthesizing a reusable HTML template from a compact blueprint."""

    template: str
    matches: list[ArchetypeMatch] = field(default_factory=list)
    meta_steps: int = 0


# Compact HTML Archetype Bank: (archetype_id, synonym prototype phrases)
ARCHETYPE_PROTOTYPES: list[tuple[str, tuple[str, ...]]] = [
    ("nav", ("nav", "navbar", "navigation", "topbar", "menubar", "header_links")),
    ("hero", ("hero", "banner", "jumbotron", "masthead", "headline", "intro_header")),
    ("stats", ("stats", "metrics", "kpi", "kpi_grid", "numbers", "scorecard", "counters")),
    ("cards", ("cards", "card_grid", "grid", "catalog", "gallery", "tiles", "features")),
    ("table", ("table", "datagrid", "rows", "spreadsheet", "ledger", "matrix", "records")),
    ("list", ("list", "ul", "bullets", "items", "checklist", "feed", "timeline")),
    ("pills", ("pills", "badges", "tags", "chips", "labels", "tokens")),
    ("footer", ("footer", "bottom", "credits", "colophon", "copyright")),
]

# Minimal modern dark/light CSS themes embedded only when full_page=True
THEMES: dict[str, str] = {
    "carbon": (
        ":root{--bg:#0b0f17;--panel:#131a26;--line:#233044;--txt:#e6edf7;"
        "--muted:#8fa3bf;--accent:#38bdf8;--pill:#1b283d}"
        "*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,sans-serif;"
        "background:var(--bg);color:var(--txt);line-height:1.5}"
        ".wrap{max-width:1040px;margin:0 auto;padding:2rem 1.25rem}"
        "nav.bar{display:flex;justify-content:space-between;align-items:center;"
        "padding:.85rem 1.25rem;border-bottom:1px solid var(--line);background:var(--panel)}"
        "nav.bar ul{display:flex;gap:1rem;list-style:none;margin:0;padding:0}"
        "nav.bar a{color:var(--muted);text-decoration:none;font-size:.92rem}"
        "header.hero{padding:2.25rem 0 1.5rem;border-bottom:1px solid var(--line);margin-bottom:1.75rem}"
        "header.hero h1{margin:0 0 .5rem;font-size:2rem;letter-spacing:-.02em}"
        "header.hero p{margin:0;color:var(--muted)}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin:1.25rem 0}"
        ".card,.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1.1rem}"
        ".stat .val{font-size:1.65rem;font-weight:700;color:var(--accent);margin:.25rem 0}"
        ".muted{color:var(--muted);font-size:.86rem}"
        "table.data{width:100%;border-collapse:collapse;margin:1.25rem 0;background:var(--panel);"
        "border:1px solid var(--line);border-radius:10px;overflow:hidden}"
        "table.data th,table.data td{padding:.7rem .9rem;text-align:left;border-bottom:1px solid var(--line)}"
        "table.data th{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}"
        "ul.clean{list-style:none;padding:0;margin:1rem 0}"
        "ul.clean li{padding:.6rem .85rem;border-bottom:1px solid var(--line)}"
        ".pills{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.6rem}"
        ".pill{background:var(--pill);color:var(--accent);border:1px solid var(--line);"
        "border-radius:999px;padding:.15rem .6rem;font-size:.76rem}"
        "footer.foot{margin-top:2.5rem;padding-top:1.25rem;border-top:1px solid var(--line);"
        "color:var(--muted);font-size:.85rem}"
    ),
    "paper": (
        ":root{--bg:#f8fafc;--panel:#ffffff;--line:#e2e8f0;--txt:#0f172a;"
        "--muted:#64748b;--accent:#0284c7;--pill:#e0f2fe}"
        "*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,sans-serif;"
        "background:var(--bg);color:var(--txt);line-height:1.5}"
        ".wrap{max-width:1040px;margin:0 auto;padding:2rem 1.25rem}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin:1.25rem 0}"
        ".card,.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1.1rem}"
        ".stat .val{font-size:1.65rem;font-weight:700;color:var(--accent)}"
        "table.data{width:100%;border-collapse:collapse;margin:1.25rem 0;background:var(--panel)}"
        "table.data th,table.data td{padding:.7rem .9rem;text-align:left;border-bottom:1px solid var(--line)}"
        ".pills{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.6rem}"
        ".pill{background:var(--pill);color:var(--accent);border-radius:999px;padding:.15rem .6rem;font-size:.76rem}"
    ),
}

# Meta-template executed by CarbonLoopedVM itself to synthesize reusable HTML templates!
# Notice it uses {@for}, {@if}, {!raw}, {{escaped}}, and </*> stack closing!
_META_SYNTH_TEMPLATE = """{# Meta-template rendered by CarbonLoopedVM to create HTML templates #}{@if full_page}<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{page_title_slot}}</title>
  {@if css}<style>{!css}</style>{@end}
</head>
<body>
{@end}{@for sec in sections}{@if sec.is_nav}<nav class="bar">
  <strong>{!sec.brand_slot}</strong>
  <ul>
  {!sec.loop_open}
    <li><a href="{!sec.href_slot}">{!sec.label_slot}</a></>
  {!sec.loop_close}
  </>
</>
{@end}{@if sec.is_open_wrap}<main class="wrap">
{@end}{@if sec.is_hero}<header class="hero">
  <h1>{!sec.h1_slot}</h1>
  {@if sec.sub_slot}{!sec.sub_guard_open}<p>{!sec.sub_slot}</p>{!sec.guard_close}{@end}
</>
{@end}{@if sec.is_stats}<section class="grid">
{!sec.loop_open}
  <div class="stat">
    <div class="muted">{!sec.f0_slot}</>
    <div class="val">{!sec.f1_slot}</>
    {@if sec.f2_slot}{!sec.f2_guard_open}<div class="muted">{!sec.f2_slot}</>{!sec.guard_close}{@end}
  </>
{!sec.loop_close}
</>
{@end}{@if sec.is_cards}<section class="grid">
{!sec.loop_open}
  <article class="card">
    {@if sec.heading_slot}<h3>{!sec.heading_slot}</h3>{@end}
    {@for fld in sec.body_fields}{@if fld.optional}{!fld.guard_open}<p>{!fld.slot}</p>{!fld.guard_close}{@else}<p>{!fld.slot}</p>{@end}
    {@end}{@if sec.has_subpills}<div class="pills">
    {!sec.sub_loop_open}
      <span class="pill">{!sec.sub_slot}</>
    {!sec.sub_loop_close}
    </>{@end}
  </>
{!sec.loop_close}
</>
{@end}{@if sec.is_table}<table class="data">
  <thead>
    <tr>
      {@for col in sec.columns}<th>{{col.header}}</th>{@end}
    </>
  </>
  <tbody>
  {!sec.loop_open}
    <tr>
      {@for col in sec.columns}<td>{!col.cell_slot}</td>{@end}
    </>
  {!sec.loop_close}
  </>
</>
{@end}{@if sec.is_list}<ul class="clean">
{!sec.loop_open}
  <li>{!sec.item_expr}</>
{!sec.loop_close}
</>
{@end}{@if sec.is_pills}<div class="pills">
{!sec.loop_open}
  <span class="pill">{!sec.item_expr}</>
{!sec.loop_close}
</>
{@end}{@if sec.is_footer}<footer class="foot">
  <p>{!sec.foot_slot}</p>
</>
{@end}{@end}</*>"""


class SemanticArchetypeRouter(nn.Module):
    """
    Routes component intent names (e.g. 'kpi_grid', 'datagrid', 'hero_banner', 'chips')
    to canonical HTML archetypes using SubwordNGramEmbedding + RMSNorm +
    Scaled Dot-Product Attention over prototype embeddings.
    """

    proto_keys: torch.Tensor

    def __init__(self, embed_dim: int = 32, temperature: float = 18.0) -> None:
        super().__init__()
        self.embed = SubwordNGramEmbedding(num_buckets=256, embed_dim=embed_dim)
        self.temperature = temperature
        self.archetype_names: list[str] = []
        proto_phrases: list[str] = []
        for arch_name, phrases in ARCHETYPE_PROTOTYPES:
            for p in phrases:
                self.archetype_names.append(arch_name)
                proto_phrases.append(p)
        with torch.no_grad():
            keys = self.embed(proto_phrases)  # (P, D)
            keys = F.normalize(keys, dim=-1)
        self.register_buffer("proto_keys", keys)

    @torch.no_grad()
    def route(self, intents: list[str]) -> list[ArchetypeMatch]:
        if not intents:
            return []
        q = F.normalize(self.embed(intents), dim=-1)  # (M, D)
        # Build 1-hot archetype value matrix so SDPA aggregates prototype support cleanly
        unique_archs = [a for a, _ in ARCHETYPE_PROTOTYPES]
        arch_to_idx = {a: i for i, a in enumerate(unique_archs)}
        v = torch.zeros(
            (len(self.archetype_names), len(unique_archs)),
            dtype=q.dtype,
            device=q.device,
        )
        for row_i, arch_name in enumerate(self.archetype_names):
            v[row_i, arch_to_idx[arch_name]] = 1.0

        # Genuine F.scaled_dot_product_attention over Archetype KV-Memory Bank
        attn_probs = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            self.proto_keys.unsqueeze(0),
            v.unsqueeze(0),
            scale=self.temperature,
        ).squeeze(0)  # (M, num_archetypes)

        results: list[ArchetypeMatch] = []
        for i, intent in enumerate(intents):
            best_idx = int(torch.argmax(attn_probs[i]).item())
            conf = float(attn_probs[i, best_idx].item())
            results.append(
                ArchetypeMatch(
                    query_intent=intent,
                    matched_archetype=unique_archs[best_idx],
                    confidence=conf,
                )
            )
        return results


def _singularize(plural: str) -> str:
    base = plural.split(".")[-1]
    if base.endswith("ies") and len(base) > 3:
        return base[:-3] + "y"
    if base.endswith("ses") and len(base) > 3:
        return base[:-2]
    if base.endswith("s") and len(base) > 1:
        return base[:-1]
    return "item"


class TemplateSynthesizer(nn.Module):
    """
    Synthesizes reusable HTML templates (with loops & conditionals) from compact
    declarative blueprints (dicts or concise blueprint specs) using SemanticArchetypeRouter
    + CarbonLoopedVM meta-execution.
    """

    def __init__(self, vm: CarbonLoopedVM | None = None) -> None:
        super().__init__()
        self.vm = vm if vm is not None else CarbonLoopedVM()
        self.router = SemanticArchetypeRouter().to(self.vm.device)
        self.eval()

    @staticmethod
    def parse_dsl(dsl: str) -> dict[str, Any]:
        """
        Parses a compact one-line or multi-line blueprint DSL into a structured spec dict.
        Example DSL:
            page title={{page_title}} theme=carbon
            hero: title, subtitle?
            stats(metrics): label, value, delta?
            table(events): time, service, region, status
            cards(teams) [pills=tags]: name, lead, mission?
            footer: copyright
        """
        spec: dict[str, Any] = {
            "full_page": False,
            "title_slot": "{{title}}",
            "theme": "carbon",
            "sections": [],
        }
        for raw_line in dsl.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("page"):
                spec["full_page"] = True
                for tok in line.split()[1:]:
                    if tok.startswith("title="):
                        val = tok.split("=", 1)[1].strip("\"'")
                        spec["title_slot"] = val if "{{" in val else f"{{{{{val}}}}}"
                    elif tok.startswith("theme="):
                        spec["theme"] = tok.split("=", 1)[1].strip("\"'")
                continue

            # Parse: component(loop_list) [pills=sublist]: field1, field2?, ...
            lhs, rhs = (line.split(":", 1) + [""])[:2]
            lhs = lhs.strip()
            fields = [f.strip() for f in rhs.split(",") if f.strip()]

            subpills: str | None = None
            if "[" in lhs and "]" in lhs:
                b_start = lhs.index("[")
                b_end = lhs.index("]")
                bracket = lhs[b_start + 1 : b_end].strip()
                if "=" in bracket:
                    _, subpills = [x.strip() for x in bracket.split("=", 1)]
                else:
                    subpills = bracket
                lhs = (lhs[:b_start] + lhs[b_end + 1 :]).strip()

            loop_over: str | None = None
            if "(" in lhs and ")" in lhs:
                p_start = lhs.index("(")
                p_end = lhs.index(")")
                loop_over = lhs[p_start + 1 : p_end].strip()
                if loop_over.startswith("for="):
                    loop_over = loop_over[4:].strip()
                kind = lhs[:p_start].strip()
            else:
                kind = lhs

            sec: dict[str, Any] = {"kind": kind, "fields": fields}
            if loop_over:
                sec["for"] = loop_over
            if subpills:
                sec["subpills"] = subpills
            spec["sections"].append(sec)
        return spec

    def synthesize(self, blueprint: str | dict[str, Any]) -> SynthesizedTemplate:
        """
        Routes every section in `blueprint` through `SemanticArchetypeRouter` and
        executes `_META_SYNTH_TEMPLATE` inside `CarbonLoopedVM` to emit a clean,
        reusable HTML template with `{@for}` loops and `{{...}}` variable slots.
        """
        spec = self.parse_dsl(blueprint) if isinstance(blueprint, str) else blueprint
        raw_sections: list[dict[str, Any]] = list(spec.get("sections", []))
        intents = [str(s.get("kind", "cards")) for s in raw_sections]
        matches = self.router.route(intents)

        full_page = bool(spec.get("full_page", True))
        theme_name = str(spec.get("theme", "carbon"))
        css = THEMES.get(theme_name, "") if full_page else ""

        prepared_sections: list[dict[str, Any]] = []
        opened_wrap = False

        for sec, match in zip(raw_sections, matches):
            arch = match.matched_archetype
            fields = [str(f) for f in sec.get("fields", [])]
            loop_list = str(sec.get("for", f"{arch}_items"))
            item_var = str(sec.get("var", _singularize(loop_list)))

            # Open <main class="wrap"> right after nav (or before first body section) on full pages
            if full_page and not opened_wrap and arch != "nav":
                prepared_sections.append({"is_open_wrap": True})
                opened_wrap = True

            def slot_for(raw_f: str, scoped: bool) -> tuple[str, str, bool]:
                optional = raw_f.endswith("?")
                clean = raw_f[:-1].strip() if optional else raw_f.strip()
                if clean == "$self":
                    expr = item_var
                elif scoped:
                    expr = f"{item_var}.{clean}"
                else:
                    expr = clean
                return expr, f"{{{{{expr}}}}}", optional

            entry: dict[str, Any] = {
                f"is_{arch}": True,
                "loop_open": f"{{@for {item_var} in {loop_list}}}",
                "loop_close": "{@end}",
                "guard_close": "{@end}",
            }

            if arch == "nav":
                brand_f = fields[0] if fields else "brand"
                href_f = fields[1] if len(fields) > 1 else "href"
                label_f = fields[2] if len(fields) > 2 else "label"
                _, entry["brand_slot"], _ = slot_for(brand_f, scoped=False)
                _, entry["href_slot"], _ = slot_for(href_f, scoped=True)
                _, entry["label_slot"], _ = slot_for(label_f, scoped=True)

            elif arch == "hero":
                h1_f = fields[0] if fields else "title"
                _, entry["h1_slot"], _ = slot_for(h1_f, scoped=False)
                if len(fields) > 1:
                    sub_expr, sub_slot, sub_opt = slot_for(fields[1], scoped=False)
                    entry["sub_slot"] = sub_slot
                    entry["sub_guard_open"] = f"{{@if {sub_expr}}}" if sub_opt else ""
                    if not sub_opt:
                        entry["guard_close"] = ""

            elif arch == "stats":
                f0 = fields[0] if fields else "label"
                f1 = fields[1] if len(fields) > 1 else "value"
                _, entry["f0_slot"], _ = slot_for(f0, scoped=True)
                _, entry["f1_slot"], _ = slot_for(f1, scoped=True)
                if len(fields) > 2:
                    f2_expr, f2_slot, f2_opt = slot_for(fields[2], scoped=True)
                    entry["f2_slot"] = f2_slot
                    entry["f2_guard_open"] = f"{{@if {f2_expr}}}" if f2_opt else ""
                    if not f2_opt:
                        entry["guard_close"] = ""

            elif arch == "cards":
                if fields:
                    _, entry["heading_slot"], _ = slot_for(fields[0], scoped=True)
                body_fields: list[dict[str, Any]] = []
                for bf in fields[1:]:
                    b_expr, b_slot, b_opt = slot_for(bf, scoped=True)
                    body_fields.append(
                        {
                            "slot": b_slot,
                            "optional": b_opt,
                            "guard_open": f"{{@if {b_expr}}}",
                            "guard_close": "{@end}",
                        }
                    )
                entry["body_fields"] = body_fields
                subpills = sec.get("subpills")
                if subpills:
                    sub_var = _singularize(str(subpills))
                    entry["has_subpills"] = True
                    entry["sub_loop_open"] = f"{{@for {sub_var} in {item_var}.{subpills}}}"
                    entry["sub_loop_close"] = "{@end}"
                    entry["sub_slot"] = f"{{{{{sub_var}}}}}"

            elif arch == "table":
                cols = fields if fields else ["name", "value"]
                col_dicts: list[dict[str, str]] = []
                for c in cols:
                    clean_c = c.rstrip("?").strip()
                    col_dicts.append(
                        {
                            "header": clean_c.replace("_", " ").title(),
                            "cell_slot": f"{{{{{item_var}.{clean_c}}}}}",
                        }
                    )
                entry["columns"] = col_dicts

            elif arch in ("list", "pills"):
                if not fields or fields == ["$self"]:
                    entry["item_expr"] = f"{{{{{item_var}}}}}"
                else:
                    parts = [slot_for(f, scoped=True)[1] for f in fields]
                    entry["item_expr"] = " — ".join(parts)

            elif arch == "footer":
                foot_f = fields[0] if fields else "copyright"
                _, entry["foot_slot"], _ = slot_for(foot_f, scoped=False)

            prepared_sections.append(entry)

        meta_context = {
            "full_page": full_page,
            "page_title_slot": str(spec.get("title_slot", "{{title}}")),
            "css": css,
            "sections": prepared_sections,
        }
        rendered = self.vm.render(_META_SYNTH_TEMPLATE, meta_context)
        return SynthesizedTemplate(
            template=rendered.html,
            matches=matches,
            meta_steps=rendered.steps,
        )
