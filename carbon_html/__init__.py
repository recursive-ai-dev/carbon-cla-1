"""
Carbon HTML Engine (`carbon_html`) — The Sub-Kilobyte Non-LLM HTML Template Engine
Built Exclusively From LLM Components.
"""

from .primitives import (
    ALiBiRecencyBias,
    KVCache,
    RMSNorm,
    RotaryEmbedding,
    SubwordNGramEmbedding,
    SwiGLU,
)
from .synthesizer import (
    ArchetypeMatch,
    SemanticArchetypeRouter,
    SynthesizedTemplate,
    TemplateSynthesizer,
)
from .vm import (
    CarbonBlock,
    CarbonLoopedVM,
    RenderResult,
    StepTrace,
    compile_template,
)


class CarbonEngine:
    """
    High-level facade uniting:
      1. `synthesize(blueprint)` -> creates reusable HTML templates with loops
      2. `render(template, data)` -> executes loops & renders HTML via CarbonLoopedVM
      3. `create_and_render(blueprint, data)` -> end-to-end blueprint -> template -> HTML
    """

    def __init__(self, device: str = "cpu") -> None:
        self.vm = CarbonLoopedVM(device=device)
        self.synthesizer = TemplateSynthesizer(vm=self.vm)

    def synthesize(self, blueprint: str | dict) -> SynthesizedTemplate:
        return self.synthesizer.synthesize(blueprint)

    def render(
        self, template: str, data: dict | None = None, *, trace: bool = False
    ) -> RenderResult:
        return self.vm.render(template, data, trace=trace)

    def create_and_render(
        self, blueprint: str | dict, data: dict, *, trace: bool = False
    ) -> tuple[SynthesizedTemplate, RenderResult]:
        tmpl = self.synthesize(blueprint)
        out = self.render(tmpl.template, data, trace=trace)
        return tmpl, out

    def stats(self) -> dict:
        vm_stats = self.vm.parameter_report()
        router_buffers = sum(b.numel() for b in self.synthesizer.router.buffers())
        return {
            **vm_stats,
            "router_buffer_floats": router_buffers,
        }


__all__ = [
    "ALiBiRecencyBias",
    "ArchetypeMatch",
    "CarbonBlock",
    "CarbonEngine",
    "CarbonLoopedVM",
    "KVCache",
    "RMSNorm",
    "RenderResult",
    "RotaryEmbedding",
    "SemanticArchetypeRouter",
    "StepTrace",
    "SubwordNGramEmbedding",
    "SwiGLU",
    "SynthesizedTemplate",
    "TemplateSynthesizer",
    "compile_template",
]
