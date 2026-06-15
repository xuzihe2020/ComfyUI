from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
from comfy_extras.color_util import normalize_palette


class BuildJsonPromptIdeogram(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        color_palette = io.Colors.Input(
            "color_palette",
            tooltip="Style color palette.",
        )
        return io.Schema(
            node_id="BuildJsonPromptIdeogram",
            display_name="Build JSON Prompt (Ideogram)",
            category="image/ideogram",
            description="Assemble the Ideogram 4 caption from Create Bounding Boxes elements plus the background and style fields.",
            inputs=[
                io.ComfyList.Input("element", tooltip="Caption elements from Create Bounding Boxes."),
                io.String.Input("high_level_description", multiline=True, default="",
                                tooltip="Optional one-line overview of the whole image (blank = omitted)."),
                io.String.Input("background", multiline=True, default="",
                                tooltip="Scene background description."),
                io.DynamicCombo.Input("style", options=[
                    io.DynamicCombo.Option("none", []),
                    io.DynamicCombo.Option("photo", [io.String.Input("photo", default="")]),
                    io.DynamicCombo.Option("art_style", [io.String.Input("art_style", default="")]),
                ]),
                io.String.Input("aesthetics", default="", tooltip="Style descriptor. Sent even when blank once a style is chosen."),
                io.String.Input("lighting", default="", tooltip="Style descriptor. Sent even when blank once a style is chosen."),
                io.String.Input("medium", default="", tooltip="Style descriptor. Sent even when blank once a style is chosen."),
                color_palette,
            ],
            outputs=[io.ComfyDict.Output(display_name="prompt")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, element, style, high_level_description="", background="",
                aesthetics="", lighting="", medium="", color_palette=None) -> io.NodeOutput:
        elements = element if isinstance(element, list) else []
        kind = style.get("style", "none") if isinstance(style, dict) else "none"
        photo = style.get("photo", "") if isinstance(style, dict) else ""
        art_style = style.get("art_style", "") if isinstance(style, dict) else ""
        palette = normalize_palette(color_palette or [])

        caption: dict = {}
        if high_level_description.strip():
            caption["high_level_description"] = high_level_description
        if kind != "none":
            style_desc: dict = {"aesthetics": aesthetics, "lighting": lighting}
            if kind == "photo":
                style_desc["photo"] = photo
                style_desc["medium"] = medium
            else:
                style_desc["medium"] = medium
                style_desc["art_style"] = art_style
            if palette:
                style_desc["color_palette"] = palette
            caption["style_description"] = style_desc
        caption["compositional_deconstruction"] = {
            "background": background,
            "elements": elements,
        }
        return io.NodeOutput(caption)


class JsonPromptExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [BuildJsonPromptIdeogram]


async def comfy_entrypoint() -> JsonPromptExtension:
    return JsonPromptExtension()
