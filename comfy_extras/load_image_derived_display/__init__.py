import folder_paths
import nodes
from aiohttp import web
from server import PromptServer


WEB_DIRECTORY = "./web"


def _as_bool(value, default=True):
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "yes", "on")


@PromptServer.instance.routes.get("/load_image/derived_display")
async def load_image_derived_display(request):
    image = request.rel_url.query.get("image", "")
    if image == "":
        return web.json_response({"clean_name": "", "root_dir": ""})

    strip_double = _as_bool(request.rel_url.query.get("strip_double_underscore_suffix"), True)
    strip_version = _as_bool(request.rel_url.query.get("strip_version_suffix"), True)

    image_path = folder_paths.get_annotated_filepath(image)
    clean_name, root_dir, _, _, _ = nodes.LoadImage.derive_save_paths(
        image_path,
        strip_double_underscore_suffix=strip_double,
        strip_version_suffix=strip_version,
    )

    return web.json_response({
        "clean_name": clean_name,
        "root_dir": root_dir,
    })


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
