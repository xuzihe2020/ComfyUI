import asyncio
import mimetypes
import os
import shutil
import folder_paths
import nodes
from app import local_file_picker
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
    source_path = request.rel_url.query.get("source_path", "")
    if image == "":
        return web.json_response({"clean_name": "", "root_dir": ""})

    strip_double = _as_bool(request.rel_url.query.get("strip_double_underscore_suffix"), True)
    strip_version = _as_bool(request.rel_url.query.get("strip_version_suffix"), True)

    image_path = source_path if source_path and os.path.isfile(source_path) else nodes.LoadImage.resolve_image_path(image)
    clean_name, root_dir, _, _, _ = nodes.LoadImage.derive_save_paths(
        image_path,
        strip_double_underscore_suffix=strip_double,
        strip_version_suffix=strip_version,
    )

    return web.json_response({
        "clean_name": clean_name,
        "root_dir": root_dir,
        "source_path": os.path.abspath(image_path),
    })


def _copy_local_image_to_input(source_path):
    source_path = os.path.abspath(source_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)

    content_type = mimetypes.guess_type(source_path)[0] or ""
    if not content_type.startswith("image/"):
        raise ValueError("Selected file is not an image")

    input_dir = os.path.abspath(folder_paths.get_input_directory())
    os.makedirs(input_dir, exist_ok=True)
    filename = os.path.basename(source_path)
    destination = os.path.abspath(os.path.join(input_dir, filename))

    if destination != input_dir and not destination.startswith(input_dir + os.sep):
        raise ValueError("Invalid image filename")

    same_file = os.path.exists(destination) and os.path.samefile(source_path, destination)
    if not same_file:
        shutil.copy2(source_path, destination)

    return filename


@PromptServer.instance.routes.get("/local_file_picker/pick_file")
@PromptServer.instance.routes.get("/load_image/local_file_picker")
async def local_file_picker_pick_file(request):
    initial_dir = request.rel_url.query.get("initial_dir", "")
    if initial_dir == "":
        initial_dir = folder_paths.get_input_directory()

    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, local_file_picker.pick_file, initial_dir)
    return web.json_response({"path": path})


@PromptServer.instance.routes.get("/local_file_picker/pick_load_image")
async def local_file_picker_pick_load_image(request):
    initial_dir = request.rel_url.query.get("initial_dir", "")
    if initial_dir == "":
        initial_dir = folder_paths.get_input_directory()

    loop = asyncio.get_running_loop()
    source_path = await loop.run_in_executor(None, local_file_picker.pick_file, initial_dir)
    if source_path == "":
        return web.json_response({"path": "", "image": ""})

    try:
        image = _copy_local_image_to_input(source_path)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    strip_double = _as_bool(request.rel_url.query.get("strip_double_underscore_suffix"), True)
    strip_version = _as_bool(request.rel_url.query.get("strip_version_suffix"), True)
    clean_name, root_dir, _, _, _ = nodes.LoadImage.derive_save_paths(
        source_path,
        strip_double_underscore_suffix=strip_double,
        strip_version_suffix=strip_version,
    )

    return web.json_response({
        "path": source_path,
        "source_path": os.path.abspath(source_path),
        "image": image,
        "clean_name": clean_name,
        "root_dir": root_dir,
    })


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
