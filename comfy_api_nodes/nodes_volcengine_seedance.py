"""Direct Volcengine Ark Seedance nodes using the account's own Beijing API key."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from urllib.parse import quote

import aiohttp
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.util import download_url_to_video_output


logger = logging.getLogger(__name__)

DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
SEEDANCE_MODELS = [
    "doubao-seedance-2-0-260128",
    "doubao-seedance-2-0-fast-260128",
]
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def _urls(value: str, *, maximum: int, label: str) -> list[str]:
    result = [line.strip() for line in (value or "").splitlines() if line.strip()]
    if len(result) > maximum:
        raise ValueError(f"{label} accepts at most {maximum} URLs, received {len(result)}.")
    for url in result:
        if not (url.startswith("https://") or url.startswith("asset://")):
            raise ValueError(f"{label} must contain HTTPS or asset:// URLs, one per line: {url}")
    return result


async def _response_json(response: aiohttp.ClientResponse) -> dict:
    try:
        payload = await response.json(content_type=None)
    except Exception:
        body = (await response.text())[:2000]
        raise RuntimeError(f"Ark returned HTTP {response.status}: {body}") from None
    if response.status >= 400:
        error = payload.get("error") or payload
        code = error.get("code", "unknown") if isinstance(error, dict) else "unknown"
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise RuntimeError(f"Ark returned HTTP {response.status} ({code}): {message}")
    return payload


async def _create_and_wait(payload: dict, poll_interval: int, max_wait_seconds: int) -> tuple[str, str]:
    api_key = os.getenv("ARK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ARK_API_KEY is empty. Add it to /home/ubuntu/ComfyUI/.env and restart comfyui.service.")

    base_url = os.getenv("ARK_BASE_URL", DEFAULT_ARK_BASE_URL).rstrip("/")
    url = f"{base_url}/contents/generations/tasks"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=90)

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.post(url, json=payload) as response:
            created = await _response_json(response)
        task_id = created.get("id")
        if not task_id:
            raise RuntimeError(f"Ark did not return a task id: {created}")

        deadline = time.monotonic() + max_wait_seconds
        last_status = None
        while time.monotonic() < deadline:
            async with session.get(f"{url}/{task_id}") as response:
                task = await _response_json(response)
            status = task.get("status")
            if status != last_status:
                logger.info("Seedance task %s status: %s", task_id, status)
                last_status = status
            if status in TERMINAL_STATUSES:
                if status != "succeeded":
                    error = task.get("error") or {}
                    raise RuntimeError(
                        f"Seedance task {task_id} ended as {status}: "
                        f"{error.get('code', 'unknown')} - {error.get('message', 'No message')}"
                    )
                video_url = (task.get("content") or {}).get("video_url")
                if not video_url:
                    raise RuntimeError(f"Seedance task {task_id} succeeded without a video URL.")
                return task_id, video_url
            await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Seedance task {task_id} did not finish within {max_wait_seconds} seconds.")


def _upload_video_to_r2(video, task_id: str) -> str:
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise RuntimeError("R2 upload requires boto3 in the ComfyUI virtual environment.") from None

    account_id = os.getenv("R2_ACCOUNT_ID", "").strip()
    endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    bucket = os.getenv("R2_BUCKET", "").strip()
    missing = [
        name
        for name, value in {
            "R2_ACCESS_KEY_ID": access_key,
            "R2_SECRET_ACCESS_KEY": secret_key,
            "R2_BUCKET": bucket,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"R2 upload is enabled but these variables are empty: {', '.join(missing)}")
    if not endpoint:
        if not account_id:
            raise RuntimeError("Set R2_ENDPOINT_URL or R2_ACCOUNT_ID before enabling R2 upload.")
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

    prefix = os.getenv("R2_PREFIX", "seedance").strip().strip("/") or "seedance"
    key = f"{prefix}/{time.strftime('%Y/%m/%d')}/{task_id}.mp4"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    source = video.get_stream_source()
    client.upload_fileobj(source, bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    if hasattr(source, "seek"):
        source.seek(0)

    public_base = os.getenv("R2_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if public_base:
        return f"{public_base}/{quote(key, safe='/')}"
    return f"r2://{bucket}/{key}"


class VolcengineSeedance2Node(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VolcengineSeedance2Node",
            display_name="Volcengine Seedance 2.0 (Beijing BYOK)",
            category="partner/video/Volcengine BYOK",
            description=(
                "Generate a Seedance 2.0 video through mainland China Volcengine Ark using ARK_API_KEY. "
                "Reference media should be public/signed Cloudflare R2 URLs, one URL per line."
            ),
            inputs=[
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Combo.Input("model", options=SEEDANCE_MODELS, default=SEEDANCE_MODELS[0]),
                IO.Combo.Input("resolution", options=["480p", "720p", "1080p"], default="720p"),
                IO.Combo.Input(
                    "ratio",
                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                    default="adaptive",
                ),
                IO.Int.Input("duration", default=5, min=4, max=15, step=1),
                IO.String.Input(
                    "first_frame_url",
                    default="",
                    optional=True,
                    tooltip="HTTPS or asset:// URL for the first frame.",
                ),
                IO.String.Input(
                    "last_frame_url",
                    default="",
                    optional=True,
                    tooltip="HTTPS or asset:// URL for the last frame.",
                ),
                IO.String.Input(
                    "reference_image_urls",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Up to 9 HTTPS/asset URLs, one per line. Refer to them as 图片1, 图片2, etc.",
                ),
                IO.String.Input(
                    "reference_video_urls",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Up to 3 HTTPS/asset URLs, one per line. Refer to them as 视频1, 视频2, etc.",
                ),
                IO.String.Input(
                    "reference_audio_urls",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Up to 3 HTTPS/asset URLs, one per line. Refer to them as 音频1, 音频2, etc.",
                ),
                IO.Boolean.Input("generate_audio", default=True),
                IO.Boolean.Input("watermark", default=False, advanced=True),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "upload_to_r2",
                    default=False,
                    tooltip="Upload the completed MP4 from memory using the R2 variables in .env.",
                    advanced=True,
                ),
                IO.Int.Input("poll_interval", default=10, min=5, max=60, advanced=True),
                IO.Int.Input("max_wait_seconds", default=1800, min=60, max=7200, advanced=True),
            ],
            outputs=[
                IO.Video.Output(display_name="video"),
                IO.String.Output(display_name="source_url"),
                IO.String.Output(display_name="task_id"),
                IO.String.Output(display_name="r2_url"),
            ],
            is_api_node=False,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        resolution: str,
        ratio: str,
        duration: int,
        first_frame_url: str = "",
        last_frame_url: str = "",
        reference_image_urls: str = "",
        reference_video_urls: str = "",
        reference_audio_urls: str = "",
        generate_audio: bool = True,
        watermark: bool = False,
        seed: int = 0,
        upload_to_r2: bool = False,
        poll_interval: int = 10,
        max_wait_seconds: int = 1800,
    ) -> IO.NodeOutput:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        if model.endswith("fast-260128") and resolution == "1080p":
            raise ValueError("Seedance 2.0 Fast supports 480p and 720p, not 1080p.")

        model_override = os.getenv("ARK_MODEL_ID", "").strip()
        content: list[dict] = [{"type": "text", "text": prompt}]

        if first_frame_url.strip():
            content.append(
                {"type": "image_url", "image_url": {"url": first_frame_url.strip()}, "role": "first_frame"}
            )
        if last_frame_url.strip():
            content.append(
                {"type": "image_url", "image_url": {"url": last_frame_url.strip()}, "role": "last_frame"}
            )
        for url in _urls(reference_image_urls, maximum=9, label="reference_image_urls"):
            content.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})
        for url in _urls(reference_video_urls, maximum=3, label="reference_video_urls"):
            content.append({"type": "video_url", "video_url": {"url": url}, "role": "reference_video"})
        for url in _urls(reference_audio_urls, maximum=3, label="reference_audio_urls"):
            content.append({"type": "audio_url", "audio_url": {"url": url}, "role": "reference_audio"})

        payload = {
            "model": model_override or model,
            "content": content,
            "generate_audio": generate_audio,
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "seed": seed,
            "watermark": watermark,
        }
        task_id, source_url = await _create_and_wait(payload, poll_interval, max_wait_seconds)
        video = await download_url_to_video_output(source_url, timeout=300, cls=cls)
        r2_url = ""
        if upload_to_r2:
            r2_url = await asyncio.to_thread(_upload_video_to_r2, video, task_id)
        return IO.NodeOutput(video, source_url, task_id, r2_url)


class VolcengineSeedanceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [VolcengineSeedance2Node]


async def comfy_entrypoint() -> VolcengineSeedanceExtension:
    return VolcengineSeedanceExtension()
