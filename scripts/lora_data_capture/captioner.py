"""Grok captioning core for LoRA training datasets.

Owns the caption contract with Grok: the strict JSON schema, the message
construction, local validation of returned captions, and the validation-retry
loop that feeds Grok its own errors (same pattern as image_description_v2).

Captions keep the literal ``{TRIGGER}`` placeholder end-to-end; substitution
with a real trigger token happens only when rendering the sibling ``.txt``
training caption (``render_caption_txt``).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

from lib.llm_client import GrokClient
from lib.llm_client.base import APIError

FIELD_EN = "caption_en"
FIELD_ZH = "caption_zh"
FIELD_MULTI = "multiple_people"
REQUIRED_FIELDS = (FIELD_EN, FIELD_ZH, FIELD_MULTI)

TRIGGER_PLACEHOLDER = "{TRIGGER}"
# Lazy-output floor only; the real length spec lives in prompts/grok_system_caption.txt.
MIN_CAPTION_WORDS = 25

# Called once per Grok attempt with keyword args: attempt, request (raw body),
# response (raw API response dict, None on transport failure), and optionally
# validation_error / transport_error strings.
LogAttempt = Callable[..., None]


class GrokValidationError(ValueError):
    """Raised when Grok returns JSON that does not match the caption shape."""


class ValidationRetriesExhausted(RuntimeError):
    """Raised when Grok keeps returning invalid captions after all retries."""


def caption_schema() -> dict[str, Any]:
    """The strict json_schema Grok must fill in."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            FIELD_EN: {
                "type": "string",
                "minLength": 1,
                "description": (
                    "English caption: 2-4 plain declarative sentences referring to "
                    f"the woman as the literal token {TRIGGER_PLACEHOLDER}."
                ),
            },
            FIELD_ZH: {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Chinese caption with exactly the same meaning as caption_en, "
                    f"also using the literal token {TRIGGER_PLACEHOLDER}."
                ),
            },
            FIELD_MULTI: {
                "type": "boolean",
                "description": "True only when any person or face other than her is visible.",
            },
        },
        "required": list(REQUIRED_FIELDS),
    }


def initial_messages(system_prompt: str, user_prompt: str, image_data_uri: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri, "detail": "high"},
                },
            ],
        },
    ]


def retry_prompt(error_reason: str) -> str:
    return (
        "Your previous response was invalid.\n"
        f"Validation error: {error_reason}\n\n"
        "Return corrected JSON only. The response must be one JSON object with exactly these fields:\n"
        f"- {FIELD_EN}: non-empty English caption containing the literal token {TRIGGER_PLACEHOLDER}\n"
        f"- {FIELD_ZH}: non-empty Chinese caption with the same meaning, also containing {TRIGGER_PLACEHOLDER}\n"
        f"- {FIELD_MULTI}: boolean\n"
        "Do not add commentary, markdown, or any extra fields."
    )


def build_request_body(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "lora_data_capture",
                "strict": True,
                "schema": caption_schema(),
            },
        },
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _normalize_caption(text: str) -> str:
    """Collapse whitespace/newlines: training captions are one line."""
    return " ".join(text.split())


def parse_and_validate_response(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GrokValidationError(f"response is not valid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise GrokValidationError("response must be a JSON object")

    expected = set(REQUIRED_FIELDS)
    unexpected = sorted(k for k in data if k not in expected)
    if unexpected:
        raise GrokValidationError(f"unexpected field present: {', '.join(unexpected)}")

    for field in REQUIRED_FIELDS:
        if field not in data:
            raise GrokValidationError(f"{field} field is missing, it must present")

    for field in (FIELD_EN, FIELD_ZH):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise GrokValidationError(f"{field} field must be a non-empty string")
        if TRIGGER_PLACEHOLDER not in value:
            raise GrokValidationError(
                f"{field} field must contain the literal token {TRIGGER_PLACEHOLDER}"
            )

    if not isinstance(data[FIELD_MULTI], bool):
        raise GrokValidationError(f"{FIELD_MULTI} field must be a boolean")

    caption_en = _normalize_caption(data[FIELD_EN])
    caption_zh = _normalize_caption(data[FIELD_ZH])

    if _has_cjk(caption_en):
        raise GrokValidationError(f"{FIELD_EN} field must be English only, no Chinese characters")
    if not _has_cjk(caption_zh):
        raise GrokValidationError(f"{FIELD_ZH} field must be written in Chinese")
    if len(caption_en.split()) < MIN_CAPTION_WORDS:
        raise GrokValidationError(
            f"{FIELD_EN} field is too short; follow the caption length rule in the system "
            "instructions and cover the full checklist"
        )

    return {FIELD_EN: caption_en, FIELD_ZH: caption_zh, FIELD_MULTI: data[FIELD_MULTI]}


# --------------------------------------------------------------------------- #
# Grok call with validation retries
# --------------------------------------------------------------------------- #

def _message_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise APIError(f"unexpected chat response shape: {exc!r}") from exc
    if not isinstance(content, str):
        raise APIError("unexpected chat response shape: message content is not a string")
    return content


def call_grok_with_validation(
    client: GrokClient,
    *,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    image_data_uri: str,
    validation_retries: int,
    request_retries: int,
    log_attempt: LogAttempt | None = None,
    on_request_retry: Callable[[int, float, Exception], None] | None = None,
) -> dict[str, Any]:
    messages = initial_messages(system_prompt, user_prompt, image_data_uri)
    last_error = "unknown validation error"
    last_content = ""

    for attempt in range(0, validation_retries + 1):
        body = build_request_body(model=model, messages=messages, temperature=temperature)

        try:
            response = client.chat_completions(body, retries=request_retries, on_retry=on_request_retry)
        except Exception as exc:
            if log_attempt is not None:
                log_attempt(attempt=attempt + 1, request=body, response=None, transport_error=str(exc))
            raise
        try:
            content = _message_content(response)
        except APIError as exc:
            if log_attempt is not None:
                log_attempt(attempt=attempt + 1, request=body, response=response, transport_error=str(exc))
            raise
        last_content = content

        try:
            data = parse_and_validate_response(content)
        except GrokValidationError as exc:
            last_error = str(exc)
            if log_attempt is not None:
                log_attempt(attempt=attempt + 1, request=body, response=response, validation_error=last_error)
            if attempt >= validation_retries:
                break
            retry_no = attempt + 1
            print(f"  ! invalid Grok caption: {last_error}", file=sys.stderr)
            print(
                f"  ... validation retry {retry_no}/{validation_retries}: sending error reason back to Grok",
                file=sys.stderr,
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": retry_prompt(last_error)})
            continue

        if log_attempt is not None:
            log_attempt(attempt=attempt + 1, request=body, response=response)
        return data

    raise ValidationRetriesExhausted(
        f"Grok returned invalid captions after {validation_retries} validation retries: {last_error}\n"
        f"Last response:\n{last_content}"
    )


# --------------------------------------------------------------------------- #
# Training caption rendering
# --------------------------------------------------------------------------- #

def render_caption_txt(caption_en: str, trigger: str) -> str:
    """The sibling .txt content: EN caption with the trigger substituted."""
    caption = caption_en
    if trigger:
        caption = caption.replace(TRIGGER_PLACEHOLDER, trigger)
    return caption.strip() + "\n"
