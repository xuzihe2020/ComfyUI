"""Hosted-API clients for repo scripts and tools — one polymorphic family.

All clients subclass `BaseAPIClient` (auth, .env-backed key resolution via
`from_env()`, urllib transport with retry/backoff, multipart encoding) and add
only their service-specific endpoints:

    GrokClient    Grok (xAI) chat completions       XAI_API_KEY
    OpenAIClient  OpenAI images/files/batches       OPENAI_API_KEY
    GeminiClient  Gemini generateContent            GEMINI_API_KEY / GOOGLE_API_KEY
    BFLClient     Black Forest Labs (FLUX) tasks    FLUX_API_KEY / BFL_API_KEY

Service defaults (base URL, model, timeout) live on the classes, e.g.
`GrokClient.DEFAULT_MODEL`. Do not write inline API clients in scripts —
extend this package instead.
"""

from lib.llm_client.base import (
    APIError,
    BaseAPIClient,
    OnRetry,
    TRANSIENT_HTTP_CODES,
    encode_multipart,
)
from lib.llm_client.bfl import BFLClient, BFLError
from lib.llm_client.gemini import GeminiClient
from lib.llm_client.grok import GrokClient
from lib.llm_client.openai import OpenAIClient, OpenAIError

__all__ = [
    "APIError",
    "BaseAPIClient",
    "BFLClient",
    "BFLError",
    "GeminiClient",
    "GrokClient",
    "OnRetry",
    "OpenAIClient",
    "OpenAIError",
    "TRANSIENT_HTTP_CODES",
    "encode_multipart",
]
