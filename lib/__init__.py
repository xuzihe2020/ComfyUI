"""Shared repo-level infrastructure for scripts and tools.

Subpackages / modules:

    lib.envfile      cross-repo .env credential loading and API-key lookup
                     (load_env_file / env_api_key / env_value)
    lib.llm_client   hosted-API clients (Grok, OpenAI, Gemini, BFL), all
                     subclasses of BaseAPIClient with .env-backed from_env()

Every hosted-API call made by this repo's scripts and tools goes through
lib.llm_client — do not duplicate API client code in scripts. Keys come from
the environment or the repo-root `.env` (see `.env.example`); real environment
variables always win over `.env` values.

Scripts put the repo root on sys.path before importing:

    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))
    from lib.llm_client import GrokClient

Everything here is stdlib-only (urllib) on purpose so it runs in the ComfyUI
venv without extra SDK dependencies.
"""
