"""Compatibility paths for shared infrastructure used by older scripts.

Subpackages / modules:

    lib.envfile      cross-repo .env credential loading and API-key lookup
                     (load_env_file / env_api_key / env_value)
    lib.llm_client   hosted-API clients (Grok, OpenAI, Gemini, BFL), all
                     subclasses of BaseAPIClient with .env-backed from_env()

The implementations now live in the sibling ``aigc-shared`` package. These
paths remain only so external callers using the old imports do not break.

Scripts put the repo root on sys.path before importing:

    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))
    from lib.llm_client import GrokClient

Install the sibling package into the ComfyUI venv before using these paths.
"""
