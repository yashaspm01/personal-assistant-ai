"""
Shared LLM service. Every module (docs, transactions, gmail, movies, news)
calls through this single wrapper — this is what makes the provider swappable
and keeps prompt/retry logic in one place instead of duplicated per module.
"""
import time
import logging
from app.config import settings

logger = logging.getLogger("llm_service")

_anthropic_client = None
_openai_client = None
_groq_client = None


class LLMServiceError(Exception):
    """Raised when the LLM call fails after retries — routers catch this and
    return a clean user-facing error instead of a raw 500 traceback."""
    pass


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_groq():
    global _groq_client
    if _groq_client is None:
        import openai
        _groq_client = openai.OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq_client


def _call_provider(provider: str, system_prompt: str, user_message: str, max_tokens: int) -> str:
    if provider == "claude":
        client = _get_anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text

    elif provider == "openai":
        client = _get_openai()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content

    elif provider == "groq":
        client = _get_groq()
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def chat(
    system_prompt: str,
    user_message: str,
    provider: str | None = None,
    max_tokens: int = 1000,
    max_retries: int = 2,
) -> str:
    """
    Single entry point for all LLM calls in the project.
    Retries transient failures (rate limits, timeouts, 5xx) with exponential
    backoff. Raises LLMServiceError (not the raw provider exception) once
    retries are exhausted, so routers can catch one clean exception type.
    """
    provider = provider or settings.llm_provider
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return _call_provider(provider, system_prompt, user_message, max_tokens)
        except Exception as e:
            last_error = e
            error_name = type(e).__name__
            # Don't retry on errors that won't fix themselves (bad key, bad request)
            if "AuthenticationError" in error_name or "BadRequestError" in error_name:
                logger.error(f"Non-retryable LLM error ({provider}): {e}")
                raise LLMServiceError(
                    f"LLM request failed ({provider}): {e}. This is likely a "
                    f"config issue (API key or request format), not transient — check .env."
                ) from e

            if attempt < max_retries:
                wait = 2 ** attempt  # 1s, 2s, 4s...
                logger.warning(
                    f"LLM call failed (attempt {attempt+1}/{max_retries+1}, {provider}): "
                    f"{error_name}: {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"LLM call failed after {max_retries+1} attempts ({provider}): {e}")

    raise LLMServiceError(
        f"LLM request failed after {max_retries+1} attempts ({provider}): {last_error}"
    )


def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> tuple[str, str]:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        context_blocks.append(f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided context below. If the answer is not contained in the context, "
        "say clearly that you don't have that information — do not guess or "
        "use outside knowledge. When you answer, cite which source number "
        "supports each claim, like [Source 1]."
    )
    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"
    return system_prompt, user_message
