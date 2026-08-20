from __future__ import annotations

import logging

from backend.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a fraud investigation assistant. The case's risk band, calibrated "
    "probability, and triggered rules are authoritative and already decided by "
    "the system — treat them as ground truth and build the report around them. "
    "The listed key factors are supporting signals showing which transaction "
    "features stood out; do not claim a direction (increased/decreased risk) "
    "for them beyond what is stated. Using ONLY the findings provided below, "
    "write a 3-4 sentence analyst report explaining why this case looks the "
    "way it does. Do not invent any numbers or facts, and do not make a "
    "decision — you explain, you never decide."
)

GENERATION_CONFIG = {
    "temperature": 0.2,
    "max_output_tokens": 200,
}

REQUEST_TIMEOUT_SECONDS = 10


def _build_user_content(findings: list[str], txn_summary: str) -> str:
    findings_block = "\n".join(f"- {f}" for f in findings)
    return f"Transaction summary: {txn_summary}\n\nFindings:\n{findings_block}"


def _fallback_text(findings: list[str]) -> str:
    return " ".join(findings)


def _generate_groq(findings: list[str], txn_summary: str, system_prompt: str) -> str | None:
    """Returns the generated report text, or None on any failure (missing
    key/model, missing package, rate limit, timeout, API error) — callers
    fall back to the deterministic findings text. Never raises, never logs
    the key."""
    api_key = settings.llm_api_key
    model_name = settings.llm_model
    if not api_key or not model_name:
        return None

    try:
        from groq import Groq
    except ImportError:
        logger.warning("groq is not installed; using fallback report")
        return None

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_user_content(findings, txn_summary)},
            ],
            temperature=GENERATION_CONFIG["temperature"],
            max_tokens=GENERATION_CONFIG["max_output_tokens"],
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text or None
    except Exception as exc:
        # Deliberately log only the exception type — API error messages can
        # echo request details, and the key must never reach the logs.
        logger.warning("Groq report generation failed (%s); using fallback report", type(exc).__name__)
        return None


# Provider registry — add "openai"/"anthropic"/"gemini"/"ollama" here later
# without touching generate_report()'s call sites. LLM_PROVIDER in .env
# selects the branch; only "groq" is implemented today.
_PROVIDERS = {
    "groq": _generate_groq,
}


def generate_report(findings: list[str], txn_summary: str, system_prompt: str = SYSTEM_PROMPT) -> dict:
    """Provider-abstract report generation. Always returns a usable report:
    {"text": str, "source": "<provider>" | "fallback"}. Never raises.

    system_prompt defaults to the SHAP/rules report prompt above —
    precedent.py's precedent explanation passes its own prompt instead,
    reusing this exact call layer (same provider registry, async safety,
    fallback/timeout/key handling) for a different content type."""
    if not findings:
        return {"text": "No findings available for this case.", "source": "fallback"}

    generate_fn = _PROVIDERS.get(settings.llm_provider.lower())
    if generate_fn is not None:
        text = generate_fn(findings, txn_summary, system_prompt)
        if text:
            return {"text": text, "source": settings.llm_provider.lower()}

    return {"text": _fallback_text(findings), "source": "fallback"}
