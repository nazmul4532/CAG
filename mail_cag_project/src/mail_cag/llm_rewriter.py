from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def choose_ollama_model(config: dict[str, Any]) -> str:
    """Return the first configured Ollama model that is installed locally."""

    llm = config["llm"]
    wanted = [llm["model"], *llm.get("fallback_models", [])]
    installed = set(list_ollama_models(llm.get("base_url", "http://127.0.0.1:11434")))
    for model in wanted:
        if model in installed:
            return model
    raise RuntimeError(
        "None of the configured Ollama models are installed: " + ", ".join(wanted)
    )


def list_ollama_models(base_url: str) -> list[str]:
    """List model names known to the local Ollama server."""

    data = _ollama_request(base_url, "/api/tags")
    return [item["name"] for item in data.get("models", [])]


def rewrite_email(
    *,
    base_url: str,
    model: str,
    label: int,
    text: str,
    candidates: int,
    temperature: float,
    top_p: float,
    prompt_style: str = "legacy",
) -> list[str]:
    """Ask Ollama for label-preserving defensive rewrites of one email."""

    prompt = build_prompt(
        label=label,
        text=text,
        candidates=candidates,
        prompt_style=prompt_style,
    )
    data = _ollama_request(
        base_url,
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "top_p": top_p},
        },
    )
    return parse_rewrites(data.get("response", ""), candidates)


def rewrite_cache_key(
    *,
    model: str,
    label: int,
    text: str,
    candidates: int,
    temperature: float,
    top_p: float,
    parent_id: str,
    prompt_style: str = "legacy",
) -> str:
    """Return a stable key for one LLM rewrite request."""

    payload = {
        "model": model,
        "parent_id": parent_id,
        "prompt": build_prompt(
            label=label,
            text=text,
            candidates=candidates,
            prompt_style=prompt_style,
        ),
        "candidates": int(candidates),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_prompt(
    label: int,
    text: str,
    candidates: int,
    prompt_style: str = "legacy",
) -> str:
    if prompt_style in {"label_aware", "label_aware_v2"}:
        return build_label_aware_prompt(label=label, text=text, candidates=candidates)
    if prompt_style != "legacy":
        raise ValueError(f"Unknown LLM rewrite prompt style: {prompt_style}")

    label_name = "phishing" if int(label) == 1 else "benign"
    return f"""You are helping a defensive email-security research experiment.

Rewrite the email below into {candidates} label-preserving version.

Rules:
- Preserve the original label: {label_name}.
- Preserve the core meaning and intent.
- Return exactly one rewritten email.
- Do not return JSON, dictionaries, lists, markdown, or explanations.
- Preserve the same spam/phishing style, URL behavior, language, and suspicious intent.
- Do not make it cleaner or more professional than the original.
- Do not add new URLs, phone numbers, attachments, brands, names, credentials, or payment instructions.
- Do not make the email more harmful or more actionable than the original.

Email:
\"\"\"
{text}
\"\"\"
"""


def build_label_aware_prompt(label: int, text: str, candidates: int) -> str:
    label_name = "phishing" if int(label) == 1 else "benign"
    if int(label) == 1:
        label_rules = """- Preserve the phishing label and the same type of social-engineering goal.
- Preserve the original attack surface: if the source asks for login, credentials, payment, account verification, clicking a link, opening an attachment, or replying, keep that same behavior.
- Preserve URL/attachment presence when present, but do not add new URLs, attachments, credentials, brands, phone numbers, or payment details.
- Keep roughly the same risk level, urgency level, and suspiciousness as the source.
- Vary wording, sentence order, formatting, and surface style so it is a realistic alternate phishing email.
- Do not make the message cleaner, safer, more professional, more detailed, more actionable, or more harmful than the source."""
    else:
        label_rules = """- Preserve the benign label, legitimate purpose, ordinary sender intent, and non-deceptive tone.
- Keep the email natural and non-spammy.
- Vary wording, sentence order, formatting, and surface style so it is a realistic alternate benign email.
- Do not introduce suspicious intent, credential requests, threats, payment pressure, impersonation, phishing-like urgency, or new links.
- Do not make it look like spam, a scam, a security warning, or an account-verification message."""

    return f"""You are helping a defensive email-security research experiment.

Rewrite the email below into {candidates} label-preserving version.

Shared rules:
- Preserve the original label: {label_name}.
- Preserve the core meaning, language, sender/recipient relationship, and normal URL/attachment behavior.
- Return exactly one rewritten email.
- Do not return JSON, dictionaries, lists, markdown, or explanations.
- Do not add new URLs, phone numbers, attachments, brands, names, credentials, or payment instructions.
- Keep the rewritten email similar in length to the source.

Label-specific rules:
{label_rules}

Email:
\"\"\"
{text}
\"\"\"
"""


def parse_rewrites(response: str, candidates: int) -> list[str]:
    """Parse the model response into one or more rewrite strings."""

    cleaned = response.strip()
    if candidates == 1:
        return [strip_code_fence(cleaned)] if cleaned else []

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()][:candidates]
    except json.JSONDecodeError:
        pass

    lines = [line.strip("- 1234567890.").strip() for line in cleaned.splitlines()]
    return [line for line in lines if line][:candidates]


def strip_code_fence(text: str) -> str:
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return text


def _ollama_request(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"Ollama request failed at {url}: {exc}") from exc
