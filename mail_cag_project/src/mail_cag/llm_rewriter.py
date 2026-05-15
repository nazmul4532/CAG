from __future__ import annotations

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
) -> list[str]:
    """Ask Ollama for label-preserving defensive rewrites of one email."""

    prompt = build_prompt(label=label, text=text, candidates=candidates)
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


def build_prompt(label: int, text: str, candidates: int) -> str:
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
{text[:4000]}
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
