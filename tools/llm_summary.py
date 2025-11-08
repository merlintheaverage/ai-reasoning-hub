import os

import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

PROVIDER = os.getenv("SUMMARY_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Import OpenAI error types at module scope (safe even if provider != openai)
try:
    from openai import OpenAI, APIError, RateLimitError, InternalServerError
except Exception:
    # not required unless PROVIDER == openai, so ignore import errors
    OpenAI = None
    APIError = RateLimitError = InternalServerError = Exception

@retry(
    wait=wait_exponential(multiplier=2, min=5, max=90),
    stop=stop_after_attempt(8),
    retry=retry_if_exception_type((RateLimitError, APIError, InternalServerError)),
)
def call_llm(prompt: str):
    if PROVIDER == "openai":
        if OpenAI is None:
            raise RuntimeError("openai SDK not available - install openai package")
        client = OpenAI()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise research summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        text = resp.choices[0].message.content
        total_tokens = resp.usage.total_tokens if resp.usage else None
        return {"text": text, "tokens": total_tokens, "model": OPENAI_MODEL}

    elif PROVIDER == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,  # Increased here too
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(part.text for part in msg.content)
        return {"text": text, "tokens": None, "model": ANTHROPIC_MODEL}

    elif PROVIDER == "ollama":
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "")
        return {"text": text, "tokens": None, "model": OLLAMA_MODEL}

    else:
        raise RuntimeError(f"Unknown SUMMARY_PROVIDER: {PROVIDER}")


def triage_paper(title: str, abstract: str) -> dict:
    """
    Stage A: Quick triage using Gemini Flash to determine if paper is worth full summary.
    Returns: {"relevant": bool, "reason": str, "model": str, "tokens": int}
    """

    prompt = f"""
You are filtering AI research papers for a reasoning-focused research hub.

Determine if this paper is relevant to AI reasoning, agents, planning, or problem-solving.

Title: {title}

Abstract: {abstract}

Output ONLY in this format:
RELEVANT: YES or NO
REASON: <one sentence explaining why>

Be strict - only mark YES if the paper directly involves:
- Reasoning capabilities in AI systems
- Agent planning or decision-making
- Problem-solving approaches
- Chain-of-thought or multi-step reasoning
- Benchmark evaluation of reasoning

Mark NO if it's primarily about:
- Pure computer vision without reasoning
- Low-level optimization
- Hardware/systems
- Domain-specific applications without reasoning focus

Be slightly permissive—err on the side of YES if uncertainty is high (we can down-score later).
""".strip()

    try:
        import google.generativeai as genai

        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 200,
            }
        )

        text = response.text.strip()

        # Parse response
        relevant = "RELEVANT: YES" in text.upper()
        reason_line = [line for line in text.split("\n") if "REASON:" in line.upper()]
        reason = reason_line[0].split(":", 1)[1].strip() if reason_line else "No reason provided"

        return {
            "relevant": relevant,
            "reason": reason,
            "model": GEMINI_MODEL,
            "tokens": 0  # Gemini free tier, effectively zero cost
        }

    except ImportError:
        print("⚠️  google-generativeai not installed, falling back to OpenAI for triage")
        return triage_with_openai(title, abstract)
    except Exception as e:
        print(f"⚠️  Gemini triage failed: {e}, falling back to OpenAI")
        return triage_with_openai(title, abstract)


def triage_with_openai(title: str, abstract: str) -> dict:
    """Fallback triage using OpenAI if Gemini fails"""
    from openai import OpenAI

    client = OpenAI()

    prompt = f"""
Determine if this paper is relevant to AI reasoning.

Title: {title}
Abstract: {abstract}

Output: RELEVANT: YES or NO
REASON: <one sentence>

Be slightly permissive—err on the side of YES if uncertainty is high.
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # Cheap for triage
        messages=[
            {"role": "system", "content": "You are a research paper filter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=100,
    )

    text = resp.choices[0].message.content
    relevant = "RELEVANT: YES" in text.upper()
    reason_line = [line for line in text.split("\n") if "REASON:" in line.upper()]
    reason = reason_line[0].split(":", 1)[1].strip() if reason_line else text

    return {
        "relevant": relevant,
        "reason": reason,
        "model": "gpt-4o-mini (fallback)",
        "tokens": resp.usage.total_tokens if resp.usage else 0  # Track fallback cost
    }
