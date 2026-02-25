"""Anti-fraud Email Analysis Inference Service - MVP"""

import json
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Configuration
# All parameters are read from environment variables so the service can be
# tuned at deploy time without touching source code.
# ---------------------------------------------------------------------------

# HuggingFace repo ID or local directory path to the model weights.
# vLLM will auto-download from HuggingFace if a repo ID is provided.
MODEL_PATH = os.getenv("MODEL_PATH", "meta-llama/Llama-3.1-8B")

# Maximum sequence length (prompt + generated tokens). Reduce this if you
# run into OOM errors on smaller GPUs.
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))

# Fraction of GPU VRAM reserved for the model KV-cache. 0.90 leaves ~10%
# for CUDA kernels and other overhead; lower if you see OOM at startup.
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))

# Number of GPUs to shard the model across. Set to >1 only when the model
# is too large to fit on a single GPU.
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

# Optional post-training quantization format: "awq", "gptq", or leave empty
# for none. Must match the quantization used when the model was saved.
QUANTIZATION = os.getenv("QUANTIZATION", "").strip() or None

# Floating-point precision used for inference weights.
# "bfloat16" is recommended for Ampere+ GPUs; use "float16" for older cards.
DTYPE = os.getenv("DTYPE", "bfloat16")


# ---------------------------------------------------------------------------
# System Prompt + Few-shot Examples
#
# The system prompt instructs the model to act as a fraud-detection assistant
# and enforces a strict JSON output schema. Few-shot examples are appended to
# the conversation so the model can learn the expected reasoning style without
# any fine-tuning.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional anti-fraud email analysis assistant. Your task is to analyze the email information provided by the user and determine whether the email is a fraudulent/phishing email.

You MUST return the result strictly in the following JSON format. Do not output anything else:
{
  "is_fraud": true/false,
  "risk_score": integer between 0 and 100,
  "reason": "analysis reasoning in English",
  "suggestion": "recommended action in English"
}

Scoring criteria:
- 0-30:  Low risk — legitimate email
- 31-60: Medium risk — exercise caution
- 61-100: High risk — likely fraudulent

Key analysis points:
1. Is the sender address suspicious? (e.g., typosquatting, free email impersonating an official domain)
2. Does the subject line use urgency language? (e.g., "URGENT", "Immediate action required", "Last chance")
3. Does the body contain phishing links, requests for sensitive information, or unrealistic reward promises?
4. Is the overall writing style professional? Are there grammar or spelling errors?"""

# Two-turn few-shot examples injected between the system prompt and the live
# user request. One example demonstrates a clear phishing email; the other
# demonstrates a benign internal company email. This helps the model calibrate
# its scoring without fine-tuning.
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            'Analyze the following email:\n'
            '{"sender": "service@paypa1.com", "subject": "Your account has been restricted — verify immediately", '
            '"content": "Dear user, we have detected unusual activity on your account. '
            'Please click the link below to verify your identity immediately, or your account will be permanently frozen within 24 hours. '
            'http://paypa1-verify.com/login"}'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "is_fraud": True,
                "risk_score": 92,
                "reason": (
                    "The sender domain paypa1.com typosquats the official PayPal domain "
                    "(digit '1' replacing the letter 'l'). The email employs high-urgency language "
                    "threatening permanent account suspension, and contains a phishing link pointing "
                    "to a non-official domain."
                ),
                "suggestion": (
                    "Do not click any links in this email. Navigate directly to paypal.com via your "
                    "browser to check your account status. Report this email as phishing and delete it."
                ),
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Analyze the following email:\n'
            '{"sender": "hr@company.com", "subject": "2024 Annual Performance Review Notice", '
            '"content": "Hi everyone, the 2024 annual performance review will kick off next month. '
            'Please log in to the internal OA system to complete your self-evaluation form. '
            'Contact HR if you have any questions."}'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "is_fraud": False,
                "risk_score": 8,
                "reason": (
                    "The sender uses an internal corporate domain. The content is a routine performance "
                    "review notification with no suspicious links, no requests for sensitive information, "
                    "and a professional writing style."
                ),
                "suggestion": (
                    "This appears to be a legitimate internal company notice and can be handled normally. "
                    "If still uncertain, verify by calling the HR department directly."
                ),
            },
            ensure_ascii=False,
        ),
    },
]


# ---------------------------------------------------------------------------
# Data Models
#
# Pydantic models serve dual purposes: request validation (FastAPI rejects
# malformed JSON automatically) and OpenAPI schema generation for the docs UI.
# ---------------------------------------------------------------------------

class EmailInput(BaseModel):
    """Incoming email data submitted by the caller."""
    sender: str    # Full sender email address
    subject: str   # Email subject line
    content: str   # Plain-text body of the email


class PredictResponse(BaseModel):
    """Structured fraud analysis result returned to the caller."""
    is_fraud: bool    # True if the model classifies the email as fraudulent
    risk_score: int   # 0-100 risk score (higher = more suspicious)
    reason: str       # Human-readable explanation of the classification
    suggestion: str   # Recommended action for the recipient


# ---------------------------------------------------------------------------
# vLLM Engine Initialization & FastAPI Lifespan
#
# Using the lifespan context manager (introduced in FastAPI 0.93) instead of
# the deprecated on_event("startup") / on_event("shutdown") decorators.
# The model is loaded once at startup and kept in the global `llm` variable
# for the lifetime of the process.
# ---------------------------------------------------------------------------

# Global LLM instance; None until the lifespan startup hook completes.
llm: LLM | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vLLM engine on startup and release it on shutdown."""
    global llm
    print(f"[startup] Loading model: {MODEL_PATH}")
    print(f"[startup] dtype={DTYPE}  quantization={QUANTIZATION}  max_model_len={MAX_MODEL_LEN}")
    print(f"[startup] gpu_memory_utilization={GPU_MEMORY_UTILIZATION}  tensor_parallel_size={TENSOR_PARALLEL_SIZE}")
    try:
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            quantization=QUANTIZATION,
            dtype=DTYPE,
            # Allow models that ship custom modeling code (e.g., some Llama variants).
            trust_remote_code=True,
        )
        print("[startup] Model loaded successfully — service is ready")
    except Exception as e:
        print(f"[error] Failed to load model: {e}")
        raise  # Re-raise so uvicorn reports a non-zero exit code
    yield  # Service is running; requests are handled here
    # --- Shutdown phase ---
    llm = None
    print("[shutdown] Model unloaded")


app = FastAPI(title="Anti-Fraud Email Analysis Service", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Prompt Construction & Output Parsing
# ---------------------------------------------------------------------------

def build_messages(email: EmailInput) -> list[dict]:
    """Build the full chat message list: system prompt + few-shot examples + live user input.

    The email fields are serialized to a compact JSON string so the model
    receives a consistent, machine-readable format rather than free-form text.
    """
    email_json = json.dumps(
        {"sender": email.sender, "subject": email.subject, "content": email.content},
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        # Unpack the two few-shot turns (user + assistant) into the history.
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Analyze the following email:\n{email_json}"},
    ]
    return messages


def parse_llm_output(text: str) -> dict:
    """Extract the JSON object from the raw LLM output string.

    Models sometimes wrap JSON in a markdown code fence (```json ... ```).
    We first try to match that pattern; if it fails we fall back to a greedy
    search for any bare JSON object in the text. Raises ValueError if neither
    pattern matches, which the caller converts to an HTTP 500 response.
    """
    # Attempt 1: JSON wrapped in a markdown code block.
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Attempt 2: Bare JSON object anywhere in the output.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from model output: {text[:200]}")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness/readiness check.

    Returns "ok" only after the model has finished loading. Orchestration
    systems (Kubernetes, Docker Compose healthcheck) can poll this endpoint
    before routing traffic to the container.
    """
    return {
        "status": "ok" if llm is not None else "model_not_loaded",
        "model": MODEL_PATH,
        "quantization": QUANTIZATION,
        "dtype": DTYPE,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(email: EmailInput):
    """Analyze an email and return a structured fraud assessment.

    Flow:
      1. Build the chat prompt (system + few-shot + live email).
      2. Run inference via vLLM with low temperature for deterministic output.
      3. Parse the JSON block from the model's response.
      4. Validate and clamp the risk_score to [0, 100] before returning.

    Raises:
      503 if the model has not finished loading yet.
      500 if the model output cannot be parsed into the expected schema.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet, please try again later")

    messages = build_messages(email)

    # Low temperature (0.1) keeps outputs nearly deterministic — important for
    # a classification task where we need consistent JSON formatting.
    # top_p=0.95 provides a small amount of nucleus sampling as a safety net.
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=512,
        top_p=0.95,
    )

    # vLLM's chat() accepts a list of conversations; we send a single one.
    outputs = llm.chat(
        messages=[messages],
        sampling_params=sampling_params,
    )

    # Extract the generated text from the first (and only) request output.
    raw_text = outputs[0].outputs[0].text.strip()

    try:
        parsed = parse_llm_output(raw_text)
        return PredictResponse(
            is_fraud=bool(parsed["is_fraud"]),
            # Clamp risk_score defensively in case the model returns out-of-range values.
            risk_score=max(0, min(100, int(parsed["risk_score"]))),
            reason=str(parsed["reason"]),
            suggestion=str(parsed["suggestion"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse model output: {exc}\nRaw output: {raw_text[:500]}",
        )
