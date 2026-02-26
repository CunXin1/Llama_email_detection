"""Anti-fraud Email Analysis Inference Service - MVP"""

import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

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
MODEL_PATH = os.getenv("MODEL_PATH", "meta-llama/Llama-3.2-3B-Instruct")

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
  "confidence_level": float between 0.0 and 1.0,
  "detected_threats": ["threat_label_1", "threat_label_2"],
  "reason": "analysis reasoning in English",
  "suggestion": "recommended action in English"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — THREAT DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyze the email across these 10 dimensions and collect all triggered threat labels:

1. [DOMAIN_MISMATCH] — 30 pts
   The sender's domain does NOT match the claimed organization.
   e.g., "amazon-business.net" claiming to be Amazon, or "ucmerced-edu.org" vs "ucmerced.edu".

2. [URL_DISCREPANCY] — 30 pts
   A link's display text differs from its actual destination URL, or the URL points to a non-official domain.
   e.g., anchor "Pay Now" → "amazon-pay-portal.com" instead of "amazon.com".

3. [CREDENTIAL_REQUEST] — 35 pts
   The email directly requests passwords, SSN, verification codes, bank details, or other sensitive credentials.

4. [TOO_GOOD_TO_BE_TRUE] — 30 pts
   The email promises unrealistic prizes, grants, windfalls, unexpected large sums of money, or fabricates large overdue amounts to create pressure.

5. [URGENCY_FEAR] — 15 pts
   The email uses threatening or time-pressured language.
   e.g., "Your account will be locked in 24 hours if you do not act."

6. [REPLY_TO_MISMATCH] — 15 pts
   The Reply-To address differs from the From address, or points to a public mailbox (gmail, yahoo, hotmail, etc.).

7. [GENERIC_SALUTATION] — 8 pts
   The greeting uses a vague placeholder instead of the recipient's real name.
   e.g., "Dear Student", "Dear Customer", "Dear User".

8. [ANOMALOUS_TIMING] — 8 pts
   The send time (date field) is unusual for the claimed sender's role.
   e.g., an official university notice sent at 3 AM local time.

9. [MISSING_SIGNATURE] — 8 pts
   The email lacks a professional signature block, or the signature contains an unreachable/fake phone number.

10. [GRAMMAR_ANOMALY] — 5 pts
    Abnormal punctuation, awkward line breaks, or machine-translation artifacts are present.
    (Note: AI-generated fraud emails may be grammatically correct — weight this lower.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — MANDATORY SCORE CALCULATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST calculate risk_score by summing the points of ALL triggered threats, then cap at 100:

  risk_score = min(100, sum of points for each label in detected_threats)

Point reference:
  DOMAIN_MISMATCH    = 30 pts
  URL_DISCREPANCY    = 30 pts
  CREDENTIAL_REQUEST = 35 pts
  TOO_GOOD_TO_BE_TRUE= 30 pts
  URGENCY_FEAR       = 15 pts
  REPLY_TO_MISMATCH  = 15 pts
  GENERIC_SALUTATION =  8 pts
  ANOMALOUS_TIMING   =  8 pts
  MISSING_SIGNATURE  =  8 pts
  GRAMMAR_ANOMALY    =  5 pts

Example: detected_threats = ["DOMAIN_MISMATCH", "URL_DISCREPANCY", "TOO_GOOD_TO_BE_TRUE"]
  → risk_score = min(100, 30 + 30 + 30) = 90  ← MUST be 90, not any other value.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — MANDATORY FRAUD CLASSIFICATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After computing risk_score, apply these HARD RULES — they override your general judgment:

  RULE A: If risk_score >= 60  → is_fraud MUST be true
  RULE B: If risk_score >= 40  → is_fraud SHOULD be true (override only if very strong counter-evidence)
  RULE C: If risk_score <= 20  → is_fraud MUST be false
  RULE D: If any HIGH-weight threat (DOMAIN_MISMATCH, URL_DISCREPANCY, CREDENTIAL_REQUEST,
           TOO_GOOD_TO_BE_TRUE) is triggered → is_fraud MUST be true, regardless of risk_score.

CRITICAL CONSISTENCY RULE: Your detected_threats, risk_score, and is_fraud MUST be logically
consistent. It is a FATAL ERROR to list high-weight threats in detected_threats while setting
is_fraud=false or risk_score below 60."""

# Two-turn few-shot examples injected between the system prompt and the live
# user request. One example demonstrates a clear phishing email; the other
# demonstrates a benign internal company email. This helps the model calibrate
# its scoring without fine-tuning.
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            'Analyze the following email:\n'
            '{"date": "2026-02-24T10:05:00Z", "sender": "registrar@ucmerced.edu", '
            '"recipient": "rsun@ucmerced.edu", "copy": "academic-advising@ucmerced.edu", '
            '"subject": "Important: Graduation Application Deadline", '
            '"content": "Hi Ruibo, this is a reminder that the deadline to apply for Spring 2026 graduation is approaching. '
            'Please ensure all your major requirements for Computer Science and Mathematics are met. '
            'Visit the official MyUCMerced portal to submit your application."}'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "is_fraud": False,
                "risk_score": 0,
                "confidence_level": 0.93,
                "detected_threats": [],
                "reason": (
                    "Score calculation: sum(triggered threats) = 0. No threat labels are triggered. The sender and CC are on the official ucmerced.edu "
                    "domain, the message content is a routine academic deadline reminder, and there are "
                    "no suspicious external links or requests for credentials or payments. The email is "
                    "personalized to the recipient and is consistent with a registrar/advising notice."
                ),
                "suggestion": (
                    "This appears legitimate. Follow the normal process by accessing the official "
                    "MyUCMerced portal through your usual bookmarks or by typing the known official "
                    "URL directly in your browser."
                ),
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Analyze the following email:\n'
            '{"date": "2026-02-25T09:15:00Z", "sender": "catcard-services@ucmerced-edu.org", '
            '"recipient": "rsun@ucmerced.edu", "copy": "security-audit@ucmerced-edu.org", '
            '"subject": "ACTION REQUIRED: CatCard Synchronization Failed", '
            '"content": "Dear Student, our system failed to sync your CatCard with the new Duo Mobile update. '
            'To maintain access to your meal plan and residence hall, you must re-authenticate your account within 12 hours. '
            'Visit http://ucmerced-verify.support/login to prevent lockout."}'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "is_fraud": True,
                "risk_score": 98,
                "confidence_level": 0.97,
                "detected_threats": [
                    "DOMAIN_MISMATCH",
                    "URL_DISCREPANCY",
                    "CREDENTIAL_REQUEST",
                    "URGENCY_FEAR",
                    "GENERIC_SALUTATION",
                ],
                "reason": (
                    "Score calculation: DOMAIN_MISMATCH(30) + URL_DISCREPANCY(30) + CREDENTIAL_REQUEST(35) "
                    "+ URGENCY_FEAR(15) + GENERIC_SALUTATION(8) = 118, capped at 100. "
                    "RULE D triggered: DOMAIN_MISMATCH, URL_DISCREPANCY, and CREDENTIAL_REQUEST are HIGH-weight threats, "
                    "so is_fraud=true is mandatory. "
                    "The sender and CC use the lookalike domain ucmerced-edu.org, which is not the official "
                    "university domain (ucmerced.edu). The email creates urgency (12-hour deadline) and asks the "
                    "recipient to re-authenticate, indicating credential harvesting. The link points to "
                    "ucmerced-verify.support, another non-official domain."
                ),
                "suggestion": (
                    "Do not click the link or enter credentials. Verify your CatCard/Duo status by navigating "
                    "to official UC Merced resources (ucmerced.edu) or contacting campus IT/Card Services via "
                    "known official phone numbers or portals."
                ),
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Analyze the following email:\n'
            '{"date": "2026-02-26T03:12:00Z", "sender": "grants@federal-student-aid.info", '
            '"recipient": "rsun@ucmerced.edu", "copy": "", '
            '"subject": "Approved: $5,000 Emergency Student Grant", '
            '"content": "Congratulations! You have been selected for a one-time Spring 2026 Emergency Student Grant of $5,000. '
            'This is part of a new government relief program. Claim your funds by logging in with your student credentials here: '
            'http://edu-grants.xyz/claim"}'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "is_fraud": True,
                "risk_score": 92,
                "confidence_level": 0.98,
                "detected_threats": [
                    "DOMAIN_MISMATCH",
                    "URL_DISCREPANCY",
                    "CREDENTIAL_REQUEST",
                    "TOO_GOOD_TO_BE_TRUE",
                    "ANOMALOUS_TIMING",
                ],
                "reason": (
                    "Score calculation: DOMAIN_MISMATCH(30) + URL_DISCREPANCY(30) + CREDENTIAL_REQUEST(35) "
                    "+ TOO_GOOD_TO_BE_TRUE(30) + ANOMALOUS_TIMING(8) = 133, capped at 100. "
                    "RULE D triggered: DOMAIN_MISMATCH, URL_DISCREPANCY, CREDENTIAL_REQUEST, and TOO_GOOD_TO_BE_TRUE are HIGH-weight threats, "
                    "so is_fraud=true is mandatory. "
                    "A government/student-aid sender would typically use a .gov domain, but this email uses "
                    "federal-student-aid.info. It promises an unusually large grant ($5,000) and directs the recipient "
                    "to enter student credentials on an unrelated .xyz domain. The message was sent at 03:12, which is "
                    "an unusual time for official grant notifications."
                ),
                "suggestion": (
                    "Do not click the link or provide credentials. Verify grants only through official school or government "
                    "websites accessed directly (e.g., known .gov or your university portal). Report the email as phishing."
                ),
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Analyze the following email:\n'
            '{"date": "2026-02-25T14:30:00Z", "sender": "billing@amazon-business.net", '
            '"recipient": "rsun@ucmerced.edu", "copy": "accounting@amazon-business.net", '
            '"subject": "Overdue Payment for Order #AZ-99120", '
            '"content": "Hello, your business account payment for the last quarter is overdue. '
            'Please find the attached statement. To avoid service suspension, please pay the outstanding balance of $2,499.00 '
            'via our secure link: http://amazon-pay-portal.com/invoice/99120"}'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "is_fraud": True,
                "risk_score": 96,
                "confidence_level": 0.96,
                "detected_threats": [
                    "DOMAIN_MISMATCH",
                    "URL_DISCREPANCY",
                    "TOO_GOOD_TO_BE_TRUE",
                    "URGENCY_FEAR",
                    "GENERIC_SALUTATION",
                ],
                "reason": (
                    "Score calculation: DOMAIN_MISMATCH(30) + URL_DISCREPANCY(30) + TOO_GOOD_TO_BE_TRUE(30) "
                    "+ URGENCY_FEAR(15) + GENERIC_SALUTATION(8) = 113, capped at 100. "
                    "RULE D triggered: DOMAIN_MISMATCH, URL_DISCREPANCY, and TOO_GOOD_TO_BE_TRUE are HIGH-weight threats, "
                    "so is_fraud=true is mandatory. "
                    "The sender domain amazon-business.net is not an official Amazon domain (amazon.com), yet the email "
                    "demands a large payment and threatens service suspension. The payment link points to "
                    "amazon-pay-portal.com, which is also not an official Amazon domain. The greeting is generic ('Hello')."
                ),
                "suggestion": (
                    "Do not use the provided link or pay based on this email. Independently verify your Amazon Business "
                    "billing status by logging in via amazon.com directly (typed/bookmarked), and contact your finance "
                    "team or Amazon support through official channels."
                ),
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Analyze the following email:\n'
            '{"date": "2026-02-25T11:00:00Z", "sender": "professor-office@ucmerced.edu", '
            '"recipient": "rsun@ucmerced.edu", "copy": "", '
            '"subject": "Updated Syllabus and Project Files", '
            '"content": "Hi all, I\'ve updated the shared folder for our research project. '
            'Please download the new documentation from this Google Drive link: http://googledrive-shared-docs.com/d/sh1928"}'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "is_fraud": True,
                "risk_score": 46,
                "confidence_level": 0.91,
                "detected_threats": [
                    "URL_DISCREPANCY",
                    "GENERIC_SALUTATION",
                    "MISSING_SIGNATURE",
                ],
                "reason": (
                    "Score calculation: URL_DISCREPANCY(30) + GENERIC_SALUTATION(8) + MISSING_SIGNATURE(8) = 46. "
                    "RULE D triggered: URL_DISCREPANCY is a HIGH-weight threat, so is_fraud=true is mandatory even if "
                    "the sender domain looks legitimate. "
                    "Although the sender is on ucmerced.edu, the link claims to be Google Drive but points to "
                    "googledrive-shared-docs.com rather than an official Google domain (e.g., drive.google.com). "
                    "This pattern is consistent with account compromise or a spoofed/malicious redirect. The message is "
                    "not personalized ('Hi all') and lacks a professional signature for a file-sharing request."
                ),
                "suggestion": (
                    "Do not open the link. Confirm the file update via a trusted channel (course site, LMS, or a direct "
                    "message to the professor). If a shared folder exists, access it through your known Google Drive "
                    "entry points rather than third-party domains."
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
    date: datetime          # Email send timestamp
    sender: str             # Full sender email address
    recipient: str          # Primary recipient email address
    copy: Optional[str] = ""  # CC recipients (optional)
    subject: str            # Email subject line
    content: str            # Plain-text body of the email


class PredictResponse(BaseModel):
    """Structured fraud analysis result returned to the caller."""
    is_fraud: bool              # True if the model classifies the email as fraudulent
    risk_score: int             # 0-100 risk score (higher = more suspicious)
    confidence_level: float     # Model's confidence in the classification (0.0–1.0)
    detected_threats: list[str] # List of triggered threat labels
    reason: str                 # Human-readable explanation of the classification
    suggestion: str             # Recommended action for the recipient


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
        {
            "date": email.date.isoformat(),
            "sender": email.sender,
            "recipient": email.recipient,
            "copy": email.copy,
            "subject": email.subject,
            "content": email.content,
        },
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
        max_tokens=4096,
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
            # Clamp confidence_level to [0.0, 1.0].
            confidence_level=max(0.0, min(1.0, float(parsed.get("confidence_level", 0.5)))),
            detected_threats=list(parsed.get("detected_threats", [])),
            reason=str(parsed["reason"]),
            suggestion=str(parsed["suggestion"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse model output: {exc}\nRaw output: {raw_text[:500]}",
        )
