# Anti-Fraud Email Analysis Service

A REST API that uses a large language model (via **vLLM**) to detect phishing and fraudulent emails. Built with **FastAPI** and runs on a single NVIDIA GPU.

Each analyzed email returns a structured JSON response with:
- `is_fraud` — whether the email is fraudulent
- `risk_score` — 0–100 risk score (higher = more suspicious)
- `confidence_level` — model confidence (0.0–1.0)
- `detected_threats` — list of triggered threat labels (e.g. `DOMAIN_MISMATCH`, `CREDENTIAL_REQUEST`)
- `reason` — explanation of the classification
- `suggestion` — recommended action

---

## Requirements

- NVIDIA GPU with ≥ 8 GB VRAM
- NVIDIA Driver ≥ 525
- Docker with NVIDIA Container Toolkit

---

## Running with Docker Compose

```bash
# Start the service (builds image on first run, model loads in ~1–3 min)
docker compose up -d

# Show the log
docker compose logs -f

# Check if the model is ready
curl http://localhost:8000/health

# Stop the service
docker compose down
```

---

## Calling the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-02-25T10:00:00Z",
    "sender": "security@amaz0n-verify.com",
    "recipient": "you@example.com",
    "subject": "URGENT: Your account has been locked",
    "content": "Click here to verify: http://amaz0n-secure.xyz/verify"
  }'
```

**Example response:**
```json
{
  "is_fraud": true,
  "risk_score": 95,
  "confidence_level": 0.97,
  "detected_threats": ["DOMAIN_MISMATCH", "URL_DISCREPANCY", "URGENCY_FEAR"],
  "reason": "The sender domain typosquats amazon.com. Urgency language and a non-official phishing link are present.",
  "suggestion": "Do not click any links. Report as phishing and delete the email."
}
```

Interactive API docs are available at **http://localhost:8000/docs**.

---

## Configuration

Edit `.env` to change settings before starting:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `NousResearch/Llama-3.2-3B-Instruct` | HuggingFace model ID or local path |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM for model cache |
| `MAX_MODEL_LEN` | `4096` | Max context length (tokens) |
| `QUANTIZATION` | _(none)_ | `awq` or `gptq` if using a quantized model |
| `DTYPE` | `bfloat16` | `bfloat16` (Ampere+) or `float16` |
| `PORT` | `8000` | Host port |
| `HF_TOKEN` | _(none)_ | HuggingFace token for gated models |

---

## License

MIT
