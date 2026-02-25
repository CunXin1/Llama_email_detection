# Anti-Fraud Email Analysis Service

A production-ready REST API that uses a large language model (LLM) via **vLLM** to detect phishing and fraudulent emails. Built with **FastAPI** and designed to run on a single NVIDIA GPU.

---

## Features

- **Zero fine-tuning required** — few-shot prompting guides the model out of the box
- **Structured JSON output** — every response includes `is_fraud`, `risk_score`, `reason`, and `suggestion`
- **Configurable at deploy time** — all parameters are environment variables; no code changes needed
- **Production-grade** — Docker image, health check endpoint, graceful startup/shutdown via FastAPI lifespan
- **Swagger UI** included at `/docs`

---

## Prerequisites

| Requirement | Version |
|---|---|
| NVIDIA GPU | VRAM ≥ 8 GB (3B model), ≥ 16 GB (8B model) |
| NVIDIA Driver | ≥ 525 |
| Docker Desktop | Latest (WSL2 backend on Windows) |
| NVIDIA Container Toolkit | Latest |

> **Windows users**: Enable WSL2 and install [Docker Desktop](https://www.docker.com/products/docker-desktop/). All commands below run in WSL, Git Bash, or PowerShell with GNU Make (`choco install make`).

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
```

### 2. Configure environment variables

```bash
cp .env.example .env   # Linux / macOS / WSL
# or
copy .env.example .env # Windows cmd
```

Edit `.env` to set your model and GPU options (defaults work for most setups).

### 3. Build the Docker image

```bash
make build
```

### 4. Start the service

```bash
make run
```

The service exposes port `8000`. Model loading takes **1–3 minutes** on first run.

### 5. Check health

```bash
make health
# or open: http://localhost:8000/health
```

Expected response (once model is ready):
```json
{"status": "ok", "model": "NousResearch/Llama-3.2-3B-Instruct", "quantization": null, "dtype": "bfloat16"}
```

### 6. Run integration tests

```bash
make test
```

### 7. Interactive API docs

Open **http://localhost:8000/docs** in your browser.

---

## Alternative: Docker Compose

```bash
docker compose up -d        # build + start
docker compose logs -f      # tail logs
docker compose down         # stop
```

---

## API Reference

### `GET /health`

Liveness/readiness check. Returns `"status": "ok"` only after the model is fully loaded.

**Response**
```json
{
  "status": "ok",
  "model": "NousResearch/Llama-3.2-3B-Instruct",
  "quantization": null,
  "dtype": "bfloat16"
}
```

---

### `POST /predict`

Analyze an email and return a structured fraud assessment.

**Request body**
```json
{
  "sender": "security@amaz0n-verify.com",
  "subject": "[URGENT] Your account has been locked",
  "content": "Click here to verify: http://amaz0n-secure.xyz/verify"
}
```

**Response**
```json
{
  "is_fraud": true,
  "risk_score": 91,
  "reason": "The sender domain typosquats amazon.com. Urgency language and a non-official phishing link are present.",
  "suggestion": "Do not click any links. Report as phishing and delete the email."
}
```

**Risk score guide**

| Score | Level | Meaning |
|---|---|---|
| 0 – 30 | Low | Legitimate email |
| 31 – 60 | Medium | Exercise caution |
| 61 – 100 | High | Likely fraudulent |

---

## Configuration

All options are set via environment variables (or `.env` file for Docker Compose / `make run`).

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `NousResearch/Llama-3.2-3B-Instruct` | HuggingFace repo ID or local path |
| `MAX_MODEL_LEN` | `4096` | Max context length (prompt + output) |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM for KV cache |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs (multi-GPU setups) |
| `QUANTIZATION` | _(none)_ | `awq` or `gptq` (must match model) |
| `DTYPE` | `bfloat16` | `bfloat16` (Ampere+) or `float16` |
| `PORT` | `8000` | Host port |
| `HF_TOKEN` | _(none)_ | HuggingFace token for gated models |

### Switching models

**Smaller / faster (3B, ~6 GB VRAM)**
```bash
make run MODEL_PATH=NousResearch/Llama-3.2-3B-Instruct
```

**Better accuracy (8B, ~16 GB VRAM) — requires HF token**
```bash
make run MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=hf_xxx
```

**Using a locally downloaded model**
```bash
# Download first
make hf-download MODEL_PATH=NousResearch/Llama-3.2-3B-Instruct MODEL_DIR=/d/models

# Run with local mount
make run-local MODEL_DIR=/d/models/Llama-3.2-3B-Instruct
```

---

## Makefile Reference

```
make build          Build Docker image
make run            Start container (GPU, auto-download model from HuggingFace)
make run-local      Start container (GPU, mount local MODEL_DIR)
make stop           Stop and remove container
make restart        stop + run
make logs           Tail container logs
make test           Run integration tests
make health         Quick /health check
make clean          Remove container + image
make hf-download    Download model weights via huggingface-cli
make push           Push image to Docker Hub
make git-init       Initialize git repo and create first commit
make help           Show all targets
```

---

## Project Structure

```
.
├── main.py               # FastAPI application + vLLM inference logic
├── test_request.py       # Integration tests
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Compose file for local development
├── requirements.txt      # Python dependencies
├── Makefile              # Build / run / deploy shortcuts
├── .env.example          # Environment variable template
└── .gitignore
```

---

## Uploading to GitHub

```bash
# 1. Initialize and commit
make git-init

# 2. Create a new repo at https://github.com/new (leave it empty, no README)

# 3. Add remote and push
git remote add origin https://github.com/<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```

---

## License

MIT
