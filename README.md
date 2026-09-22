---
title: RAG Chatbot UK NHS
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: UK NHS Long-term Plan RAG chatbot
---

# GENAI RAG Chatbot

UK NHS Long-term Plan RAG chatbot (FastAPI + Streamlit) deployed as a Hugging Face Space.

## Space setup (required)

After deploying, configure these under **Space → Settings → Variables and secrets**:

| Name | Type | Value |
|------|------|--------|
| `HF_TOKEN` | Secret | User access token with **Inference Providers** permission |
| `SENTENCE_TRANSFORMERS` | Variable | `sentence-transformers/all-MiniLM-L6-v2` |
| `HF_MODEL` | Variable (optional) | Defaults to `Qwen/Qwen2.5-7B-Instruct` |

Also:

1. Enable Inference Providers at [hf.co/settings/inference-providers](https://huggingface.co/settings/inference-providers) (uses monthly HF credits).
2. Redeploy / rebuild the Space after pushing code so it picks up the Inference Providers client.

Generation uses Hugging Face Inference Providers (`InferenceClient` chat completions), not the retired `api-inference.huggingface.co` endpoint.
