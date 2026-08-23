#!/usr/bin/env bash
# Download your model weight file.
#
# Rules:
#   - Must be idempotent (safe to run multiple times).
#   - Must download without any credentials (public URL only).
#   - The output path must match `_runtime.model_path` in metadata.json.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
MODEL_DIR="$HERE/model"
# After the Colab merge+quantize+upload step, set HEKIMA_MODEL_URL to your Hekima GGUF
# (e.g. https://huggingface.co/coderhema/hekima-tiny-aya-q4_k_m.gguf) and rename
# MODEL_FILE below to the Hekima filename (e.g. hekima-tiny-aya-q4_k_m.gguf).
MODEL_FILE="$MODEL_DIR/tiny-aya-global-q4_k_m.gguf"

# Public Cohere Hugging Face GGUF (no auth required). Override with HEKIMA_MODEL_URL
# once Hekima is built and uploaded.
MODEL_URL="${HEKIMA_MODEL_URL:-https://huggingface.co/CohereLabs/tiny-aya-global-GGUF/resolve/main/tiny-aya-global-q4_k_m.gguf}"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
  echo "model already present at $MODEL_FILE — skipping download"
  exit 0
fi

echo "downloading $MODEL_URL → $MODEL_FILE (~2.1 GB)…"

if command -v curl > /dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$MODEL_FILE.partial" "$MODEL_URL"
elif command -v wget > /dev/null 2>&1; then
  wget --show-progress -O "$MODEL_FILE.partial" "$MODEL_URL"
else
  echo "error: neither curl nor wget found" >&2
  exit 1
fi

mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "done: $MODEL_FILE"
