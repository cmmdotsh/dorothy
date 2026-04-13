#!/usr/bin/env bash
# Setup script for running Gemma 4 31B via mlx-lm on macstudio.local.
#
# Run this ON the Mac Studio:
#   ssh cmur@macstudio.local
#   bash scripts/setup_mlx_gemma.sh
#
# This installs mlx-lm, downloads the model, creates a launchd service,
# and starts it on port 8081.

set -euo pipefail

MODEL="mlx-community/gemma-4-31b-it-8bit"
PORT=8081
HOST="0.0.0.0"
MAX_TOKENS=4096
PROMPT_CACHE_BYTES=8589934592  # 8 GB KV cache limit
VENV_DIR="$HOME/.mlx-gemma"
PYTHON="/opt/homebrew/bin/python3.13"
PLIST_NAME="com.dorothy.mlx-gemma"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
LOG_DIR="$HOME/Library/Logs/mlx-gemma"

echo "=== Dorothy MLX Gemma 4 Setup ==="
echo "Model:  $MODEL"
echo "Port:   $PORT"
echo "Venv:   $VENV_DIR"
echo ""

# ── 1. Create venv & install mlx-lm ──────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

echo "Installing mlx-lm..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install --upgrade mlx-lm

# ── 2. Pre-download the model ────────────────────────────────────────────────
echo ""
echo "Downloading model (this may take a while on first run)..."
"$VENV_DIR/bin/python" -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download('${MODEL}')
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(path) for f in fns)
print(f'Model ready at: {path} ({total / 1024**3:.1f} GB)')
"

# ── 3. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "Running quick generation smoke test..."
"$VENV_DIR/bin/python" -c "
from mlx_lm import load, generate
model, tokenizer = load('${MODEL}')
result = generate(model, tokenizer, prompt='Say hello in one sentence.', max_tokens=32)
print(f'Smoke test output: {result}')
del model, tokenizer
"

# ── 4. Kill existing nohup process if running ────────────────────────────────
echo ""
echo "Stopping any existing mlx-lm processes..."
pkill -f "mlx_lm server" 2>/dev/null || true
pkill -f "mlx_lm.server" 2>/dev/null || true
sleep 1

# ── 5. Create launchd plist ──────────────────────────────────────────────────
echo "Creating launchd service at $PLIST_PATH..."
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$PLIST_PATH")"

cat > "$PLIST_PATH" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${VENV_DIR}/bin/python</string>
        <string>-m</string>
        <string>mlx_lm</string>
        <string>server</string>
        <string>--model</string>
        <string>${MODEL}</string>
        <string>--host</string>
        <string>${HOST}</string>
        <string>--port</string>
        <string>${PORT}</string>
        <string>--max-tokens</string>
        <string>${MAX_TOKENS}</string>
        <string>--prompt-cache-bytes</string>
        <string>${PROMPT_CACHE_BYTES}</string>
        <string>--pipeline</string>
        <string>--chat-template-args</string>
        <string>{"enable_thinking": false}</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/stderr.log</string>

    <key>ProcessType</key>
    <string>Interactive</string>
</dict>
</plist>
PLIST

# ── 6. Load the service ─────────────────────────────────────────────────────
echo "Loading launchd service..."

# Unload if already loaded (ignore errors)
launchctl bootout "gui/$(id -u)/${PLIST_NAME}" 2>/dev/null || true
sleep 1

launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"

echo ""
echo "Waiting for server to start..."
for i in $(seq 1 90); do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "Server is up on port $PORT after ${i}s!"
        curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool
        break
    fi
    if [ "$i" -eq 90 ]; then
        echo "Server did not start within 90s. Check logs:"
        echo "  tail -f $LOG_DIR/stderr.log"
        exit 1
    fi
    sleep 1
done

# ── 7. Verify with a chat completion ─────────────────────────────────────────
echo ""
echo "Testing /v1/chat/completions..."
RESPONSE=$(curl -s --max-time 120 "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mlx-community/gemma-4-31b-it-8bit",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 32,
        "temperature": 0.3
    }')

echo "$RESPONSE" | python3 -m json.tool

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Service:  $PLIST_NAME"
echo "Endpoint: http://$(hostname):${PORT}/v1/chat/completions"
echo "Logs:     tail -f $LOG_DIR/stderr.log"
echo ""
echo "Management commands:"
echo "  Stop:    launchctl bootout gui/\$(id -u)/${PLIST_NAME}"
echo "  Start:   launchctl bootstrap gui/\$(id -u) $PLIST_PATH"
echo "  Status:  launchctl print gui/\$(id -u)/${PLIST_NAME}"
echo ""
echo "Dorothy config (set as env vars or in .env):"
echo "  REVIEWER_BASE_URL=http://macstudio.local:${PORT}"
echo "  REVIEWER_MODEL=mlx-community/gemma-4-31b-it-8bit"
