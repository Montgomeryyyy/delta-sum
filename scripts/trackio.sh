# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

export GRADIO_SERVER_PORT=8000

echo "Run this command locally to port forward \`ssh -NL ${GRADIO_SERVER_PORT}:localhost:${GRADIO_SERVER_PORT} ${HOSTNAME}\`"

uv run trackio show
