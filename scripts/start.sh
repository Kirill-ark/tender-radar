#!/usr/bin/env bash
# Start everything: the tunnel first, then n8n with the tunnel's public address.
# A quick tunnel gets a new random address on every start, and n8n must know it
# to register the Telegram webhook, so the address is read from the tunnel log.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose up -d --build service tunnel

echo "Waiting for the tunnel address..."
url=""
for _ in $(seq 1 30); do
  url=$(docker compose logs tunnel 2>&1 | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1 || true)
  [ -n "$url" ] && break
  sleep 2
done
[ -n "$url" ] || { echo "Tunnel did not report an address"; exit 1; }

WEBHOOK_URL="$url" docker compose up -d n8n
echo "n8n:            http://localhost:5678"
echo "public address: $url"
