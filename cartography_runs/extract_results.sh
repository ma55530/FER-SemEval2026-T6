#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="cartography_results_${TIMESTAMP}.tar.gz"

mkdir -p output/logs

if docker inspect cartography-full >/dev/null 2>&1; then
  docker cp cartography-full:/app/output ./output
  docker logs cartography-full > output/logs/cartography-full.log 2>&1 || true
fi

if docker inspect cartography-clarity >/dev/null 2>&1; then
  docker cp cartography-clarity:/app/output ./output
  docker logs cartography-clarity > output/logs/cartography-clarity.log 2>&1 || true
fi

if [ ! -d "output" ]; then
  echo "output/ not found"
  exit 1
fi

tar -czf "$ARCHIVE" output/
echo "Created $ARCHIVE"
