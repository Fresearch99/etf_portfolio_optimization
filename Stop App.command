#!/usr/bin/env bash
cd "$(dirname "$0")"
curl -s -X POST http://127.0.0.1:8050/shutdown || true